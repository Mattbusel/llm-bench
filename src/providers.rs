//! # Module: Providers
//!
//! Issue one inference request to an OpenAI-compatible Chat Completions
//! endpoint or the Anthropic Messages API, measure timing, read token usage
//! and price the request.
//!
//! Both calls can stream (server-sent events, parsed with
//! `eventsource-stream`), which gives a real time to first token. Timing is
//! measured with `std::time::Instant` (monotonic). Nothing here retries;
//! see `runner.rs`.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use eventsource_stream::Eventsource;
use futures::StreamExt;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use reqwest::{Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{debug, instrument};

use crate::error::BenchError;
use crate::types::{BenchResult, CostSource, ProviderConfig};

//  Pricing

/// A model with a list price, in the units llm-bench used before 0.4
/// (USD per 1 000 tokens). Built from the `llm-prices` table.
pub struct ModelInfo {
    /// Provider that serves the model (`openai`, `anthropic`, `gemini`,
    /// `deepseek`), as used in `--models`.
    pub provider: &'static str,
    /// Model id as sent to the API.
    pub model: &'static str,
    /// USD per 1 000 prompt tokens.
    pub prompt_per_1k: f64,
    /// USD per 1 000 completion tokens.
    pub completion_per_1k: f64,
}

/// Ids that `llm-prices` does not resolve on its own, mapped to the row
/// they are billed as.
const ALIASES: &[(&str, &str)] = &[
    ("gpt-4-turbo", "gpt-4-turbo-2024-04-09"),
    ("gpt-4-turbo-preview", "gpt-4-turbo-2024-04-09"),
];

/// Providers whose models are billed at the first-party list price. Models
/// served by Groq, Together, Fireworks, Ollama or a custom URL are billed
/// at that host's own price, which only a `[prices]` entry in `--config`
/// can supply, so they are not looked up here.
pub const LIST_PRICED_PROVIDERS: &[&str] =
    &["openai", "anthropic", "gemini", "deepseek", "openrouter"];

/// The list price of `model` from the [`llm_prices`] crate (dated,
/// source-cited, USD per 1M tokens). Accepts dated ids, provider prefixes
/// (`openai/gpt-4o-mini`, as OpenRouter names models) and `gpt-4-turbo`.
/// `None` for unknown models and for rows without both an input and an
/// output price (embeddings).
pub fn list_price(model: &str) -> Option<&'static llm_prices::Price> {
    let id = ALIASES
        .iter()
        .find(|(alias, _)| alias.eq_ignore_ascii_case(model.trim()))
        .map(|(_, target)| *target)
        .unwrap_or(model);
    llm_prices::lookup(id).filter(|p| p.input.is_some() && p.output.is_some())
}

/// Whether `model` has a list price (see [`list_price`]).
pub fn has_known_price(model: &str) -> bool {
    list_price(model).is_some()
}

/// Token counts of one response, as priced.
#[derive(Debug, Clone, Copy, Default)]
struct TokenUsage {
    /// Prompt tokens as reported (OpenAI-compatible: cached ones included;
    /// Anthropic: the uncached part only).
    prompt: u32,
    completion: u32,
    /// Prompt tokens read from the prompt cache.
    cached: u32,
    /// Prompt tokens written to the cache (Anthropic).
    cache_write: u32,
}

/// Cost of one request and where it came from. The provider's own figure
/// wins when the response carries one (OpenRouter reports `usage.cost`).
/// Otherwise providers in [`LIST_PRICED_PROVIDERS`] are billed at the
/// `llm-prices` list price, with cached prompt tokens at the cache price.
/// Anything else costs 0.0 with no source, shown as "n/a" (a `[prices]`
/// entry in `--config` is applied later, by the runner).
fn request_cost(
    config: &ProviderConfig,
    usage: TokenUsage,
    reported: Option<f64>,
) -> (f64, Option<CostSource>) {
    if let Some(c) = reported.filter(|c| c.is_finite() && *c >= 0.0) {
        return (c, Some(CostSource::Provider));
    }
    if !LIST_PRICED_PROVIDERS.contains(&config.name.as_str()) {
        return (0.0, None);
    }
    let Some(price) = list_price(&config.model) else {
        return (0.0, None);
    };
    // Anthropic reports the uncached input separately; OpenAI-compatible
    // APIs include cached tokens in prompt_tokens.
    let uncached = if config.name == "anthropic" {
        usage.prompt
    } else {
        usage.prompt.saturating_sub(usage.cached)
    };
    let detailed = llm_prices::Usage {
        input: u64::from(uncached),
        output: u64::from(usage.completion),
        cache_read: u64::from(usage.cached),
        cache_write_5m: u64::from(usage.cache_write),
        ..llm_prices::Usage::default()
    };
    let cost = price
        .cost_usage(&detailed)
        .map(|c| c.total)
        .or_else(|| price.cost(u64::from(usage.prompt), u64::from(usage.completion)));
    match cost {
        Some(c) => (c, Some(CostSource::LlmPrices)),
        None => (0.0, None),
    }
}

/// USD cost of a request at the `llm-prices` list price of `model` (no
/// cached tokens), or 0.0 for a model without one.
pub fn compute_cost(model: &str, prompt_tokens: u32, completion_tokens: u32) -> f64 {
    list_price(model)
        .and_then(|p| p.cost(u64::from(prompt_tokens), u64::from(completion_tokens)))
        .unwrap_or(0.0)
}

/// Every listed model with an input and output price in `llm-prices`,
/// sorted by provider (`anthropic`, `deepseek`, `gemini`, `openai`) and model.
pub fn supported_models() -> &'static [ModelInfo] {
    static MODELS: OnceLock<Vec<ModelInfo>> = OnceLock::new();
    MODELS.get_or_init(|| {
        let mut v: Vec<ModelInfo> = llm_prices::all()
            .iter()
            .filter(|p| p.status == "listed")
            .filter_map(|p| {
                Some(ModelInfo {
                    provider: if p.provider == "google" { "gemini" } else { p.provider },
                    model: p.model,
                    prompt_per_1k: p.input? / 1000.0,
                    completion_per_1k: p.output? / 1000.0,
                })
            })
            .collect();
        v.sort_by(|a, b| a.provider.cmp(b.provider).then(a.model.cmp(b.model)));
        v
    })
}

/// Date the bundled `llm-prices` table was generated (ISO 8601).
pub fn prices_date() -> &'static str {
    llm_prices::GENERATED
}

//  Wire types

#[derive(Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize, Debug)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize, Debug)]
struct OpenAiChoice {
    message: OpenAiResponseMessage,
}

#[derive(Deserialize, Debug)]
struct OpenAiResponseMessage {
    content: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    /// USD, reported by OpenRouter.
    #[serde(default)]
    cost: Option<f64>,
    /// OpenAI (and most compatible servers): cached prompt tokens.
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
    /// DeepSeek: prompt tokens served from its context cache.
    #[serde(default)]
    prompt_cache_hit_tokens: Option<u32>,
}

#[derive(Deserialize, Debug, Default)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u32>,
}

impl OpenAiUsage {
    fn tokens(&self) -> TokenUsage {
        let cached = self
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .or(self.prompt_cache_hit_tokens)
            .unwrap_or(0)
            .min(self.prompt_tokens);
        TokenUsage {
            prompt: self.prompt_tokens,
            completion: self.completion_tokens,
            cached,
            cache_write: 0,
        }
    }
}

#[derive(Deserialize, Debug)]
struct OpenAiChunk {
    #[serde(default)]
    choices: Vec<OpenAiChunkChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize, Debug)]
struct OpenAiChunkChoice {
    #[serde(default)]
    delta: OpenAiDelta,
}

#[derive(Deserialize, Debug, Default)]
struct OpenAiDelta {
    #[serde(default)]
    content: Option<String>,
}

#[derive(Deserialize, Debug)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    usage: Option<AnthropicUsage>,
}

#[derive(Deserialize, Debug)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
    #[serde(default)]
    cache_read_input_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: u32,
}

impl AnthropicUsage {
    fn tokens(&self) -> TokenUsage {
        TokenUsage {
            prompt: self.input_tokens,
            completion: self.output_tokens,
            cached: self.cache_read_input_tokens,
            cache_write: self.cache_creation_input_tokens,
        }
    }
}

//  Shared helpers

/// The Chat Completions URL for an OpenAI-compatible base URL. Bases that
/// already end in a version segment (`.../v1beta/openai` for Gemini,
/// `.../v1`) get `/chat/completions`; bare hosts get `/v1/chat/completions`.
pub fn chat_completions_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    let after_scheme = base.split_once("://").map_or(base, |(_, rest)| rest);
    // Only look at the path, never the host name.
    let last = match after_scheme.split_once('/') {
        Some((_, path)) => path.rsplit('/').next().unwrap_or(""),
        None => "",
    };
    let is_version = |seg: &str| {
        seg.strip_prefix('v').is_some_and(|rest| {
            rest.starts_with(|c: char| c.is_ascii_digit())
                && rest.chars().all(|c| c.is_ascii_alphanumeric())
        })
    };
    let versioned = last == "openai" || is_version(last);
    if versioned {
        format!("{base}/chat/completions")
    } else {
        format!("{base}/v1/chat/completions")
    }
}

fn retry_after(headers: &HeaderMap) -> Option<u64> {
    let v = headers.get(RETRY_AFTER)?.to_str().ok()?.trim();
    if let Ok(secs) = v.parse::<u64>() {
        return Some(secs);
    }
    let when = httpdate::parse_http_date(v).ok()?;
    Some(
        when.duration_since(std::time::SystemTime::now())
            .unwrap_or(Duration::ZERO)
            .as_secs(),
    )
}

/// Turn a non-2xx response into the matching error.
async fn status_error(resp: Response, provider: &str, model: &str) -> BenchError {
    let status = resp.status().as_u16();
    if status == 429 {
        return BenchError::RateLimited {
            provider: provider.into(),
            retry_after_secs: retry_after(resp.headers()),
        };
    }
    let body = resp.text().await.unwrap_or_default();
    BenchError::ApiError {
        provider: provider.into(),
        model: model.into(),
        status,
        body,
    }
}

fn tokens_per_second(completion_tokens: u32, total_ms: u64) -> f64 {
    if total_ms > 0 {
        f64::from(completion_tokens) / (total_ms as f64 / 1000.0)
    } else {
        0.0
    }
}

/// Measured outcome of one call, before pricing.
struct Measured {
    text: String,
    usage: TokenUsage,
    reported_cost: Option<f64>,
    total_ms: u64,
    ttft_ms: Option<u64>,
}

fn finish(config: &ProviderConfig, prompt: &str, run_idx: u32, m: Measured) -> BenchResult {
    let (cost_usd, cost_source) = request_cost(config, m.usage, m.reported_cost);
    let (prompt_tokens, completion_tokens) = (m.usage.prompt, m.usage.completion);
    debug!(
        provider = %config.name,
        model = %config.model,
        total_ms = m.total_ms,
        ttft_ms = ?m.ttft_ms,
        prompt_tokens,
        completion_tokens,
        "call complete"
    );
    BenchResult {
        provider: config.name.clone(),
        model: config.model.clone(),
        prompt: prompt.to_owned(),
        latency_ms: m.ttft_ms.unwrap_or(m.total_ms),
        total_ms: m.total_ms,
        prompt_tokens,
        completion_tokens,
        cost_usd,
        tokens_per_second: tokens_per_second(completion_tokens, m.total_ms),
        response_text: m.text,
        run_index: run_idx,
        ttft_ms: m.ttft_ms,
        retries: 0,
        tpot_ms: match m.ttft_ms {
            Some(t) if completion_tokens > 1 && m.total_ms >= t => {
                Some((m.total_ms - t) as f64 / f64::from(completion_tokens - 1))
            }
            _ => None,
        },
        cached_prompt_tokens: m.usage.cached,
        cache_write_prompt_tokens: m.usage.cache_write,
        cost_source,
    }
}

fn stream_error(provider: &str, model: &str, msg: impl Into<String>) -> BenchError {
    BenchError::ApiError {
        provider: provider.into(),
        model: model.into(),
        status: 200,
        body: msg.into(),
    }
}

//  OpenAI-compatible provider

/// Issue one request to an OpenAI-compatible Chat Completions endpoint.
///
/// `base_url` is the server root (`https://api.openai.com`,
/// `http://localhost:11434`) or a versioned base such as Gemini's
/// `https://generativelanguage.googleapis.com/v1beta/openai`; see
/// [`chat_completions_url`]. With `stream`, the answer is read as
/// server-sent events and the time to the first text is recorded.
///
/// # Panics
/// This function never panics.
#[instrument(skip(client, config), fields(model = %config.model, run = run_idx))]
pub async fn run_openai(
    client: &Client,
    config: &ProviderConfig,
    base_url: &str,
    prompt: &str,
    run_idx: u32,
    stream: bool,
) -> Result<BenchResult, BenchError> {
    let url = chat_completions_url(base_url);
    let mut body = json!({
        "model": config.model,
        "messages": [ChatMessage { role: "user", content: prompt }],
    });
    // OpenAI's reasoning models (o1, o3, o4-mini, ...) reject `max_tokens`
    // with HTTP 400; the official API takes `max_completion_tokens` for every
    // chat model. Other servers (Ollama, vLLM, OpenRouter, Gemini) keep
    // `max_tokens`.
    let limit_field = if config.name == "openai" && config.base_url.is_empty() {
        "max_completion_tokens"
    } else {
        "max_tokens"
    };
    body[limit_field] = json!(config.max_tokens);
    if stream {
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage": true});
    }
    if config.name == "openrouter" {
        // Ask OpenRouter to report the request's real cost in `usage.cost`.
        body["usage"] = json!({"include": true});
    }

    let mut req = client.post(&url).json(&body);
    if !config.api_key.is_empty() {
        req = req.bearer_auth(&config.api_key);
    }
    let start = Instant::now();
    let resp = req.send().await?;
    if !resp.status().is_success() {
        return Err(status_error(resp, &config.name, &config.model).await);
    }

    let measured = if stream {
        let mut events = resp.bytes_stream().eventsource();
        let mut text = String::new();
        let mut ttft: Option<u64> = None;
        let mut usage = OpenAiUsage::default();
        while let Some(ev) = events.next().await {
            let ev = ev.map_err(|e| stream_error(&config.name, &config.model, e.to_string()))?;
            let data = ev.data.trim();
            if data == "[DONE]" {
                break;
            }
            if data.is_empty() {
                continue;
            }
            let v: Value = serde_json::from_str(data)?;
            if let Some(err) = v.get("error") {
                return Err(stream_error(&config.name, &config.model, err.to_string()));
            }
            let chunk: OpenAiChunk = serde_json::from_value(v)?;
            for c in chunk.choices {
                if let Some(t) = c.delta.content.filter(|t| !t.is_empty()) {
                    if ttft.is_none() {
                        ttft = Some(start.elapsed().as_millis() as u64);
                    }
                    text.push_str(&t);
                }
            }
            if let Some(u) = chunk.usage {
                usage = u;
            }
        }
        Measured {
            text,
            usage: usage.tokens(),
            reported_cost: usage.cost,
            total_ms: start.elapsed().as_millis() as u64,
            ttft_ms: ttft,
        }
    } else {
        let total_ms = start.elapsed().as_millis() as u64;
        let parsed: OpenAiResponse = resp.json().await?;
        let usage = parsed.usage.unwrap_or_default();
        Measured {
            text: parsed
                .choices
                .into_iter()
                .next()
                .and_then(|c| c.message.content)
                .unwrap_or_default(),
            usage: usage.tokens(),
            reported_cost: usage.cost,
            total_ms,
            ttft_ms: None,
        }
    };
    Ok(finish(config, prompt, run_idx, measured))
}

//  Anthropic provider

/// Issue one request to the Anthropic Messages API (`{base_url}/v1/messages`).
/// With `stream`, the answer is read as server-sent events and the time to
/// the first text delta is recorded.
///
/// # Panics
/// This function never panics.
#[instrument(skip(client, config), fields(model = %config.model, run = run_idx))]
pub async fn run_anthropic(
    client: &Client,
    config: &ProviderConfig,
    base_url: &str,
    prompt: &str,
    run_idx: u32,
    stream: bool,
) -> Result<BenchResult, BenchError> {
    let url = format!("{}/v1/messages", base_url.trim_end_matches('/'));
    let mut body = json!({
        "model": config.model,
        "max_tokens": config.max_tokens,
        "messages": [ChatMessage { role: "user", content: prompt }],
    });
    if stream {
        body["stream"] = json!(true);
    }

    let mut req = client
        .post(&url)
        .header("anthropic-version", "2023-06-01")
        .json(&body);
    if !config.api_key.is_empty() {
        req = req.header("x-api-key", &config.api_key);
    }
    let start = Instant::now();
    let resp = req.send().await?;
    if !resp.status().is_success() {
        return Err(status_error(resp, "anthropic", &config.model).await);
    }

    let measured = if stream {
        let mut events = resp.bytes_stream().eventsource();
        let mut text = String::new();
        let mut ttft: Option<u64> = None;
        let mut usage = AnthropicUsage::default();
        while let Some(ev) = events.next().await {
            let ev = ev.map_err(|e| stream_error("anthropic", &config.model, e.to_string()))?;
            if ev.data.trim().is_empty() {
                continue;
            }
            let v: Value = serde_json::from_str(&ev.data)?;
            match v.get("type").and_then(Value::as_str).unwrap_or(&ev.event) {
                "message_start" => {
                    if let Some(u) = v.pointer("/message/usage") {
                        let u: AnthropicUsage = serde_json::from_value(u.clone())?;
                        usage.input_tokens = u.input_tokens;
                        usage.output_tokens = usage.output_tokens.max(u.output_tokens);
                        usage.cache_read_input_tokens = u.cache_read_input_tokens;
                        usage.cache_creation_input_tokens = u.cache_creation_input_tokens;
                    }
                }
                "content_block_delta" => {
                    if let Some(t) = v
                        .pointer("/delta/text")
                        .and_then(Value::as_str)
                        .filter(|t| !t.is_empty())
                    {
                        if ttft.is_none() {
                            ttft = Some(start.elapsed().as_millis() as u64);
                        }
                        text.push_str(t);
                    }
                }
                "message_delta" => {
                    if let Some(n) = v.pointer("/usage/output_tokens").and_then(Value::as_u64) {
                        usage.output_tokens = u32::try_from(n).unwrap_or(u32::MAX);
                    }
                }
                "message_stop" => break,
                "error" => {
                    return Err(stream_error("anthropic", &config.model, v.to_string()));
                }
                _ => {}
            }
        }
        Measured {
            text,
            usage: usage.tokens(),
            reported_cost: None,
            total_ms: start.elapsed().as_millis() as u64,
            ttft_ms: ttft,
        }
    } else {
        let total_ms = start.elapsed().as_millis() as u64;
        let parsed: AnthropicResponse = resp.json().await?;
        let usage = parsed.usage.unwrap_or_default();
        Measured {
            text: parsed
                .content
                .into_iter()
                .filter(|b| b.block_type == "text")
                .filter_map(|b| b.text)
                .collect::<Vec<_>>()
                .join(""),
            usage: usage.tokens(),
            reported_cost: None,
            total_ms,
            ttft_ms: None,
        }
    };
    Ok(finish(config, prompt, run_idx, measured))
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    //  Cost calculation

    #[test]
    fn test_compute_cost_gpt4o_mini_known_values() {
        // 1000 prompt + 1000 completion tokens at known rates
        let cost = compute_cost("gpt-4o-mini", 1000, 1000);
        let expected = 0.000150 + 0.000600; // 0.00075
        assert!(
            (cost - expected).abs() < 1e-9,
            "expected {expected}, got {cost}"
        );
    }

    #[test]
    fn test_compute_cost_gpt4o_known_values() {
        let cost = compute_cost("gpt-4o", 1000, 1000);
        let expected = 0.0025 + 0.010; // $2.50 / $10 per million
        assert!((cost - expected).abs() < 1e-9);
    }

    #[test]
    fn test_compute_cost_claude_haiku_known_values() {
        let cost = compute_cost("claude-haiku-4-5", 1000, 1000);
        let expected = 0.001 + 0.005; // 0.006
        assert!((cost - expected).abs() < 1e-9);
    }

    #[test]
    fn test_compute_cost_claude_opus_known_values() {
        let cost = compute_cost("claude-opus-5", 1000, 1000);
        let expected = 0.005 + 0.025;
        assert!((cost - expected).abs() < 1e-9);
    }

    #[test]
    fn test_compute_cost_unknown_model_is_unpriced() {
        // 0.2 charged a made-up $0.002 per 1k tokens here.
        assert_eq!(compute_cost("unknown-model-xyz", 1000, 1000), 0.0);
    }

    #[test]
    fn test_compute_cost_zero_tokens_returns_zero() {
        let cost = compute_cost("gpt-4o", 0, 0);
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_compute_cost_only_prompt_tokens() {
        let cost = compute_cost("gpt-4o", 1000, 0);
        let expected = 0.0025;
        assert!((cost - expected).abs() < 1e-9);
    }

    #[test]
    fn test_compute_cost_only_completion_tokens() {
        let cost = compute_cost("gpt-4o", 0, 1000);
        let expected = 0.010;
        assert!((cost - expected).abs() < 1e-9);
    }

    #[test]
    fn test_compute_cost_scales_linearly() {
        let cost_1k = compute_cost("gpt-4o-mini", 1000, 1000);
        let cost_2k = compute_cost("gpt-4o-mini", 2000, 2000);
        assert!(
            (cost_2k - 2.0 * cost_1k).abs() < 1e-9,
            "cost should scale linearly"
        );
    }

    //  List prices (llm-prices)

    fn per_1m(model: &str) -> (f64, f64) {
        let p = list_price(model).unwrap_or_else(|| panic!("{model} has no list price"));
        (p.input.unwrap_or(0.0), p.output.unwrap_or(0.0))
    }

    #[test]
    fn test_list_price_gpt4_turbo_alias() {
        assert_eq!(per_1m("gpt-4-turbo"), (10.0, 30.0));
        assert_eq!(per_1m("gpt-4-turbo-preview"), (10.0, 30.0));
    }

    #[test]
    fn test_list_price_claude_sonnet() {
        assert_eq!(per_1m("claude-sonnet-5"), (2.0, 10.0));
    }

    #[test]
    fn test_list_price_dated_alias_equals_latest() {
        assert_eq!(per_1m("claude-haiku-4-5-20251001"), per_1m("claude-haiku-4-5"));
    }

    #[test]
    fn test_list_price_openrouter_style_ids_and_deepseek() {
        assert_eq!(per_1m("openai/gpt-4o-mini"), per_1m("gpt-4o-mini"));
        assert!(has_known_price("deepseek-v4-flash"));
        // Embeddings have no output price, so no request cost.
        assert!(!has_known_price("text-embedding-3-small"));
    }

    //  supported_models

    #[test]
    fn test_supported_models_non_empty() {
        assert!(!supported_models().is_empty());
    }

    #[test]
    fn test_supported_models_all_have_non_negative_prices() {
        for m in supported_models() {
            assert!(m.prompt_per_1k >= 0.0, "model {}", m.model);
            assert!(m.completion_per_1k > 0.0, "model {}", m.model);
        }
    }

    #[test]
    fn test_supported_models_providers_are_known() {
        for m in supported_models() {
            assert!(
                ["openai", "anthropic", "gemini", "deepseek"].contains(&m.provider),
                "unknown provider: {}",
                m.provider
            );
        }
        assert!(supported_models().iter().any(|m| m.provider == "gemini"));
        assert!(supported_models().iter().any(|m| m.provider == "deepseek"));
        assert_eq!(prices_date().len(), 10);
    }

    #[test]
    fn test_supported_models_contains_gpt4o_mini() {
        let found = supported_models().iter().any(|m| m.model == "gpt-4o-mini");
        assert!(found, "gpt-4o-mini should be in supported models");
    }

    #[test]
    fn test_supported_models_contains_claude_haiku() {
        let found = supported_models()
            .iter()
            .any(|m| m.model == "claude-haiku-4-5");
        assert!(found, "haiku should be in supported models");
    }

    #[test]
    fn test_has_known_price() {
        assert!(has_known_price("gpt-4o-mini"));
        assert!(has_known_price("claude-haiku-4-5-20251001"));
        assert!(!has_known_price("qwen2.5:1.5b"));
    }

    fn tokens(prompt: u32, completion: u32) -> TokenUsage {
        TokenUsage {
            prompt,
            completion,
            ..TokenUsage::default()
        }
    }

    #[test]
    fn test_request_cost_unpriced_model_is_zero_and_reported_cost_wins() {
        let mut c = openai_config("", "qwen2.5:1.5b");
        c.base_url = "http://localhost:11434".into();
        assert_eq!(request_cost(&c, tokens(100, 100), None), (0.0, None));
        c.base_url = String::new();
        assert_eq!(request_cost(&c, tokens(100, 100), None), (0.0, None));
        // A priced model through a proxy is still priced.
        let mut p = openai_config("", "gpt-4o-mini");
        p.base_url = "http://proxy:8080".into();
        let (cost, source) = request_cost(&p, tokens(1000, 1000), None);
        assert!((cost - 0.00075).abs() < 1e-12, "{cost}");
        assert_eq!(source, Some(CostSource::LlmPrices));
        // A cost reported by the provider (OpenRouter) beats the table.
        assert_eq!(
            request_cost(&p, tokens(1000, 1000), Some(0.0123)),
            (0.0123, Some(CostSource::Provider))
        );
        assert!(request_cost(&p, tokens(1000, 1000), Some(f64::NAN)).0 > 0.0);
    }

    #[test]
    fn test_request_cost_hosts_without_list_prices_are_unpriced() {
        // gpt-4o-mini served by Groq or a custom provider is not billed at
        // OpenAI's price; a [prices] entry has to say what it costs.
        for name in ["groq", "together", "fireworks", "ollama", "my-vllm"] {
            let mut c = openai_config("", "gpt-4o-mini");
            c.name = name.into();
            assert_eq!(request_cost(&c, tokens(1000, 1000), None), (0.0, None), "{name}");
        }
        let mut d = openai_config("", "deepseek-v4-flash");
        d.name = "deepseek".into();
        assert!(request_cost(&d, tokens(1000, 1000), None).0 > 0.0);
    }

    #[test]
    fn test_request_cost_bills_cached_prompt_tokens_at_cache_price() {
        // gpt-4o-mini: $0.15 input, $0.075 cached input, $0.60 output per 1M.
        let c = openai_config("", "gpt-4o-mini");
        let usage = TokenUsage {
            prompt: 2_000_000,
            completion: 0,
            cached: 1_000_000,
            cache_write: 0,
        };
        let (cost, _) = request_cost(&c, usage, None);
        assert!((cost - (0.15 + 0.075)).abs() < 1e-9, "{cost}");
        // Anthropic reports the uncached input on its own: claude-haiku-4-5
        // at $1 input, $0.10 cache read, $1.25 5-minute cache write per 1M.
        let a = anthropic_config("", "claude-haiku-4-5");
        let usage = TokenUsage {
            prompt: 1_000_000,
            completion: 0,
            cached: 1_000_000,
            cache_write: 1_000_000,
        };
        let (cost, _) = request_cost(&a, usage, None);
        assert!((cost - (1.0 + 0.10 + 1.25)).abs() < 1e-9, "{cost}");
    }

    #[tokio::test]
    async fn test_run_openai_reads_cached_tokens() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 2000, "completion_tokens": 10,
                          "prompt_tokens_details": {"cached_tokens": 1536}}
            })))
            .mount(&server)
            .await;
        let config = openai_config("k", "gpt-4o-mini");
        let r = run_openai(&Client::new(), &config, &server.uri(), "hi", 0, false)
            .await
            .unwrap();
        assert_eq!((r.prompt_tokens, r.cached_prompt_tokens), (2000, 1536));
        let expected = (464.0 * 0.15 + 1536.0 * 0.075 + 10.0 * 0.6) / 1e6;
        assert!((r.cost_usd - expected).abs() < 1e-12, "{}", r.cost_usd);
        assert_eq!(r.cost_source, Some(CostSource::LlmPrices));
    }

    #[test]
    fn test_chat_completions_url() {
        assert_eq!(
            chat_completions_url("http://localhost:11434"),
            "http://localhost:11434/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://openrouter.ai/api/v1/"),
            "https://openrouter.ai/api/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url("https://generativelanguage.googleapis.com/v1beta/openai"),
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        );
        // A host name that starts with v and a digit is not a version segment.
        assert_eq!(
            chat_completions_url("http://v2box:8000"),
            "http://v2box:8000/v1/chat/completions"
        );
    }

    //  OpenAI wire mock

    fn openai_config(api_key: &str, model: &str) -> ProviderConfig {
        ProviderConfig {
            name: "openai".into(),
            model: model.to_owned(),
            api_key: api_key.to_owned(),
            base_url: String::new(),
            max_tokens: 64,
        }
    }

    fn anthropic_config(api_key: &str, model: &str) -> ProviderConfig {
        ProviderConfig {
            name: "anthropic".into(),
            model: model.to_owned(),
            api_key: api_key.to_owned(),
            base_url: String::new(),
            max_tokens: 64,
        }
    }

    #[tokio::test]
    async fn test_run_openai_success_parses_result() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "Hello!"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            })))
            .mount(&server)
            .await;

        let client = Client::new();
        let config = openai_config("test-key", "gpt-4o-mini");
        let result = run_openai(&client, &config, &server.uri(), "Say hi", 0, false).await;

        assert!(result.is_ok(), "expected Ok, got: {result:?}");
        let r = result.unwrap_or_else(|_| unreachable!());
        assert_eq!(r.provider, "openai");
        assert_eq!(r.model, "gpt-4o-mini");
        assert_eq!(r.response_text, "Hello!");
        assert_eq!(r.prompt_tokens, 10);
        assert_eq!(r.completion_tokens, 5);
        assert!(r.cost_usd > 0.0);
        assert_eq!(r.run_index, 0);
    }

    #[tokio::test]
    async fn test_run_openai_401_returns_api_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
            .mount(&server)
            .await;

        let client = Client::new();
        let config = openai_config("bad-key", "gpt-4o-mini");
        let result = run_openai(&client, &config, &server.uri(), "hello", 0, false).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, BenchError::ApiError { status: 401, .. }),
            "expected ApiError with 401, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn test_run_openai_429_returns_rate_limited() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(429)
                    .insert_header("retry-after", "60")
                    .set_body_string("rate limited"),
            )
            .mount(&server)
            .await;

        let client = Client::new();
        let config = openai_config("test-key", "gpt-4o-mini");
        let result = run_openai(&client, &config, &server.uri(), "hello", 0, false).await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(
                err,
                BenchError::RateLimited {
                    provider: _,
                    retry_after_secs: Some(60)
                }
            ),
            "expected RateLimited with retry 60, got: {err:?}"
        );
    }

    #[tokio::test]
    async fn test_run_openai_missing_usage_defaults_to_zero() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "Hi"}}]
                // no "usage" field
            })))
            .mount(&server)
            .await;

        let client = Client::new();
        let config = openai_config("test-key", "gpt-4o-mini");
        let result = run_openai(&client, &config, &server.uri(), "hello", 0, false).await;

        assert!(result.is_ok());
        let r = result.unwrap_or_else(|_| unreachable!());
        assert_eq!(r.prompt_tokens, 0);
        assert_eq!(r.completion_tokens, 0);
    }

    #[tokio::test]
    async fn test_run_openai_run_index_stored() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3}
            })))
            .mount(&server)
            .await;

        let client = Client::new();
        let config = openai_config("test-key", "gpt-4o");
        let result = run_openai(&client, &config, &server.uri(), "hello", 7, false).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap_or_else(|_| unreachable!()).run_index, 7);
    }

    //  Anthropic wire mock

    #[tokio::test]
    async fn test_run_anthropic_success_parses_result() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(header("x-api-key", "ant-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{"type": "text", "text": "Hello from Claude!"}],
                "usage": {"input_tokens": 12, "output_tokens": 8}
            })))
            .mount(&server)
            .await;

        let client = Client::new();
        let config = anthropic_config("ant-key", "claude-haiku-4-5");
        let result = run_anthropic(&client, &config, &server.uri(), "Hi", 0, false).await;

        assert!(result.is_ok(), "expected Ok, got: {result:?}");
        let r = result.unwrap_or_else(|_| unreachable!());
        assert_eq!(r.provider, "anthropic");
        assert_eq!(r.response_text, "Hello from Claude!");
        assert_eq!(r.prompt_tokens, 12);
        assert_eq!(r.completion_tokens, 8);
        assert!(r.cost_usd > 0.0);
    }

    #[tokio::test]
    async fn test_run_anthropic_401_returns_api_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(401).set_body_string("auth error"))
            .mount(&server)
            .await;

        let client = Client::new();
        let config = anthropic_config("bad-key", "claude-haiku-4-5");
        let result = run_anthropic(&client, &config, &server.uri(), "hi", 0, false).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BenchError::ApiError { status: 401, .. }
        ));
    }

    #[tokio::test]
    async fn test_run_anthropic_429_returns_rate_limited() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limit"))
            .mount(&server)
            .await;

        let client = Client::new();
        let config = anthropic_config("ant-key", "claude-haiku-4-5");
        let result = run_anthropic(&client, &config, &server.uri(), "hi", 0, false).await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BenchError::RateLimited { .. }
        ));
    }

    #[tokio::test]
    async fn test_run_anthropic_multiple_text_blocks_concatenated() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [
                    {"type": "text", "text": "Part 1. "},
                    {"type": "text", "text": "Part 2."}
                ],
                "usage": {"input_tokens": 5, "output_tokens": 10}
            })))
            .mount(&server)
            .await;

        let client = Client::new();
        let config = anthropic_config("ant-key", "claude-haiku-4-5");
        let result = run_anthropic(&client, &config, &server.uri(), "tell me", 0, false).await;

        assert!(result.is_ok());
        let r = result.unwrap_or_else(|_| unreachable!());
        assert_eq!(r.response_text, "Part 1. Part 2.");
    }

    #[tokio::test]
    async fn test_run_anthropic_non_text_blocks_excluded() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "fn"},
                    {"type": "text", "text": "Final answer."}
                ],
                "usage": {"input_tokens": 5, "output_tokens": 10}
            })))
            .mount(&server)
            .await;

        let client = Client::new();
        let config = anthropic_config("ant-key", "claude-haiku-4-5");
        let result = run_anthropic(&client, &config, &server.uri(), "use tool", 0, false).await;

        assert!(result.is_ok());
        let r = result.unwrap_or_else(|_| unreachable!());
        assert_eq!(r.response_text, "Final answer.");
    }

    #[tokio::test]
    async fn test_run_anthropic_missing_usage_defaults_to_zero() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{"type": "text", "text": "hi"}]
            })))
            .mount(&server)
            .await;

        let client = Client::new();
        let config = anthropic_config("ant-key", "claude-haiku-4-5");
        let result = run_anthropic(&client, &config, &server.uri(), "hi", 0, false).await;

        assert!(result.is_ok());
        let r = result.unwrap_or_else(|_| unreachable!());
        assert_eq!(r.prompt_tokens, 0);
        assert_eq!(r.completion_tokens, 0);
    }

    #[tokio::test]
    async fn test_run_anthropic_tokens_per_second_non_negative() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{"type": "text", "text": "response"}],
                "usage": {"input_tokens": 10, "output_tokens": 20}
            })))
            .mount(&server)
            .await;

        let client = Client::new();
        let config = anthropic_config("ant-key", "claude-haiku-4-5");
        let result = run_anthropic(&client, &config, &server.uri(), "prompt", 0, false).await;

        assert!(result.is_ok());
        let r = result.unwrap_or_else(|_| unreachable!());
        assert!(r.tokens_per_second >= 0.0);
    }

    #[tokio::test]
    async fn test_official_openai_uses_max_completion_tokens() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(|req: &wiremock::Request| {
                let b: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
                assert_eq!(b["max_completion_tokens"], 64);
                assert!(b.get("max_tokens").is_none());
                ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "choices": [{"message": {"content": "ok"}}]
                }))
            })
            .expect(1)
            .mount(&server)
            .await;
        // base_url empty = the official endpoint (the test passes the mock's
        // URL as the resolved base, as the runner would pass api.openai.com).
        let config = openai_config("k", "o3");
        run_openai(&Client::new(), &config, &server.uri(), "hi", 0, false)
            .await
            .unwrap();

        // A custom server keeps max_tokens.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(serde_json::json!({"max_tokens": 64})))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "ok"}}]
            })))
            .expect(1)
            .mount(&server)
            .await;
        let mut config = openai_config("", "llama3.2");
        config.base_url = server.uri();
        run_openai(&Client::new(), &config, &server.uri(), "hi", 0, false)
            .await
            .unwrap();
    }

    //  Streaming

    fn sse(events: &[&str]) -> ResponseTemplate {
        let body: String = events.iter().map(|e| format!("{e}\n\n")).collect();
        ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
    }

    #[tokio::test]
    async fn test_run_openai_stream_reads_text_usage_and_ttft() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(serde_json::json!({
                "stream": true,
                "stream_options": {"include_usage": true}
            })))
            .respond_with(sse(&[
                r#"data: {"choices":[{"delta":{"role":"assistant","content":""}}]}"#,
                r#"data: {"choices":[{"delta":{"content":"Hel"}}]}"#,
                r#"data: {"choices":[{"delta":{"content":"lo é"}}]}"#,
                r#"data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":3}}"#,
                "data: [DONE]",
            ]))
            .expect(1)
            .mount(&server)
            .await;

        let config = openai_config("k", "gpt-4o-mini");
        let r = run_openai(&Client::new(), &config, &server.uri(), "hi", 0, true)
            .await
            .unwrap();
        assert_eq!(r.response_text, "Hello é");
        assert_eq!((r.prompt_tokens, r.completion_tokens), (9, 3));
        let ttft = r.ttft_ms.expect("streamed request has a TTFT");
        assert_eq!(r.latency_ms, ttft);
        assert!(ttft <= r.total_ms);
        assert!(r.cost_usd > 0.0);
    }

    #[tokio::test]
    async fn test_run_anthropic_stream_reads_events() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(wiremock::matchers::body_partial_json(serde_json::json!({"stream": true})))
            .respond_with(sse(&[
                "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":12,\"output_tokens\":1}}}",
                "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}",
                "event: ping\ndata: {\"type\":\"ping\"}",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi \"}}",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"there\"}}",
                "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}",
                "event: message_stop\ndata: {\"type\":\"message_stop\"}",
            ]))
            .mount(&server)
            .await;

        let config = anthropic_config("k", "claude-haiku-4-5");
        let r = run_anthropic(&Client::new(), &config, &server.uri(), "hi", 0, true)
            .await
            .unwrap();
        assert_eq!(r.response_text, "Hi there");
        assert_eq!((r.prompt_tokens, r.completion_tokens), (12, 5));
        assert!(r.ttft_ms.is_some());
    }

    #[tokio::test]
    async fn test_stream_error_event_is_an_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(sse(&[
                "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}",
            ]))
            .mount(&server)
            .await;
        let config = anthropic_config("k", "claude-haiku-4-5");
        let e = run_anthropic(&Client::new(), &config, &server.uri(), "hi", 0, true)
            .await
            .unwrap_err();
        assert!(e.to_string().contains("Overloaded"), "{e}");
    }

    #[tokio::test]
    async fn test_openrouter_reported_cost_is_used() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(
                serde_json::json!({"usage": {"include": true}}),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.000042}
            })))
            .expect(1)
            .mount(&server)
            .await;
        let mut config = openai_config("k", "meta-llama/llama-3.3-70b-instruct");
        config.name = "openrouter".into();
        let base = format!("{}/api/v1", server.uri());
        let r = run_openai(&Client::new(), &config, &base, "hi", 0, false)
            .await
            .unwrap();
        assert_eq!(r.provider, "openrouter");
        assert!((r.cost_usd - 0.000042).abs() < 1e-12);
    }
}
