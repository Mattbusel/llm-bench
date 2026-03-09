//! # Module: Providers
//!
//! ## Responsibility
//! Issue inference requests to OpenAI and Anthropic REST APIs, measure timing,
//! parse token usage, and compute per-request cost.
//!
//! ## Guarantees
//! - All network calls are async and cancellation-safe.
//! - Timing is measured with `std::time::Instant` (monotonic).
//! - Cost calculation uses static per-token price tables.
//! - This module never panics.
//!
//! ## NOT Responsible For
//! - Retry logic (see `runner.rs`)
//! - Progress reporting (see `main.rs`)
//! - Output formatting (see `report.rs`)

use std::time::Instant;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument};

use crate::error::BenchError;
use crate::types::{BenchResult, ProviderConfig};

//  Pricing tables (USD per 1 000 tokens) 

/// USD per 1 000 prompt tokens for each model.
fn prompt_price_per_1k(model: &str) -> f64 {
    match model {
        "gpt-4o" => 0.005,
        "gpt-4o-mini" => 0.000150,
        "gpt-4-turbo" => 0.010,
        "gpt-4-turbo-preview" => 0.010,
        "claude-3-5-sonnet-20241022" | "claude-3-5-sonnet-latest" => 0.003,
        "claude-3-5-haiku-20241022" | "claude-3-5-haiku-latest" => 0.0008,
        "claude-3-opus-20240229" | "claude-3-opus-latest" => 0.015,
        _ => 0.002, // conservative fallback
    }
}

/// USD per 1 000 completion tokens for each model.
fn completion_price_per_1k(model: &str) -> f64 {
    match model {
        "gpt-4o" => 0.015,
        "gpt-4o-mini" => 0.000600,
        "gpt-4-turbo" => 0.030,
        "gpt-4-turbo-preview" => 0.030,
        "claude-3-5-sonnet-20241022" | "claude-3-5-sonnet-latest" => 0.015,
        "claude-3-5-haiku-20241022" | "claude-3-5-haiku-latest" => 0.004,
        "claude-3-opus-20240229" | "claude-3-opus-latest" => 0.075,
        _ => 0.002, // conservative fallback
    }
}

/// Compute the USD cost for a request given token counts.
pub fn compute_cost(model: &str, prompt_tokens: u32, completion_tokens: u32) -> f64 {
    let p = f64::from(prompt_tokens) * prompt_price_per_1k(model) / 1000.0;
    let c = f64::from(completion_tokens) * completion_price_per_1k(model) / 1000.0;
    p + c
}

//  OpenAI wire types 

#[derive(Serialize)]
struct OpenAiRequest<'a> {
    model: &'a str,
    messages: Vec<OpenAiMessage<'a>>,
    max_tokens: u32,
}

#[derive(Serialize)]
struct OpenAiMessage<'a> {
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

#[derive(Deserialize, Debug)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

//  Anthropic wire types 

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: Vec<AnthropicMessage<'a>>,
}

#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a str,
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

#[derive(Deserialize, Debug)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

//  OpenAI provider 

/// Issue one inference request to the OpenAI chat-completions endpoint.
///
/// # Arguments
/// * `client`     -  shared `reqwest::Client` (connection pool reuse)
/// * `config`     -  provider configuration (model, API key, max_tokens)
/// * `base_url`   -  API base URL (overrideable for testing)
/// * `prompt`     -  the user prompt string
/// * `run_idx`    -  zero-based run index within the prompt batch
///
/// # Returns
/// `Ok(BenchResult)` on success, or a typed `BenchError` on failure.
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
) -> Result<BenchResult, BenchError> {
    let url = format!("{base_url}/v1/chat/completions");
    let body = OpenAiRequest {
        model: &config.model,
        messages: vec![OpenAiMessage {
            role: "user",
            content: prompt,
        }],
        max_tokens: config.max_tokens,
    };

    let start = Instant::now();
    let resp = client
        .post(&url)
        .bearer_auth(&config.api_key)
        .json(&body)
        .send()
        .await?;

    let status = resp.status().as_u16();
    let total_ms = start.elapsed().as_millis() as u64;

    if status == 429 {
        let retry = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        return Err(BenchError::RateLimited {
            provider: "openai".into(),
            retry_after_secs: retry,
        });
    }

    if !resp.status().is_success() {
        let body_text = resp.text().await.unwrap_or_default();
        return Err(BenchError::ApiError {
            provider: "openai".into(),
            model: config.model.clone(),
            status,
            body: body_text,
        });
    }

    let parsed: OpenAiResponse = resp.json().await?;

    let response_text = parsed
        .choices
        .into_iter()
        .next()
        .and_then(|c| c.message.content)
        .unwrap_or_default();

    let (prompt_tokens, completion_tokens) = parsed
        .usage
        .map(|u| (u.prompt_tokens, u.completion_tokens))
        .unwrap_or((0, 0));

    let cost_usd = compute_cost(&config.model, prompt_tokens, completion_tokens);
    let tokens_per_second = if total_ms > 0 {
        f64::from(completion_tokens) / (total_ms as f64 / 1000.0)
    } else {
        0.0
    };

    debug!(
        model = %config.model,
        total_ms,
        prompt_tokens,
        completion_tokens,
        "openai call complete"
    );

    Ok(BenchResult {
        provider: "openai".into(),
        model: config.model.clone(),
        prompt: prompt.to_owned(),
        latency_ms: total_ms, // non-streaming: TTFT == total
        total_ms,
        prompt_tokens,
        completion_tokens,
        cost_usd,
        tokens_per_second,
        response_text,
        run_index: run_idx,
    })
}

//  Anthropic provider 

/// Issue one inference request to the Anthropic messages endpoint.
///
/// # Arguments
/// * `client`     -  shared `reqwest::Client`
/// * `config`     -  provider configuration
/// * `base_url`   -  API base URL (overrideable for testing)
/// * `prompt`     -  the user prompt string
/// * `run_idx`    -  zero-based run index
///
/// # Returns
/// `Ok(BenchResult)` on success, or a typed `BenchError` on failure.
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
) -> Result<BenchResult, BenchError> {
    let url = format!("{base_url}/v1/messages");
    let body = AnthropicRequest {
        model: &config.model,
        max_tokens: config.max_tokens,
        messages: vec![AnthropicMessage {
            role: "user",
            content: prompt,
        }],
    };

    let start = Instant::now();
    let resp = client
        .post(&url)
        .header("x-api-key", &config.api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await?;

    let status = resp.status().as_u16();
    let total_ms = start.elapsed().as_millis() as u64;

    if status == 429 {
        let retry = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        return Err(BenchError::RateLimited {
            provider: "anthropic".into(),
            retry_after_secs: retry,
        });
    }

    if !resp.status().is_success() {
        let body_text = resp.text().await.unwrap_or_default();
        return Err(BenchError::ApiError {
            provider: "anthropic".into(),
            model: config.model.clone(),
            status,
            body: body_text,
        });
    }

    let parsed: AnthropicResponse = resp.json().await?;

    let response_text = parsed
        .content
        .into_iter()
        .filter(|b| b.block_type == "text")
        .filter_map(|b| b.text)
        .collect::<Vec<_>>()
        .join("");

    let (prompt_tokens, completion_tokens) = parsed
        .usage
        .map(|u| (u.input_tokens, u.output_tokens))
        .unwrap_or((0, 0));

    let cost_usd = compute_cost(&config.model, prompt_tokens, completion_tokens);
    let tokens_per_second = if total_ms > 0 {
        f64::from(completion_tokens) / (total_ms as f64 / 1000.0)
    } else {
        0.0
    };

    debug!(
        model = %config.model,
        total_ms,
        prompt_tokens,
        completion_tokens,
        "anthropic call complete"
    );

    Ok(BenchResult {
        provider: "anthropic".into(),
        model: config.model.clone(),
        prompt: prompt.to_owned(),
        latency_ms: total_ms,
        total_ms,
        prompt_tokens,
        completion_tokens,
        cost_usd,
        tokens_per_second,
        response_text,
        run_index: run_idx,
    })
}

/// Return a human-readable table of all supported models and their pricing.
pub fn supported_models() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            provider: "openai",
            model: "gpt-4o",
            prompt_per_1k: 0.005,
            completion_per_1k: 0.015,
        },
        ModelInfo {
            provider: "openai",
            model: "gpt-4o-mini",
            prompt_per_1k: 0.000150,
            completion_per_1k: 0.000600,
        },
        ModelInfo {
            provider: "openai",
            model: "gpt-4-turbo",
            prompt_per_1k: 0.010,
            completion_per_1k: 0.030,
        },
        ModelInfo {
            provider: "anthropic",
            model: "claude-3-5-sonnet-20241022",
            prompt_per_1k: 0.003,
            completion_per_1k: 0.015,
        },
        ModelInfo {
            provider: "anthropic",
            model: "claude-3-5-haiku-20241022",
            prompt_per_1k: 0.0008,
            completion_per_1k: 0.004,
        },
        ModelInfo {
            provider: "anthropic",
            model: "claude-3-opus-20240229",
            prompt_per_1k: 0.015,
            completion_per_1k: 0.075,
        },
    ]
}

/// Metadata about a supported model.
pub struct ModelInfo {
    pub provider: &'static str,
    pub model: &'static str,
    pub prompt_per_1k: f64,
    pub completion_per_1k: f64,
}

//  Tests 

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
        let expected = 0.005 + 0.015; // 0.02
        assert!((cost - expected).abs() < 1e-9);
    }

    #[test]
    fn test_compute_cost_claude_haiku_known_values() {
        let cost = compute_cost("claude-3-5-haiku-20241022", 1000, 1000);
        let expected = 0.0008 + 0.004; // 0.0048
        assert!((cost - expected).abs() < 1e-9);
    }

    #[test]
    fn test_compute_cost_claude_opus_known_values() {
        let cost = compute_cost("claude-3-opus-20240229", 1000, 1000);
        let expected = 0.015 + 0.075;
        assert!((cost - expected).abs() < 1e-9);
    }

    #[test]
    fn test_compute_cost_unknown_model_uses_fallback() {
        let cost = compute_cost("unknown-model-xyz", 1000, 1000);
        // fallback: 0.002 + 0.002 = 0.004
        assert!(cost > 0.0, "fallback cost should be positive");
    }

    #[test]
    fn test_compute_cost_zero_tokens_returns_zero() {
        let cost = compute_cost("gpt-4o", 0, 0);
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_compute_cost_only_prompt_tokens() {
        let cost = compute_cost("gpt-4o", 1000, 0);
        let expected = 0.005;
        assert!((cost - expected).abs() < 1e-9);
    }

    #[test]
    fn test_compute_cost_only_completion_tokens() {
        let cost = compute_cost("gpt-4o", 0, 1000);
        let expected = 0.015;
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

    //  Price tables 

    #[test]
    fn test_prompt_price_per_1k_gpt4_turbo() {
        assert!((prompt_price_per_1k("gpt-4-turbo") - 0.010).abs() < 1e-9);
    }

    #[test]
    fn test_completion_price_per_1k_gpt4_turbo() {
        assert!((completion_price_per_1k("gpt-4-turbo") - 0.030).abs() < 1e-9);
    }

    #[test]
    fn test_prompt_price_claude_sonnet() {
        assert!(
            (prompt_price_per_1k("claude-3-5-sonnet-20241022") - 0.003).abs() < 1e-9
        );
    }

    #[test]
    fn test_completion_price_claude_sonnet() {
        assert!(
            (completion_price_per_1k("claude-3-5-sonnet-20241022") - 0.015).abs() < 1e-9
        );
    }

    #[test]
    fn test_prompt_price_latest_alias_equals_dated() {
        let dated = prompt_price_per_1k("claude-3-5-sonnet-20241022");
        let latest = prompt_price_per_1k("claude-3-5-sonnet-latest");
        assert!((dated - latest).abs() < 1e-9, "aliases should have same price");
    }

    //  supported_models 

    #[test]
    fn test_supported_models_non_empty() {
        assert!(!supported_models().is_empty());
    }

    #[test]
    fn test_supported_models_all_have_positive_prices() {
        for m in supported_models() {
            assert!(m.prompt_per_1k > 0.0, "model {} has zero prompt price", m.model);
            assert!(
                m.completion_per_1k > 0.0,
                "model {} has zero completion price",
                m.model
            );
        }
    }

    #[test]
    fn test_supported_models_providers_are_known() {
        for m in supported_models() {
            assert!(
                m.provider == "openai" || m.provider == "anthropic",
                "unknown provider: {}",
                m.provider
            );
        }
    }

    #[test]
    fn test_supported_models_contains_gpt4o_mini() {
        let found = supported_models()
            .iter()
            .any(|m| m.model == "gpt-4o-mini");
        assert!(found, "gpt-4o-mini should be in supported models");
    }

    #[test]
    fn test_supported_models_contains_claude_haiku() {
        let found = supported_models()
            .iter()
            .any(|m| m.model == "claude-3-5-haiku-20241022");
        assert!(found, "haiku should be in supported models");
    }

    //  OpenAI wire mock 

    fn openai_config(api_key: &str, model: &str) -> ProviderConfig {
        ProviderConfig {
            name: "openai".into(),
            model: model.to_owned(),
            api_key: api_key.to_owned(),
            max_tokens: 64,
        }
    }

    fn anthropic_config(api_key: &str, model: &str) -> ProviderConfig {
        ProviderConfig {
            name: "anthropic".into(),
            model: model.to_owned(),
            api_key: api_key.to_owned(),
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
        let result = run_openai(&client, &config, &server.uri(), "Say hi", 0).await;

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
        let result = run_openai(&client, &config, &server.uri(), "hello", 0).await;

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
        let result = run_openai(&client, &config, &server.uri(), "hello", 0).await;

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
        let result = run_openai(&client, &config, &server.uri(), "hello", 0).await;

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
        let result = run_openai(&client, &config, &server.uri(), "hello", 7).await;

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
        let config = anthropic_config("ant-key", "claude-3-5-haiku-20241022");
        let result = run_anthropic(&client, &config, &server.uri(), "Hi", 0).await;

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
        let config = anthropic_config("bad-key", "claude-3-5-haiku-20241022");
        let result = run_anthropic(&client, &config, &server.uri(), "hi", 0).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BenchError::ApiError { status: 401, .. }));
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
        let config = anthropic_config("ant-key", "claude-3-5-haiku-20241022");
        let result = run_anthropic(&client, &config, &server.uri(), "hi", 0).await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BenchError::RateLimited { .. }));
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
        let config = anthropic_config("ant-key", "claude-3-5-haiku-20241022");
        let result = run_anthropic(&client, &config, &server.uri(), "tell me", 0).await;

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
        let config = anthropic_config("ant-key", "claude-3-5-haiku-20241022");
        let result = run_anthropic(&client, &config, &server.uri(), "use tool", 0).await;

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
        let config = anthropic_config("ant-key", "claude-3-5-haiku-20241022");
        let result = run_anthropic(&client, &config, &server.uri(), "hi", 0).await;

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
        let config = anthropic_config("ant-key", "claude-3-5-haiku-20241022");
        let result = run_anthropic(&client, &config, &server.uri(), "prompt", 0).await;

        assert!(result.is_ok());
        let r = result.unwrap_or_else(|_| unreachable!());
        assert!(r.tokens_per_second >= 0.0);
    }
}
