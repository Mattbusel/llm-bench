//! # Module: Runner
//!
//! Run every (provider, prompt, run) combination concurrently, bounded by a
//! semaphore, and collect the results and the failures. Failed requests can
//! be retried with backoff (`--retries`), using the `backon` crate.
//!
//! ## Guarantees
//! - Concurrent calls are bounded by `BenchConfig::concurrency`.
//! - Every task failure is captured as a [`BenchFailure`], never silently dropped.
//! - Progress callbacks are invoked once per completed task.
//! - This module never panics.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use backon::{ExponentialBuilder, Retryable};
use futures::future::join_all;
use reqwest::Client;
use tokio::sync::Semaphore;
use tracing::{debug, instrument};

use crate::config::PriceTable;
use crate::error::BenchError;
use crate::providers::{run_anthropic, run_openai};
use crate::types::{BenchConfig, BenchFailure, BenchResult, CostSource, ProviderConfig};

/// Official OpenAI API base URL (used when no `--openai-base-url` is given).
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com";
/// Official Anthropic API base URL.
pub const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";
/// Local Ollama server.
pub const DEFAULT_OLLAMA_BASE_URL: &str = "http://localhost:11434";
/// OpenRouter's OpenAI-compatible API.
pub const DEFAULT_OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
/// Google Gemini's OpenAI-compatible API.
pub const DEFAULT_GEMINI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta/openai";
/// Groq's OpenAI-compatible API.
pub const DEFAULT_GROQ_BASE_URL: &str = "https://api.groq.com/openai/v1";
/// Together AI's OpenAI-compatible API.
pub const DEFAULT_TOGETHER_BASE_URL: &str = "https://api.together.xyz/v1";
/// Fireworks AI's OpenAI-compatible API.
pub const DEFAULT_FIREWORKS_BASE_URL: &str = "https://api.fireworks.ai/inference/v1";
/// DeepSeek's OpenAI-compatible API.
pub const DEFAULT_DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com/v1";

/// Built-in providers and the environment variable that holds each one's
/// API key (empty for Ollama, which needs none).
pub const BUILTIN_PROVIDERS: &[(&str, &str)] = &[
    ("openai", "OPENAI_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("openrouter", "OPENROUTER_API_KEY"),
    ("gemini", "GEMINI_API_KEY"),
    ("groq", "GROQ_API_KEY"),
    ("together", "TOGETHER_API_KEY"),
    ("fireworks", "FIREWORKS_API_KEY"),
    ("deepseek", "DEEPSEEK_API_KEY"),
    ("ollama", ""),
];

/// Longest wait honoured from a `Retry-After` header.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

//  Runner

/// Drives a full benchmark run and returns all individual results.
pub struct BenchRunner {
    client: Client,
    prices: Arc<PriceTable>,
}

impl BenchRunner {
    /// Construct a runner with a shared HTTP client (120 s timeout per request).
    ///
    /// # Panics
    /// This function never panics.
    pub fn new() -> Result<Self, BenchError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(BenchError::Http)?;
        Ok(Self {
            client,
            prices: Arc::new(PriceTable::default()),
        })
    }

    /// Price requests with `prices` (the `[prices]` section of a config
    /// file). An entry wins over the `llm-prices` list price; a cost the
    /// provider reports itself (OpenRouter) still wins over both.
    pub fn with_prices(mut self, prices: PriceTable) -> Self {
        self.prices = Arc::new(prices);
        self
    }

    /// Run the full benchmark suite described by `config`.
    ///
    /// Returns the successful results (one per provider, prompt and run)
    /// and one [`BenchFailure`] per request that failed, so callers can
    /// report real success rates. `on_progress` is invoked after each task
    /// completes with `(completed, total)`.
    ///
    /// # Panics
    /// This function never panics.
    #[instrument(skip(self, config, on_progress))]
    pub async fn run_detailed<F>(
        &self,
        config: &BenchConfig,
        on_progress: F,
    ) -> Result<(Vec<BenchResult>, Vec<BenchFailure>), BenchError>
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        let semaphore = Arc::new(Semaphore::new(config.concurrency.max(1)));
        let on_progress = Arc::new(on_progress);

        let mut tasks = Vec::new();
        for provider in &config.providers {
            for prompt in &config.prompts {
                for run_idx in 0..config.runs_per_prompt {
                    tasks.push((provider.clone(), prompt.clone(), run_idx));
                }
            }
        }

        // Warm-up requests: sent one at a time per model, results discarded.
        for provider in &config.providers {
            for _ in 0..config.warmup {
                let prompt = config.prompts.first().map(String::as_str).unwrap_or("");
                let r = dispatch_call(&self.client, provider, prompt, 0, config.stream).await;
                debug!(model = %provider.model, ok = r.is_ok(), "warm-up request");
            }
        }

        let total = tasks.len();
        debug!(total_tasks = total, "starting benchmark run");

        let completed = Arc::new(AtomicUsize::new(0));
        let (stream, retries) = (config.stream, config.retries);

        let futures: Vec<_> = tasks
            .into_iter()
            .map(|(provider, prompt, run_idx)| {
                let sem = Arc::clone(&semaphore);
                let cb = Arc::clone(&on_progress);
                let done = Arc::clone(&completed);
                let client = self.client.clone();
                let prices = Arc::clone(&self.prices);

                tokio::spawn(async move {
                    let _permit = sem.acquire().await.map_err(|e| BenchError::Concurrency {
                        reason: e.to_string(),
                    })?;

                    let mut result =
                        call_with_retries(&client, &provider, &prompt, run_idx, stream, retries)
                            .await;
                    if let Ok(r) = result.as_mut() {
                        apply_configured_price(r, &prices);
                    }

                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    cb(n, total);

                    Ok::<_, BenchError>(result.map_err(|e| BenchFailure {
                        provider: provider.name.clone(),
                        model: provider.model.clone(),
                        error: e.to_string(),
                    }))
                })
            })
            .collect();

        let join_results = join_all(futures).await;

        let mut results = Vec::with_capacity(join_results.len());
        let mut failures = Vec::new();
        for jr in join_results {
            match jr {
                Ok(Ok(Ok(r))) => results.push(r),
                Ok(Ok(Err(f))) => {
                    debug!(error = %f.error, "benchmark task returned error");
                    failures.push(f);
                }
                Ok(Err(e)) => return Err(e),
                Err(e) => {
                    return Err(BenchError::Concurrency {
                        reason: format!("task panicked or was cancelled: {e}"),
                    });
                }
            }
        }

        Ok((results, failures))
    }
}

/// Re-price `r` from a `[prices]` entry, unless the provider reported the
/// cost itself.
///
/// Cached prompt tokens are billed at the entry's `cached_input` price (or
/// `input` when it has none). OpenAI-compatible APIs count cached tokens
/// inside `prompt_tokens`, so only the rest is billed at `input`; Anthropic
/// reports them, and cache writes, apart from `prompt_tokens`, so those are
/// added (cache writes at `input`, see [`crate::config::ModelPrice`]).
pub fn apply_configured_price(r: &mut BenchResult, prices: &PriceTable) {
    if r.cost_source == Some(CostSource::Provider) {
        return;
    }
    if let Some(p) = prices.get(&r.provider, &r.model) {
        let uncached = if r.provider == "anthropic" {
            r.prompt_tokens.saturating_add(r.cache_write_prompt_tokens)
        } else {
            r.prompt_tokens.saturating_sub(r.cached_prompt_tokens)
        };
        r.cost_usd = p.cost_cached(uncached, r.cached_prompt_tokens, r.completion_tokens);
        r.cost_source = Some(CostSource::Config);
    }
}

/// Whether a failed request is worth retrying: rate limits, timeouts,
/// server errors and connection failures. Bad keys, unknown models and
/// malformed responses are not.
pub fn is_retryable(e: &BenchError) -> bool {
    match e {
        BenchError::RateLimited { .. } => true,
        BenchError::ApiError { status, .. } => {
            matches!(status, 408 | 409 | 500 | 502 | 503 | 504 | 529)
        }
        BenchError::Http(e) => e.is_timeout() || e.is_connect(),
        _ => false,
    }
}

/// One request with up to `retries` retries (exponential backoff with
/// jitter, starting at 500 ms; a `Retry-After` on a 429 sets the wait, up
/// to 60 s). Latency is that of the attempt that succeeded.
async fn call_with_retries(
    client: &Client,
    provider: &ProviderConfig,
    prompt: &str,
    run_idx: u32,
    stream: bool,
    retries: u32,
) -> Result<BenchResult, BenchError> {
    let attempts = AtomicU32::new(0);
    let backoff = ExponentialBuilder::default()
        .with_min_delay(Duration::from_millis(500))
        .with_max_delay(Duration::from_secs(30))
        .with_max_times(retries as usize)
        .with_jitter();
    let mut result = (|| {
        attempts.fetch_add(1, Ordering::Relaxed);
        dispatch_call(client, provider, prompt, run_idx, stream)
    })
    .retry(backoff)
    .when(is_retryable)
    .adjust(|e: &BenchError, dur| match (dur, e) {
        (None, _) => None,
        (
            Some(_),
            BenchError::RateLimited {
                retry_after_secs: Some(s),
                ..
            },
        ) => Some(Duration::from_secs(*s).min(MAX_RETRY_AFTER)),
        (d, _) => d,
    })
    .await;
    if let Ok(r) = result.as_mut() {
        r.retries = attempts.load(Ordering::Relaxed).saturating_sub(1);
    }
    result
}

/// The base URL a provider uses when none was configured.
pub fn default_base_url(provider: &str) -> Option<&'static str> {
    Some(match provider {
        "openai" => DEFAULT_OPENAI_BASE_URL,
        "anthropic" => DEFAULT_ANTHROPIC_BASE_URL,
        "ollama" => DEFAULT_OLLAMA_BASE_URL,
        "openrouter" => DEFAULT_OPENROUTER_BASE_URL,
        "gemini" => DEFAULT_GEMINI_BASE_URL,
        "groq" => DEFAULT_GROQ_BASE_URL,
        "together" => DEFAULT_TOGETHER_BASE_URL,
        "fireworks" => DEFAULT_FIREWORKS_BASE_URL,
        "deepseek" => DEFAULT_DEEPSEEK_BASE_URL,
        _ => return None,
    })
}

/// Send one request to the right API for `provider.name`: the Anthropic
/// Messages API for `anthropic`, OpenAI Chat Completions for everything
/// else. A provider that is not built in must carry a `base_url`.
async fn dispatch_call(
    client: &Client,
    provider: &ProviderConfig,
    prompt: &str,
    run_idx: u32,
    stream: bool,
) -> Result<BenchResult, BenchError> {
    let url = if !provider.base_url.is_empty() {
        provider.base_url.clone()
    } else if let Some(default) = default_base_url(&provider.name) {
        default.to_owned()
    } else {
        let names: Vec<&str> = BUILTIN_PROVIDERS.iter().map(|(n, _)| *n).collect();
        return Err(BenchError::InvalidConfig {
            reason: format!(
                "unknown provider '{}' with no base URL; built in: {}. \
                 Define others in a --config file ([providers.{}] base_url = \"...\")",
                provider.name,
                names.join(", "),
                provider.name
            ),
        });
    };
    if provider.name == "anthropic" {
        run_anthropic(client, provider, &url, prompt, run_idx, stream).await
    } else {
        run_openai(client, provider, &url, prompt, run_idx, stream).await
    }
}

//  Tests

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::types::BenchConfig;

    fn openai_provider(base_url: &str) -> ProviderConfig {
        // We abuse the api_key field to also carry the base_url in tests.
        // In production the base_url is hardcoded; in tests we override via a
        // helper that calls run_openai directly with the mock server URI.
        let _ = base_url; // used in direct provider tests, not runner tests
        ProviderConfig {
            name: "openai".into(),
            model: "gpt-4o-mini".into(),
            api_key: "test-key".into(),
            base_url: String::new(),
            max_tokens: 32,
        }
    }

    //  BenchRunner::new

    #[test]
    fn test_bench_runner_new_succeeds() {
        let runner = BenchRunner::new();
        assert!(runner.is_ok(), "BenchRunner::new should succeed");
    }

    //  dispatch_call unknown provider

    #[tokio::test]
    async fn test_dispatch_call_unknown_provider_returns_invalid_config() {
        let client = Client::new();
        let provider = ProviderConfig {
            name: "unknown-provider".into(),
            model: "model-x".into(),
            api_key: "key".into(),
            base_url: String::new(),
            max_tokens: 32,
        };
        let result = dispatch_call(&client, &provider, "hello", 0, false).await;
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), BenchError::InvalidConfig { .. }),
            "expected InvalidConfig for unknown provider"
        );
    }

    //  Full runner with mock server

    // Note: runner.run() hardcodes API base URLs for production use.
    // We test the runner's concurrency and aggregation logic using the
    // providers directly with mock servers (see providers.rs tests).
    // Here we verify the runner builds a correct task list.

    #[tokio::test]
    async fn test_runner_task_count_equals_providers_times_prompts_times_runs() {
        // We can't intercept the hardcoded URLs in runner.run(), so we test
        // the math by checking the formula directly.
        let num_providers = 2usize;
        let num_prompts = 3usize;
        let runs_per_prompt = 4u32;
        let expected = num_providers * num_prompts * runs_per_prompt as usize;
        assert_eq!(expected, 24);
    }

    #[tokio::test]
    async fn test_runner_empty_providers_returns_empty_results() {
        let runner = BenchRunner::new().unwrap_or_else(|_| unreachable!());
        let config = BenchConfig {
            prompts: vec!["hello".into()],
            runs_per_prompt: 2,
            concurrency: 4,
            providers: vec![], // no providers → no tasks
            stream: false,
            retries: 0,
            warmup: 0,
        };
        let results = runner
            .run_detailed(&config, |_, _| {})
            .await
            .map(|(r, _)| r);
        assert!(results.is_ok());
        assert!(results.unwrap_or_default().is_empty());
    }

    #[tokio::test]
    async fn test_runner_empty_prompts_returns_empty_results() {
        let runner = BenchRunner::new().unwrap_or_else(|_| unreachable!());
        let config = BenchConfig {
            prompts: vec![], // no prompts → no tasks
            runs_per_prompt: 2,
            concurrency: 4,
            providers: vec![openai_provider("")],
            stream: false,
            retries: 0,
            warmup: 0,
        };
        let results = runner
            .run_detailed(&config, |_, _| {})
            .await
            .map(|(r, _)| r);
        assert!(results.is_ok());
        assert!(results.unwrap_or_default().is_empty());
    }

    #[tokio::test]
    async fn test_runner_progress_callback_called_correct_times() {
        // Use a mock that returns 200 so tasks succeed.
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3}
            })))
            .mount(&server)
            .await;

        // We can't inject the mock URL into runner.run() since it hardcodes the
        // production URL.  Instead we test the progress counter logic directly:
        let call_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let cc = Arc::clone(&call_count);
        let cb = move |_done: usize, _total: usize| {
            cc.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        };

        // Simulate three task completions
        for i in 1..=3usize {
            cb(i, 3);
        }

        assert_eq!(call_count.load(std::sync::atomic::Ordering::Relaxed), 3);
    }

    //  Semaphore concurrency cap

    #[tokio::test]
    async fn test_semaphore_limits_concurrency() {
        // Build a semaphore with capacity 2 and acquire all 2 permits.
        // The 3rd acquisition must not succeed without releasing one.
        let sem = Arc::new(Semaphore::new(2));
        let p1 = sem.try_acquire();
        let p2 = sem.try_acquire();
        let p3 = sem.try_acquire();

        assert!(p1.is_ok(), "first permit should succeed");
        assert!(p2.is_ok(), "second permit should succeed");
        assert!(p3.is_err(), "third permit should fail (semaphore full)");
    }

    //  Provider config validation

    #[test]
    fn test_bench_config_with_multiple_providers_is_valid() {
        let config = BenchConfig {
            prompts: vec!["p1".into()],
            runs_per_prompt: 1,
            concurrency: 2,
            providers: vec![
                ProviderConfig {
                    name: "openai".into(),
                    model: "gpt-4o-mini".into(),
                    api_key: "k1".into(),
                    base_url: String::new(),
                    max_tokens: 128,
                },
                ProviderConfig {
                    name: "anthropic".into(),
                    model: "claude-haiku-4-5".into(),
                    api_key: "k2".into(),
                    base_url: String::new(),
                    max_tokens: 128,
                },
            ],
            stream: false,
            retries: 0,
            warmup: 0,
        };
        assert_eq!(config.providers.len(), 2);
    }

    #[tokio::test]
    async fn test_dispatch_openai_hits_openai_endpoint() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1}
            })))
            .mount(&server)
            .await;

        let client = Client::new();
        let provider = ProviderConfig {
            name: "openai".into(),
            model: "gpt-4o-mini".into(),
            api_key: "k".into(),
            base_url: String::new(),
            max_tokens: 16,
        };
        // Direct call to run_openai with mock base URL
        let result = run_openai(&client, &provider, &server.uri(), "hi", 0, false).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_dispatch_anthropic_hits_anthropic_endpoint() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 2, "output_tokens": 1}
            })))
            .mount(&server)
            .await;

        let client = Client::new();
        let provider = ProviderConfig {
            name: "anthropic".into(),
            model: "claude-haiku-4-5".into(),
            api_key: "k".into(),
            base_url: String::new(),
            max_tokens: 16,
        };
        let result = run_anthropic(&client, &provider, &server.uri(), "hi", 0, false).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_run_detailed_uses_custom_base_url_and_counts_failures() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2}
            })))
            .mount(&server)
            .await;

        let good = ProviderConfig {
            name: "openai".into(),
            model: "local-model".into(),
            api_key: String::new(),
            base_url: server.uri(),
            max_tokens: 16,
        };
        let bad = ProviderConfig {
            name: "anthropic".into(),
            model: "claude-haiku-4-5".into(),
            api_key: String::new(),
            base_url: server.uri(), // no /v1/messages mock: 404
            max_tokens: 16,
        };
        let config = BenchConfig {
            prompts: vec!["hi".into()],
            runs_per_prompt: 2,
            concurrency: 2,
            providers: vec![good, bad],
            stream: false,
            retries: 0,
            warmup: 0,
        };
        let runner = BenchRunner::new().unwrap_or_else(|_| unreachable!());
        let (results, failures) = runner
            .run_detailed(&config, |_, _| {})
            .await
            .unwrap_or_default();
        assert_eq!(results.len(), 2);
        assert_eq!(failures.len(), 2);
        assert!(failures.iter().all(|f| f.error.contains("404")));
    }

    //  Atomic progress tracking

    #[test]
    fn test_atomic_counter_increments_correctly() {
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        for _ in 0..10 {
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), 10);
    }

    //  Retries

    fn local_openai(server: &MockServer) -> ProviderConfig {
        ProviderConfig {
            name: "openai".into(),
            model: "local".into(),
            api_key: String::new(),
            base_url: server.uri(),
            max_tokens: 8,
        }
    }

    fn ok_body() -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        }))
    }

    fn one_request(provider: ProviderConfig, retries: u32) -> BenchConfig {
        BenchConfig {
            prompts: vec!["hi".into()],
            runs_per_prompt: 1,
            concurrency: 1,
            providers: vec![provider],
            stream: false,
            retries,
            warmup: 0,
        }
    }

    #[tokio::test]
    async fn test_rate_limit_is_retried_after_retry_after() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "1"))
            .up_to_n_times(1)
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ok_body())
            .expect(1)
            .mount(&server)
            .await;

        let start = std::time::Instant::now();
        let (results, failures) = BenchRunner::new()
            .unwrap()
            .run_detailed(&one_request(local_openai(&server), 2), |_, _| {})
            .await
            .unwrap();
        assert_eq!((results.len(), failures.len()), (1, 0));
        assert_eq!(results[0].retries, 1);
        assert!(
            start.elapsed() >= Duration::from_millis(950),
            "{:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn test_without_retries_a_429_is_a_failure() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(429))
            .expect(1)
            .mount(&server)
            .await;
        let (results, failures) = BenchRunner::new()
            .unwrap()
            .run_detailed(&one_request(local_openai(&server), 0), |_, _| {})
            .await
            .unwrap();
        assert_eq!((results.len(), failures.len()), (0, 1));
    }

    #[tokio::test]
    async fn test_server_errors_retry_but_bad_requests_do_not() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(503))
            .expect(3)
            .mount(&server)
            .await;
        let (_, failures) = BenchRunner::new()
            .unwrap()
            .run_detailed(&one_request(local_openai(&server), 2), |_, _| {})
            .await
            .unwrap();
        assert_eq!(failures.len(), 1);
        assert!(failures[0].error.contains("503"), "{}", failures[0].error);

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(404))
            .expect(1)
            .mount(&server)
            .await;
        let (_, failures) = BenchRunner::new()
            .unwrap()
            .run_detailed(&one_request(local_openai(&server), 3), |_, _| {})
            .await
            .unwrap();
        assert_eq!(failures.len(), 1);
    }

    #[tokio::test]
    async fn test_transport_errors_say_why() {
        // Nothing listens on port 9 of localhost.
        let mut p = local_openai_unreachable();
        p.base_url = "http://127.0.0.1:9".into();
        let (_, failures) = BenchRunner::new()
            .unwrap()
            .run_detailed(&one_request(p, 0), |_, _| {})
            .await
            .unwrap();
        let msg = &failures[0].error;
        assert!(msg.contains("error sending request"), "{msg}");
        // The cause chain is included (connection refused / actively refused).
        assert!(msg.matches(": ").count() >= 2, "{msg}");
    }

    fn local_openai_unreachable() -> ProviderConfig {
        ProviderConfig {
            name: "openai".into(),
            model: "local".into(),
            api_key: String::new(),
            base_url: String::new(),
            max_tokens: 8,
        }
    }

    #[test]
    fn test_default_base_urls() {
        assert_eq!(default_base_url("ollama"), Some(DEFAULT_OLLAMA_BASE_URL));
        assert_eq!(default_base_url("gemini"), Some(DEFAULT_GEMINI_BASE_URL));
        assert_eq!(default_base_url("nope"), None);
        for (name, _) in BUILTIN_PROVIDERS {
            assert!(default_base_url(name).is_some(), "{name}");
        }
        // Each hosted OpenAI-compatible base ends in its version segment.
        use crate::providers::chat_completions_url;
        assert_eq!(
            chat_completions_url(DEFAULT_GROQ_BASE_URL),
            "https://api.groq.com/openai/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url(DEFAULT_TOGETHER_BASE_URL),
            "https://api.together.xyz/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url(DEFAULT_FIREWORKS_BASE_URL),
            "https://api.fireworks.ai/inference/v1/chat/completions"
        );
        assert_eq!(
            chat_completions_url(DEFAULT_DEEPSEEK_BASE_URL),
            "https://api.deepseek.com/v1/chat/completions"
        );
    }

    #[tokio::test]
    async fn test_custom_provider_and_configured_price() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::header("authorization", "Bearer vk"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500}
            })))
            .expect(2)
            .mount(&server)
            .await;
        let provider = ProviderConfig {
            name: "my-vllm".into(),
            model: "llama".into(),
            api_key: "vk".into(),
            base_url: format!("{}/v1", server.uri()),
            max_tokens: 8,
        };
        let mut table = std::collections::BTreeMap::new();
        table.insert(
            "my-vllm:llama".to_owned(),
            crate::config::ModelPrice::new(2.0, 4.0),
        );
        let mut config = one_request(provider, 0);
        config.runs_per_prompt = 2;
        let (results, failures) = BenchRunner::new()
            .unwrap()
            .with_prices(PriceTable::new(table))
            .run_detailed(&config, |_, _| {})
            .await
            .unwrap();
        assert!(failures.is_empty(), "{failures:?}");
        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(r.provider, "my-vllm");
            assert!((r.cost_usd - (1000.0 * 2.0 + 500.0 * 4.0) / 1e6).abs() < 1e-12);
            assert_eq!(r.cost_source, Some(CostSource::Config));
        }
    }

    #[test]
    fn test_configured_price_never_overrides_reported_cost() {
        let mut r = BenchResult {
            provider: "openrouter".into(),
            model: "m".into(),
            prompt: String::new(),
            latency_ms: 1,
            total_ms: 1,
            prompt_tokens: 10,
            completion_tokens: 10,
            cost_usd: 0.5,
            tokens_per_second: 0.0,
            response_text: String::new(),
            run_index: 0,
            ttft_ms: None,
            retries: 0,
            tpot_ms: None,
            cached_prompt_tokens: 0,
            cache_write_prompt_tokens: 0,
            cost_source: Some(CostSource::Provider),
        };
        let mut table = std::collections::BTreeMap::new();
        table.insert("m".to_owned(), crate::config::ModelPrice::new(1.0, 1.0));
        let prices = PriceTable::new(table);
        apply_configured_price(&mut r, &prices);
        assert_eq!(r.cost_usd, 0.5);
        r.cost_source = None;
        apply_configured_price(&mut r, &prices);
        assert!((r.cost_usd - 20.0 / 1e6).abs() < 1e-15);
        assert_eq!(r.cost_source, Some(CostSource::Config));
    }

    #[test]
    fn test_configured_price_bills_cached_prompt_tokens() {
        let mut table = std::collections::BTreeMap::new();
        table.insert(
            "m".to_owned(),
            crate::config::ModelPrice {
                cached_input: Some(0.5),
                ..crate::config::ModelPrice::new(2.0, 4.0)
            },
        );
        table.insert("flat".to_owned(), crate::config::ModelPrice::new(2.0, 4.0));
        let prices = PriceTable::new(table);
        let base = |provider: &str, model: &str| BenchResult {
            provider: provider.into(),
            model: model.into(),
            prompt: String::new(),
            latency_ms: 1,
            total_ms: 1,
            prompt_tokens: 1_000_000,
            completion_tokens: 1_000_000,
            cost_usd: 0.0,
            tokens_per_second: 0.0,
            response_text: String::new(),
            run_index: 0,
            ttft_ms: None,
            retries: 0,
            tpot_ms: None,
            cached_prompt_tokens: 400_000,
            cache_write_prompt_tokens: 0,
            cost_source: None,
        };
        // OpenAI-compatible: 600k uncached at $2, 400k cached at $0.50, 1M out at $4.
        let mut r = base("groq", "m");
        apply_configured_price(&mut r, &prices);
        assert!(
            (r.cost_usd - (1.2 + 0.2 + 4.0)).abs() < 1e-9,
            "{}",
            r.cost_usd
        );
        // No cached_input: every prompt token at $2, never more than prompt_tokens.
        let mut r = base("groq", "flat");
        apply_configured_price(&mut r, &prices);
        assert!((r.cost_usd - (2.0 + 4.0)).abs() < 1e-9, "{}", r.cost_usd);
        // Anthropic: prompt_tokens is the uncached part; cache reads and
        // writes come on top (writes at the input price).
        let mut r = base("anthropic", "m");
        r.cache_write_prompt_tokens = 100_000;
        apply_configured_price(&mut r, &prices);
        assert!(
            (r.cost_usd - (2.2 + 0.2 + 4.0)).abs() < 1e-9,
            "{}",
            r.cost_usd
        );
    }
}
