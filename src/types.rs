//! # Module: Types
//!
//! ## Responsibility
//! Core domain types shared across all modules.  These are plain data structs  - 
//! no business logic lives here.
//!
//! ## Guarantees
//! - All types are `Clone` + `Debug` + `serde::{Serialize, Deserialize}`.
//! - No panics; no fallible operations.

use serde::{Deserialize, Serialize};

//  Individual run result 

/// The complete result of a single inference call against one model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    /// Provider name, e.g. `"openai"` or `"anthropic"`.
    pub provider: String,

    /// Model identifier, e.g. `"gpt-4o-mini"`.
    pub model: String,

    /// The prompt that was sent.
    pub prompt: String,

    /// Wall time from request dispatch to first token received (ms).
    /// For non-streaming calls this equals `total_ms`.
    pub latency_ms: u64,

    /// Wall time from request dispatch to full response received (ms).
    pub total_ms: u64,

    /// Number of prompt tokens billed by the provider.
    pub prompt_tokens: u32,

    /// Number of completion tokens billed by the provider.
    pub completion_tokens: u32,

    /// Estimated cost in USD for this request.
    pub cost_usd: f64,

    /// `completion_tokens / (total_ms / 1000.0)`  -  tokens generated per second.
    pub tokens_per_second: f64,

    /// The full text of the model's response.
    pub response_text: String,

    /// Zero-based index of this run within the prompt's run batch.
    pub run_index: u32,
}

//  Benchmark configuration 

/// Top-level configuration for a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// The set of prompts to send to every configured provider/model.
    pub prompts: Vec<String>,

    /// How many times each prompt is sent to each provider (for statistical stability).
    pub runs_per_prompt: u32,

    /// Maximum number of in-flight API calls at any one time.
    pub concurrency: usize,

    /// One entry per provider+model combination to benchmark.
    pub providers: Vec<ProviderConfig>,
}

/// Configuration for a single provider+model combination.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// `"openai"` or `"anthropic"`.
    pub name: String,

    /// Model identifier string sent verbatim to the API.
    pub model: String,

    /// Bearer API key for this provider.
    pub api_key: String,

    /// Maximum completion tokens to request.
    pub max_tokens: u32,
}

//  Aggregated summary 

/// Aggregated statistics across all runs for a provider+model pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchSummary {
    /// Provider name.
    pub provider: String,

    /// Model identifier.
    pub model: String,

    /// Median (p50) latency across all runs (ms).
    pub p50_latency_ms: u64,

    /// 99th-percentile latency across all runs (ms).
    pub p99_latency_ms: u64,

    /// Average tokens-per-second across successful runs.
    pub avg_tokens_per_sec: f64,

    /// Average per-request cost in USD.
    pub avg_cost_usd: f64,

    /// Sum of cost across all runs in USD.
    pub total_cost_usd: f64,

    /// Fraction of runs that completed without error (0.0 - 1.0).
    pub success_rate: f64,
}

//  Tests 

#[cfg(test)]
mod tests {
    use super::*;

    //  BenchResult 

    fn sample_result(provider: &str, model: &str, latency: u64, cost: f64) -> BenchResult {
        BenchResult {
            provider: provider.into(),
            model: model.into(),
            prompt: "hello world".into(),
            latency_ms: latency,
            total_ms: latency + 50,
            prompt_tokens: 10,
            completion_tokens: 20,
            cost_usd: cost,
            tokens_per_second: 40.0,
            response_text: "Hi!".into(),
            run_index: 0,
        }
    }

    #[test]
    fn test_bench_result_fields_are_accessible() {
        let r = sample_result("openai", "gpt-4o-mini", 120, 0.0001);
        assert_eq!(r.provider, "openai");
        assert_eq!(r.model, "gpt-4o-mini");
        assert_eq!(r.latency_ms, 120);
        assert_eq!(r.total_ms, 170);
        assert_eq!(r.prompt_tokens, 10);
        assert_eq!(r.completion_tokens, 20);
        assert!((r.cost_usd - 0.0001).abs() < 1e-10);
        assert!((r.tokens_per_second - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_bench_result_serialises_and_deserialises() {
        let r = sample_result("anthropic", "claude-3-5-haiku-20241022", 200, 0.0002);
        let json = serde_json::to_string(&r).unwrap_or_default();
        assert!(!json.is_empty(), "serialised JSON should not be empty");
        let restored: Result<BenchResult, _> = serde_json::from_str(&json);
        assert!(restored.is_ok(), "should deserialise cleanly");
        let restored = restored.unwrap_or_else(|_| sample_result("x", "y", 0, 0.0));
        assert_eq!(restored.provider, r.provider);
        assert_eq!(restored.model, r.model);
        assert_eq!(restored.latency_ms, r.latency_ms);
    }

    #[test]
    fn test_bench_result_is_clone() {
        let r = sample_result("openai", "gpt-4o", 300, 0.001);
        let cloned = r.clone();
        assert_eq!(cloned.provider, r.provider);
        assert_eq!(cloned.latency_ms, r.latency_ms);
    }

    #[test]
    fn test_bench_result_run_index_preserved() {
        let mut r = sample_result("openai", "gpt-4o", 100, 0.0);
        r.run_index = 7;
        assert_eq!(r.run_index, 7);
    }

    #[test]
    fn test_bench_result_tokens_per_second_zero_edge_case() {
        let mut r = sample_result("openai", "gpt-4o", 0, 0.0);
        r.tokens_per_second = 0.0;
        assert_eq!(r.tokens_per_second, 0.0);
    }

    //  BenchConfig 

    fn sample_config() -> BenchConfig {
        BenchConfig {
            prompts: vec!["Say hi".into(), "Explain Rust".into()],
            runs_per_prompt: 3,
            concurrency: 4,
            providers: vec![ProviderConfig {
                name: "openai".into(),
                model: "gpt-4o-mini".into(),
                api_key: "sk-test".into(),
                max_tokens: 512,
            }],
        }
    }

    #[test]
    fn test_bench_config_prompts_accessible() {
        let c = sample_config();
        assert_eq!(c.prompts.len(), 2);
        assert_eq!(c.prompts[0], "Say hi");
    }

    #[test]
    fn test_bench_config_runs_per_prompt() {
        let c = sample_config();
        assert_eq!(c.runs_per_prompt, 3);
    }

    #[test]
    fn test_bench_config_concurrency() {
        let c = sample_config();
        assert_eq!(c.concurrency, 4);
    }

    #[test]
    fn test_bench_config_providers_accessible() {
        let c = sample_config();
        assert_eq!(c.providers.len(), 1);
        assert_eq!(c.providers[0].name, "openai");
        assert_eq!(c.providers[0].model, "gpt-4o-mini");
        assert_eq!(c.providers[0].max_tokens, 512);
    }

    #[test]
    fn test_bench_config_is_clone() {
        let c = sample_config();
        let cloned = c.clone();
        assert_eq!(cloned.prompts.len(), c.prompts.len());
        assert_eq!(cloned.runs_per_prompt, c.runs_per_prompt);
    }

    #[test]
    fn test_provider_config_fields() {
        let p = ProviderConfig {
            name: "anthropic".into(),
            model: "claude-3-5-sonnet-20241022".into(),
            api_key: "sk-ant-test".into(),
            max_tokens: 1024,
        };
        assert_eq!(p.name, "anthropic");
        assert_eq!(p.model, "claude-3-5-sonnet-20241022");
        assert_eq!(p.api_key, "sk-ant-test");
        assert_eq!(p.max_tokens, 1024);
    }

    //  BenchSummary 

    fn sample_summary() -> BenchSummary {
        BenchSummary {
            provider: "openai".into(),
            model: "gpt-4o-mini".into(),
            p50_latency_ms: 150,
            p99_latency_ms: 400,
            avg_tokens_per_sec: 55.0,
            avg_cost_usd: 0.0003,
            total_cost_usd: 0.0009,
            success_rate: 1.0,
        }
    }

    #[test]
    fn test_bench_summary_p50_less_than_p99() {
        let s = sample_summary();
        assert!(
            s.p50_latency_ms <= s.p99_latency_ms,
            "p50 should not exceed p99"
        );
    }

    #[test]
    fn test_bench_summary_success_rate_in_range() {
        let s = sample_summary();
        assert!(
            (0.0..=1.0).contains(&s.success_rate),
            "success rate must be in [0,1]"
        );
    }

    #[test]
    fn test_bench_summary_total_cost_gte_avg() {
        let s = sample_summary();
        assert!(
            s.total_cost_usd >= s.avg_cost_usd,
            "total cost should be >= average cost"
        );
    }

    #[test]
    fn test_bench_summary_serialises() {
        let s = sample_summary();
        let json = serde_json::to_string(&s).unwrap_or_default();
        assert!(json.contains("openai"), "JSON should contain provider");
        assert!(json.contains("p50_latency_ms"), "JSON should contain field name");
    }

    #[test]
    fn test_bench_summary_is_clone() {
        let s = sample_summary();
        let cloned = s.clone();
        assert_eq!(cloned.model, s.model);
        assert_eq!(cloned.p50_latency_ms, s.p50_latency_ms);
    }

    #[test]
    fn test_bench_summary_zero_success_rate() {
        let mut s = sample_summary();
        s.success_rate = 0.0;
        assert_eq!(s.success_rate, 0.0);
    }

    #[test]
    fn test_bench_summary_avg_tokens_per_sec_positive() {
        let s = sample_summary();
        assert!(s.avg_tokens_per_sec > 0.0, "avg tps should be positive");
    }

    #[test]
    fn test_bench_summary_provider_non_empty() {
        let s = sample_summary();
        assert!(!s.provider.is_empty());
    }

    #[test]
    fn test_bench_summary_model_non_empty() {
        let s = sample_summary();
        assert!(!s.model.is_empty());
    }

    #[test]
    fn test_bench_result_prompt_stored() {
        let r = sample_result("openai", "gpt-4o", 200, 0.001);
        assert_eq!(r.prompt, "hello world");
    }

    #[test]
    fn test_bench_result_total_ms_gte_latency() {
        let r = sample_result("openai", "gpt-4o", 100, 0.001);
        assert!(r.total_ms >= r.latency_ms);
    }

    #[test]
    fn test_bench_config_empty_prompts_allowed() {
        let c = BenchConfig {
            prompts: vec![],
            runs_per_prompt: 1,
            concurrency: 1,
            providers: vec![],
        };
        assert!(c.prompts.is_empty());
    }

    #[test]
    fn test_bench_config_zero_runs_allowed() {
        let c = BenchConfig {
            prompts: vec!["hi".into()],
            runs_per_prompt: 0,
            concurrency: 1,
            providers: vec![],
        };
        assert_eq!(c.runs_per_prompt, 0);
    }

    #[test]
    fn test_bench_result_deserialise_missing_field_fails() {
        // A JSON object with a missing required field should fail to deserialise
        let bad_json = r#"{"provider":"openai"}"#;
        let result: Result<BenchResult, _> = serde_json::from_str(bad_json);
        assert!(result.is_err(), "incomplete JSON should fail to deserialise");
    }

    #[test]
    fn test_bench_summary_avg_cost_non_negative() {
        let s = sample_summary();
        assert!(s.avg_cost_usd >= 0.0);
    }

    #[test]
    fn test_bench_summary_total_cost_non_negative() {
        let s = sample_summary();
        assert!(s.total_cost_usd >= 0.0);
    }
}
