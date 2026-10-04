//! A latency gate for CI: benchmark one model and exit with status 1 when
//! its p95 time to first token is above a limit.
//!
//! ```text
//! cargo run --example latency_gate -- ollama:qwen2.5:0.5b 800
//! OPENAI_API_KEY=sk-... cargo run --example latency_gate -- openai:gpt-4o-mini 1500
//! ```
//!
//! Prefixes: `openai:` (OPENAI_API_KEY), `anthropic:` (ANTHROPIC_API_KEY),
//! `ollama:` (no key; OLLAMA_HOST or localhost:11434), `openrouter:`
//! (OPENROUTER_API_KEY), `gemini:` (GEMINI_API_KEY).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use llm_bench::runner::BenchRunner;
use llm_bench::stats::model_reports;
use llm_bench::types::{BenchConfig, ProviderConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let target = args.next().unwrap_or_else(|| "ollama:qwen2.5:0.5b".into());
    let limit_ms: f64 = args.next().map(|s| s.parse()).transpose()?.unwrap_or(1000.0);
    let (provider, model) = target.split_once(':').ok_or("use provider:model")?;
    let key_var = match provider {
        "openai" => "OPENAI_API_KEY",
        "anthropic" => "ANTHROPIC_API_KEY",
        "openrouter" => "OPENROUTER_API_KEY",
        "gemini" => "GEMINI_API_KEY",
        _ => "",
    };
    let base_url = match provider {
        "ollama" => std::env::var("OLLAMA_HOST")
            .map(|h| if h.contains("://") { h } else { format!("http://{h}") })
            .unwrap_or_default(),
        _ => String::new(),
    };
    let config = BenchConfig {
        prompts: vec!["Reply with one word: ready".into()],
        runs_per_prompt: 10,
        concurrency: 2,
        providers: vec![ProviderConfig {
            name: provider.into(),
            model: model.into(),
            api_key: std::env::var(key_var).unwrap_or_default(),
            base_url,
            max_tokens: 16,
        }],
        stream: true,
        retries: 2,
        warmup: 1,
    };
    let (results, failures) = BenchRunner::new()?.run_detailed(&config, |_, _| {}).await?;
    let report = model_reports(&results, &failures)
        .into_iter()
        .next()
        .ok_or("no results")?;
    let Some(ttft) = report.ttft_ms else {
        eprintln!("{model}: every request failed: {failures:?}");
        std::process::exit(2);
    };
    println!("{model}: p95 time to first token {} ms (limit {limit_ms} ms), {} failed", ttft.p95, report.failed);
    if ttft.p95 > limit_ms || report.failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}
