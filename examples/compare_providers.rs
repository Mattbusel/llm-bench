//! Compare every provider you have a key for (plus a local Ollama if it is
//! running) on the same prompts, and print cost and speed side by side.
//!
//! ```text
//! cargo run --example compare_providers
//! ```
//!
//! Providers without a key are skipped, so this runs anywhere; with no keys
//! and no Ollama it says so and exits.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use llm_bench::runner::BenchRunner;
use llm_bench::stats::model_reports;
use llm_bench::types::{BenchConfig, ProviderConfig};

fn provider(name: &str, model: &str, key_var: Option<&str>) -> Option<ProviderConfig> {
    let api_key = match key_var {
        Some(var) => std::env::var(var).ok().filter(|k| !k.trim().is_empty())?,
        None => String::new(),
    };
    Some(ProviderConfig {
        name: name.into(),
        model: model.into(),
        api_key,
        base_url: String::new(),
        max_tokens: 128,
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut providers: Vec<ProviderConfig> = [
        provider("openai", "gpt-4o-mini", Some("OPENAI_API_KEY")),
        provider("anthropic", "claude-haiku-4-5", Some("ANTHROPIC_API_KEY")),
        provider("gemini", "gemini-2.5-flash", Some("GEMINI_API_KEY")),
        provider("openrouter", "meta-llama/llama-3.3-70b-instruct", Some("OPENROUTER_API_KEY")),
    ]
    .into_iter()
    .flatten()
    .collect();
    let ollama_up = reqwest::Client::new()
        .get("http://localhost:11434/api/tags")
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
        .is_ok();
    if ollama_up {
        providers.extend(provider("ollama", "qwen2.5:0.5b", None));
    }
    if providers.is_empty() {
        println!("No API keys set and no Ollama on localhost:11434; nothing to compare.");
        return Ok(());
    }

    let config = BenchConfig {
        prompts: vec![
            "Explain what a mutex is in two sentences.".into(),
            "Write a haiku about latency.".into(),
        ],
        runs_per_prompt: 3,
        concurrency: 4,
        providers,
        stream: true,
        retries: 2,
        warmup: 0,
    };
    let (results, failures) = BenchRunner::new()?.run_detailed(&config, |_, _| {}).await?;
    println!("{:<40} {:>10} {:>10} {:>12}", "model", "TTFT p50", "tok/s", "cost/request");
    for m in model_reports(&results, &failures) {
        let ttft = m.ttft_ms.map(|s| format!("{:.0} ms", s.p50)).unwrap_or_else(|| "-".into());
        let tps = m.tokens_per_second.map(|s| format!("{:.1}", s.mean)).unwrap_or_else(|| "-".into());
        let cost = if m.succeeded > 0 && m.total_cost_usd > 0.0 {
            format!("${:.6}", m.total_cost_usd / m.succeeded as f64)
        } else {
            "n/a".into()
        };
        println!("{:<40} {:>10} {:>10} {:>12}", format!("{}:{}", m.provider, m.model), ttft, tps, cost);
    }
    Ok(())
}
