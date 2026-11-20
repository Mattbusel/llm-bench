//! Compare every provider you have a key for (plus a local Ollama if it is
//! running) on the same prompts, and print the comparison table as
//! Markdown: time to first token, p50/p95 latency, tokens per second, error
//! rate and cost per request.
//!
//! ```text
//! cargo run --example compare_providers
//! ```
//!
//! Providers without a key are skipped, so this runs anywhere; with no keys
//! and no Ollama it says so and exits.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use llm_bench::report::markdown_table;
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
        provider(
            "openrouter",
            "meta-llama/llama-3.3-70b-instruct",
            Some("OPENROUTER_API_KEY"),
        ),
        provider("groq", "llama-3.3-70b-versatile", Some("GROQ_API_KEY")),
        provider(
            "together",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            Some("TOGETHER_API_KEY"),
        ),
        provider(
            "fireworks",
            "accounts/fireworks/models/llama-v3p3-70b-instruct",
            Some("FIREWORKS_API_KEY"),
        ),
        provider("deepseek", "deepseek-v4-flash", Some("DEEPSEEK_API_KEY")),
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
    // Groq, Together and Fireworks show n/a for cost unless you price them
    // with `BenchRunner::with_prices` (the `[prices]` of a config file).
    print!("{}", markdown_table(&model_reports(&results, &failures)));
    Ok(())
}
