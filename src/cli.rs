//! # Module: CLI
//!
//! ## Responsibility
//! Define and parse command-line arguments using `clap` derive macros.
//! Map parsed arguments into domain types (`BenchConfig`, `ProviderConfig`).
//!
//! ## Guarantees
//! - All required values are validated at parse time.
//! - Environment variable fallbacks are applied by `clap`.
//! - This module never panics.
//!
//! ## NOT Responsible For
//! - Making network calls (see `runner.rs` / `providers.rs`)
//! - Output rendering (see `report.rs`)

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use crate::error::BenchError;
use crate::types::{BenchConfig, ProviderConfig};

/// Examples shown at the bottom of `llm-bench --help`.
const HELP_EXAMPLES: &str = "Examples:
  llm-bench models
      List the models llm-bench knows prices for.

  export OPENAI_API_KEY=sk-...   (PowerShell: $env:OPENAI_API_KEY=\"sk-...\")
  llm-bench run --models gpt-4o-mini --prompts \"Say hi\"
      One model, one prompt, 3 runs, results table.

  llm-bench run --models gpt-4o-mini,claude-haiku-4-5 --prompt-file prompts.txt --runs 5
      Compare two providers on your own prompts.

  llm-bench run --models ollama:llama3.2 --prompts \"Say hi\" --stream
      A local Ollama model, no key needed; --stream adds time to first token.

  llm-bench run --models openrouter:meta-llama/llama-3.3-70b-instruct,gemini:gemini-2.5-flash --prompts \"Say hi\"
      OpenRouter (OPENROUTER_API_KEY) and Gemini (GEMINI_API_KEY).

  llm-bench run --openai-base-url http://localhost:8000 --models openai:my-model --prompts \"Say hi\"
      Any OpenAI-compatible server (vLLM, LM Studio, a proxy); no key needed.";

//  Top-level CLI

/// Universal LLM provider benchmark CLI.
///
/// Compare OpenAI and Anthropic models on latency, cost, quality, and
/// token throughput from the command line.
#[derive(Debug, Parser)]
#[command(
    name = "llm-bench",
    version,
    about = "Benchmark LLMs (OpenAI, Anthropic, Ollama, OpenRouter, Gemini, any OpenAI-compatible server) on latency, cost and throughput",
    long_about = None,
    after_help = HELP_EXAMPLES
)]
pub struct Cli {
    /// The sub-command to run.
    #[command(subcommand)]
    pub command: Command,
}

/// Available sub-commands.
#[derive(Debug, Subcommand)]
#[allow(clippy::large_enum_variant)]
pub enum Command {
    /// Run a benchmark against configured providers and models.
    Run(RunArgs),

    /// List all supported models with pricing information.
    Models,

    /// Print the tool version.
    Version,
}

//  `run` sub-command

/// Arguments for the `run` sub-command.
#[derive(Debug, Parser)]
pub struct RunArgs {
    /// OpenAI API key. Falls back to OPENAI_API_KEY environment variable.
    #[arg(long, env = "OPENAI_API_KEY", hide_env_values = true)]
    pub openai_key: Option<String>,

    /// Anthropic API key. Falls back to ANTHROPIC_API_KEY environment variable.
    #[arg(long, env = "ANTHROPIC_API_KEY", hide_env_values = true)]
    pub anthropic_key: Option<String>,

    /// Base URL of an OpenAI-compatible API (Ollama, vLLM, LM Studio, a proxy).
    /// Accepts either `http://host:port` or `http://host:port/v1`. When set,
    /// an API key is optional.
    #[arg(long, env = "OPENAI_BASE_URL")]
    pub openai_base_url: Option<String>,

    /// Base URL for the Anthropic Messages API (for proxies or gateways).
    #[arg(long, env = "ANTHROPIC_BASE_URL")]
    pub anthropic_base_url: Option<String>,

    /// OpenRouter API key, for `openrouter:` models.
    #[arg(long, env = "OPENROUTER_API_KEY", hide_env_values = true)]
    pub openrouter_key: Option<String>,

    /// Google Gemini API key, for `gemini:` models.
    #[arg(long, env = "GEMINI_API_KEY", hide_env_values = true)]
    pub gemini_key: Option<String>,

    /// Ollama server for `ollama:` models (default http://localhost:11434).
    /// `OLLAMA_HOST` is read too, with or without `http://`.
    #[arg(long, env = "OLLAMA_HOST")]
    pub ollama_url: Option<String>,

    /// Comma-separated list of model identifiers to benchmark.
    ///
    /// Prefix with the provider: `openai:`, `anthropic:`, `ollama:`,
    /// `openrouter:` or `gemini:`. Bare `gpt-*`, `o1*`, `o3*`, `o4*`,
    /// `claude-*` and `gemini-*` names are matched automatically.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "gpt-4o-mini,claude-haiku-4-5"
    )]
    pub models: Vec<String>,

    /// Inline prompts (comma-separated).  Use `--prompt-file` for many prompts.
    #[arg(long, value_delimiter = ',')]
    pub prompts: Vec<String>,

    /// Path to a file containing one prompt per line.
    #[arg(long)]
    pub prompt_file: Option<PathBuf>,

    /// Number of times each prompt is sent to each model.
    #[arg(long, default_value_t = 3)]
    pub runs: u32,

    /// Maximum number of concurrent in-flight API calls.
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,

    /// Output format.
    #[arg(long, default_value = "table")]
    pub output: OutputFormat,

    /// Save full JSON results to this file.
    #[arg(long)]
    pub output_file: Option<PathBuf>,

    /// Maximum completion tokens per request.
    #[arg(long, default_value_t = 512)]
    pub max_tokens: u32,

    /// Stream responses to measure time to first token. Latency columns
    /// then show time to first token instead of time to the full answer.
    #[arg(long)]
    pub stream: bool,

    /// Retry a request this many times on HTTP 429, 5xx, timeouts and
    /// connection errors (backoff with jitter; Retry-After is honoured).
    #[arg(long, default_value_t = 0)]
    pub retries: u32,

    /// Send this many requests to each model first and discard them (cold
    /// starts, local models loading into memory).
    #[arg(long, default_value_t = 0)]
    pub warmup: u32,

    /// Write per-model statistics (mean, stddev, p50/p90/p95/p99 of latency,
    /// TTFT, total time and time per output token) to this JSON file.
    #[arg(long)]
    pub summary_file: Option<PathBuf>,
}

/// Output format selector.
#[derive(Debug, Clone, ValueEnum)]
pub enum OutputFormat {
    /// Pretty ASCII table (default).
    Table,
    /// JSON array of all BenchResult records.
    Json,
    /// CSV, one row per successful request.
    Csv,
}

//  Config builder

/// Build a `BenchConfig` from parsed `RunArgs`.
///
/// Resolves prompts from both inline `--prompts` and `--prompt-file`,
/// and maps model strings to `ProviderConfig` entries.
///
/// # Errors
/// Returns `BenchError::InvalidConfig` if:
/// - No prompts are provided via either flag.
/// - A model string cannot be resolved to a known provider.
/// - An API key is missing for a required provider.
///
/// # Panics
/// This function never panics.
pub fn build_config(args: &RunArgs) -> Result<BenchConfig, BenchError> {
    //  Collect prompts
    let mut prompts: Vec<String> = args.prompts.clone();

    if let Some(ref path) = args.prompt_file {
        let content = std::fs::read_to_string(path).map_err(BenchError::Io)?;
        for line in content.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                prompts.push(trimmed.to_owned());
            }
        }
    }

    if prompts.is_empty() {
        return Err(BenchError::InvalidConfig {
            reason: "No prompts provided. Use --prompts or --prompt-file.".into(),
        });
    }

    //  Resolve providers
    let mut providers: Vec<ProviderConfig> = Vec::new();

    for model_str in &args.models {
        let model_str = model_str.trim();
        if model_str.is_empty() {
            continue;
        }

        let (provider_name, model) = resolve_model(model_str)?;

        let key = |k: &Option<String>| k.as_deref().unwrap_or("").trim().to_owned();
        let (api_key, base_url, env_var) = match provider_name {
            "openai" => (
                key(&args.openai_key),
                normalize_base_url(args.openai_base_url.as_deref()),
                "OPENAI_API_KEY",
            ),
            "anthropic" => (
                key(&args.anthropic_key),
                normalize_base_url(args.anthropic_base_url.as_deref()),
                "ANTHROPIC_API_KEY",
            ),
            "openrouter" => (key(&args.openrouter_key), String::new(), "OPENROUTER_API_KEY"),
            "gemini" => (key(&args.gemini_key), String::new(), "GEMINI_API_KEY"),
            "ollama" => (
                String::new(),
                normalize_ollama_url(args.ollama_url.as_deref()),
                "",
            ),
            other => {
                return Err(BenchError::InvalidConfig {
                    reason: format!("unknown provider '{other}'"),
                })
            }
        };

        // Hosted endpoints need a key; a custom base URL (local server,
        // proxy) and Ollama may not.
        if api_key.is_empty() && base_url.is_empty() && provider_name != "ollama" {
            return Err(BenchError::InvalidConfig {
                reason: format!(
                    "no API key for {provider_name} (needed for model '{model}').\n  \
                     Fix: set {env_var} (bash: export {env_var}=...  PowerShell: $env:{env_var}=\"...\")\n  \
                     or pass --{provider_name}-key <KEY>, or leave '{model}' out of --models."
                ),
            });
        }

        providers.push(ProviderConfig {
            name: provider_name.to_owned(),
            model: model.to_owned(),
            api_key,
            base_url,
            max_tokens: args.max_tokens,
        });
    }

    if providers.is_empty() {
        return Err(BenchError::InvalidConfig {
            reason: "No valid provider+model combinations resolved from --models.".into(),
        });
    }

    Ok(BenchConfig {
        prompts,
        runs_per_prompt: args.runs,
        concurrency: args.concurrency,
        providers,
        stream: args.stream,
        retries: args.retries,
        warmup: args.warmup,
    })
}

/// Normalise a user-supplied base URL: trim whitespace, trailing slashes and a
/// trailing `/v1`, because the providers append `/v1/...` themselves.
/// `None` or blank input yields an empty string (meaning "official URL").
///
/// # Panics
/// This function never panics.
fn normalize_base_url(raw: Option<&str>) -> String {
    let mut s = raw.unwrap_or("").trim().trim_end_matches('/').to_owned();
    if s.ends_with("/v1") {
        s.truncate(s.len() - 3);
    }
    s.trim_end_matches('/').to_owned()
}

/// Ollama's address from `--ollama-url` / `OLLAMA_HOST`. `OLLAMA_HOST` is
/// often a bind address such as `0.0.0.0:11434` without a scheme; that is
/// turned into `http://localhost:11434`. Empty means the default.
fn normalize_ollama_url(raw: Option<&str>) -> String {
    let raw = raw.unwrap_or("").trim();
    if raw.is_empty() {
        return String::new();
    }
    let with_scheme = if raw.contains("://") {
        raw.to_owned()
    } else {
        format!("http://{raw}")
    };
    let fixed = with_scheme.replace("://0.0.0.0", "://localhost");
    normalize_base_url(Some(&fixed))
}

/// Provider prefixes accepted in `--models`.
const PREFIXES: [&str; 5] = ["openai", "anthropic", "ollama", "openrouter", "gemini"];

/// Resolve a model string to `(provider_name, model_id)`.
///
/// Accepts:
/// - `"openai:gpt-4o-mini"` → `("openai", "gpt-4o-mini")`
/// - `"anthropic:claude-haiku-4-5"` → `("anthropic", "claude-haiku-4-5")`
/// - Bare names are matched against known model prefixes.
///
/// # Panics
/// This function never panics.
fn resolve_model(s: &str) -> Result<(&'static str, &str), BenchError> {
    for p in PREFIXES {
        if let Some(rest) = s.strip_prefix(p).and_then(|r| r.strip_prefix(':')) {
            if !rest.is_empty() {
                return Ok((p, rest));
            }
        }
    }

    // Auto-detect from known model name prefixes
    if s.starts_with("gpt-")
        || s.starts_with("chatgpt-")
        || s.starts_with("o1")
        || s.starts_with("o3")
        || s.starts_with("o4")
    {
        return Ok(("openai", s));
    }
    if s.starts_with("claude-") {
        return Ok(("anthropic", s));
    }
    if s.starts_with("gemini-") {
        return Ok(("gemini", s));
    }

    Err(BenchError::InvalidConfig {
        reason: format!(
            "Cannot determine provider for model '{s}'. \
             Prefix it with openai:, anthropic:, ollama:, openrouter: or gemini:, e.g. ollama:{s}"
        ),
    })
}

//  Tests

#[cfg(test)]
mod tests {
    use super::*;

    //  resolve_model

    #[test]
    fn test_resolve_model_openai_prefix() {
        let (provider, model) = resolve_model("openai:gpt-4o").unwrap_or(("", ""));
        assert_eq!(provider, "openai");
        assert_eq!(model, "gpt-4o");
    }

    #[test]
    fn test_resolve_model_anthropic_prefix() {
        let (provider, model) = resolve_model("anthropic:claude-haiku-4-5").unwrap_or(("", ""));
        assert_eq!(provider, "anthropic");
        assert_eq!(model, "claude-haiku-4-5");
    }

    #[test]
    fn test_resolve_model_bare_gpt_autodetects_openai() {
        let (provider, model) = resolve_model("gpt-4o-mini").unwrap_or(("", ""));
        assert_eq!(provider, "openai");
        assert_eq!(model, "gpt-4o-mini");
    }

    #[test]
    fn test_resolve_model_bare_claude_autodetects_anthropic() {
        let (provider, model) = resolve_model("claude-haiku-4-5").unwrap_or(("", ""));
        assert_eq!(provider, "anthropic");
        assert_eq!(model, "claude-haiku-4-5");
    }

    #[test]
    fn test_resolve_model_bare_o1_autodetects_openai() {
        let (provider, model) = resolve_model("o1-mini").unwrap_or(("", ""));
        assert_eq!(provider, "openai");
        assert_eq!(model, "o1-mini");
    }

    #[test]
    fn test_resolve_model_unknown_returns_error() {
        let result = resolve_model("llama-3-70b");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BenchError::InvalidConfig { .. }
        ));
    }

    #[test]
    fn test_normalize_base_url_strips_v1_and_slashes() {
        assert_eq!(
            normalize_base_url(Some("http://localhost:11434/v1/")),
            "http://localhost:11434"
        );
        assert_eq!(normalize_base_url(Some(" http://x:1/ ")), "http://x:1");
        assert_eq!(normalize_base_url(None), "");
        assert_eq!(normalize_base_url(Some("   ")), "");
    }

    #[test]
    fn test_build_config_custom_base_url_allows_missing_key() {
        let mut args = minimal_run_args("openai:llama3.2", "openai", "");
        args.openai_key = None;
        args.openai_base_url = Some("http://127.0.0.1:8099/v1".into());
        let config = build_config(&args);
        assert!(config.is_ok(), "expected Ok: {config:?}");
        let config = config.unwrap_or_else(|_| unreachable!());
        assert_eq!(config.providers[0].base_url, "http://127.0.0.1:8099");
        assert!(config.providers[0].api_key.is_empty());
    }

    #[test]
    fn test_build_config_missing_key_error_says_how_to_fix() {
        let mut args = minimal_run_args("gpt-4o-mini", "openai", "");
        args.openai_key = None;
        let err = build_config(&args)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("OPENAI_API_KEY"), "got: {err}");
        assert!(err.contains("--openai-key"), "got: {err}");
    }

    #[test]
    fn test_resolve_model_empty_string_returns_error() {
        let result = resolve_model("");
        assert!(result.is_err());
    }

    //  build_config

    fn minimal_run_args(model: &str, key_name: &str, key_val: &str) -> RunArgs {
        let (openai_key, anthropic_key) = if key_name == "openai" {
            (Some(key_val.to_owned()), None)
        } else {
            (None, Some(key_val.to_owned()))
        };
        RunArgs {
            openai_key,
            anthropic_key,
            openai_base_url: None,
            anthropic_base_url: None,
            models: vec![model.to_owned()],
            prompts: vec!["Say hi".to_owned()],
            prompt_file: None,
            runs: 2,
            concurrency: 3,
            output: OutputFormat::Table,
            output_file: None,
            max_tokens: 128,
            openrouter_key: None,
            gemini_key: None,
            ollama_url: None,
            stream: false,
            retries: 0,
            warmup: 0,
            summary_file: None,
        }
    }

    #[test]
    fn test_build_config_openai_model_produces_openai_provider() {
        let args = minimal_run_args("gpt-4o-mini", "openai", "sk-test");
        let config = build_config(&args);
        assert!(config.is_ok(), "expected Ok: {config:?}");
        let config = config.unwrap_or_else(|_| unreachable!());
        assert_eq!(config.providers.len(), 1);
        assert_eq!(config.providers[0].name, "openai");
        assert_eq!(config.providers[0].model, "gpt-4o-mini");
    }

    #[test]
    fn test_build_config_anthropic_model_produces_anthropic_provider() {
        let args = minimal_run_args("claude-haiku-4-5", "anthropic", "sk-ant-test");
        let config = build_config(&args);
        assert!(config.is_ok());
        let config = config.unwrap_or_else(|_| unreachable!());
        assert_eq!(config.providers[0].name, "anthropic");
    }

    #[test]
    fn test_build_config_runs_per_prompt_propagated() {
        let args = minimal_run_args("gpt-4o-mini", "openai", "sk-test");
        let config = build_config(&args).unwrap_or_else(|_| unreachable!());
        assert_eq!(config.runs_per_prompt, 2);
    }

    #[test]
    fn test_build_config_concurrency_propagated() {
        let args = minimal_run_args("gpt-4o-mini", "openai", "sk-test");
        let config = build_config(&args).unwrap_or_else(|_| unreachable!());
        assert_eq!(config.concurrency, 3);
    }

    #[test]
    fn test_build_config_no_prompts_returns_error() {
        let args = RunArgs {
            openai_key: Some("sk-test".into()),
            anthropic_key: None,
            openai_base_url: None,
            anthropic_base_url: None,
            models: vec!["gpt-4o-mini".into()],
            prompts: vec![],
            prompt_file: None,
            runs: 1,
            concurrency: 1,
            output: OutputFormat::Table,
            output_file: None,
            max_tokens: 128,
            openrouter_key: None,
            gemini_key: None,
            ollama_url: None,
            stream: false,
            retries: 0,
            warmup: 0,
            summary_file: None,
        };
        let result = build_config(&args);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BenchError::InvalidConfig { .. }
        ));
    }

    #[test]
    fn test_build_config_missing_openai_key_returns_error() {
        let args = RunArgs {
            openai_key: None,
            anthropic_key: None,
            openai_base_url: None,
            anthropic_base_url: None,
            models: vec!["gpt-4o-mini".into()],
            prompts: vec!["hello".into()],
            prompt_file: None,
            runs: 1,
            concurrency: 1,
            output: OutputFormat::Table,
            output_file: None,
            max_tokens: 128,
            openrouter_key: None,
            gemini_key: None,
            ollama_url: None,
            stream: false,
            retries: 0,
            warmup: 0,
            summary_file: None,
        };
        let result = build_config(&args);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BenchError::InvalidConfig { .. }
        ));
    }

    #[test]
    fn test_build_config_missing_anthropic_key_returns_error() {
        let args = RunArgs {
            openai_key: None,
            anthropic_key: None,
            openai_base_url: None,
            anthropic_base_url: None,
            models: vec!["claude-haiku-4-5".into()],
            prompts: vec!["hello".into()],
            prompt_file: None,
            runs: 1,
            concurrency: 1,
            output: OutputFormat::Table,
            output_file: None,
            max_tokens: 128,
            openrouter_key: None,
            gemini_key: None,
            ollama_url: None,
            stream: false,
            retries: 0,
            warmup: 0,
            summary_file: None,
        };
        let result = build_config(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_config_explicit_openai_prefix() {
        let args = RunArgs {
            openai_key: Some("sk-test".into()),
            anthropic_key: None,
            openai_base_url: None,
            anthropic_base_url: None,
            models: vec!["openai:gpt-4o".into()],
            prompts: vec!["test".into()],
            prompt_file: None,
            runs: 1,
            concurrency: 1,
            output: OutputFormat::Table,
            output_file: None,
            max_tokens: 256,
            openrouter_key: None,
            gemini_key: None,
            ollama_url: None,
            stream: false,
            retries: 0,
            warmup: 0,
            summary_file: None,
        };
        let config = build_config(&args).unwrap_or_else(|_| unreachable!());
        assert_eq!(config.providers[0].model, "gpt-4o");
    }

    #[test]
    fn test_build_config_multiple_models_produces_multiple_providers() {
        let args = RunArgs {
            openai_key: Some("sk-openai".into()),
            anthropic_key: Some("sk-ant".into()),
            openai_base_url: None,
            anthropic_base_url: None,
            models: vec!["gpt-4o-mini".into(), "claude-haiku-4-5".into()],
            prompts: vec!["hello".into()],
            prompt_file: None,
            runs: 1,
            concurrency: 2,
            output: OutputFormat::Table,
            output_file: None,
            max_tokens: 128,
            openrouter_key: None,
            gemini_key: None,
            ollama_url: None,
            stream: false,
            retries: 0,
            warmup: 0,
            summary_file: None,
        };
        let config = build_config(&args).unwrap_or_else(|_| unreachable!());
        assert_eq!(config.providers.len(), 2);
    }

    #[test]
    fn test_build_config_max_tokens_propagated() {
        let args = minimal_run_args("gpt-4o-mini", "openai", "sk-test");
        let config = build_config(&args).unwrap_or_else(|_| unreachable!());
        assert_eq!(config.providers[0].max_tokens, 128);
    }

    #[test]
    fn test_build_config_prompt_file_not_found_returns_io_error() {
        let args = RunArgs {
            openai_key: Some("sk-test".into()),
            anthropic_key: None,
            openai_base_url: None,
            anthropic_base_url: None,
            models: vec!["gpt-4o-mini".into()],
            prompts: vec![],
            prompt_file: Some(PathBuf::from("/nonexistent/path/prompts.txt")),
            runs: 1,
            concurrency: 1,
            output: OutputFormat::Table,
            output_file: None,
            max_tokens: 128,
            openrouter_key: None,
            gemini_key: None,
            ollama_url: None,
            stream: false,
            retries: 0,
            warmup: 0,
            summary_file: None,
        };
        let result = build_config(&args);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BenchError::Io(_)));
    }

    #[test]
    fn test_build_config_prompt_file_loaded() {
        use std::io::Write;

        // Build a temp file with two prompts
        let path = std::env::temp_dir().join(format!(
            "llm_bench_test_{}.txt",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        {
            let mut f = std::fs::File::create(&path)
                .unwrap_or_else(|_| unreachable!("failed to create temp file"));
            writeln!(f, "First prompt").unwrap_or(());
            writeln!(f, "Second prompt").unwrap_or(());
            f.flush().unwrap_or(());
        }

        let args = RunArgs {
            openai_key: Some("sk-test".into()),
            anthropic_key: None,
            openai_base_url: None,
            anthropic_base_url: None,
            models: vec!["gpt-4o-mini".into()],
            prompts: vec![],
            prompt_file: Some(path.clone()),
            runs: 1,
            concurrency: 1,
            output: OutputFormat::Table,
            output_file: None,
            max_tokens: 128,
            openrouter_key: None,
            gemini_key: None,
            ollama_url: None,
            stream: false,
            retries: 0,
            warmup: 0,
            summary_file: None,
        };
        let config = build_config(&args).unwrap_or_else(|_| unreachable!());
        assert_eq!(config.prompts.len(), 2);
        assert_eq!(config.prompts[0], "First prompt");

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_resolve_model_new_prefixes() {
        assert_eq!(resolve_model("ollama:qwen2.5:0.5b").ok(), Some(("ollama", "qwen2.5:0.5b")));
        assert_eq!(
            resolve_model("openrouter:meta-llama/llama-3.3-70b-instruct").ok(),
            Some(("openrouter", "meta-llama/llama-3.3-70b-instruct"))
        );
        assert_eq!(resolve_model("gemini-2.5-flash").ok(), Some(("gemini", "gemini-2.5-flash")));
        assert_eq!(resolve_model("o4-mini").ok(), Some(("openai", "o4-mini")));
        // A colon inside a bare model id is not a provider prefix.
        assert!(resolve_model("qwen2.5:0.5b").is_err());
        assert!(resolve_model("ollama:").is_err());
    }

    #[test]
    fn test_build_config_ollama_needs_no_key_and_reads_ollama_host() {
        let mut args = minimal_run_args("ollama:llama3.2", "openai", "");
        args.openai_key = None;
        let c = build_config(&args).unwrap();
        assert_eq!(c.providers[0].name, "ollama");
        assert_eq!(c.providers[0].base_url, "");
        args.ollama_url = Some("0.0.0.0:11434".into());
        let c = build_config(&args).unwrap();
        assert_eq!(c.providers[0].base_url, "http://localhost:11434");
    }

    #[test]
    fn test_build_config_hosted_providers_need_their_keys() {
        let mut args = minimal_run_args("openrouter:openai/gpt-4o-mini", "openai", "sk");
        let err = build_config(&args).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(err.contains("OPENROUTER_API_KEY"), "{err}");
        args.openrouter_key = Some("or-key".into());
        let c = build_config(&args).unwrap();
        assert_eq!(c.providers[0].api_key, "or-key");

        let mut args = minimal_run_args("gemini:gemini-2.5-flash", "openai", "sk");
        assert!(build_config(&args).is_err());
        args.gemini_key = Some("g".into());
        assert!(build_config(&args).is_ok());
    }

    #[test]
    fn test_stream_and_retries_propagate() {
        let mut args = minimal_run_args("gpt-4o-mini", "openai", "sk-test");
        args.stream = true;
        args.retries = 3;
        let c = build_config(&args).unwrap();
        assert!(c.stream);
        assert_eq!(c.retries, 3);
    }

    //  OutputFormat

    #[test]
    fn test_output_format_is_debug_printable() {
        let t = format!("{:?}", OutputFormat::Table);
        let j = format!("{:?}", OutputFormat::Json);
        assert!(t.contains("Table"));
        assert!(j.contains("Json"));
    }
}
