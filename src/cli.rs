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

use crate::config::{FileConfig, PriceTable};
use crate::error::BenchError;
use crate::runner::BUILTIN_PROVIDERS;
use crate::types::{BenchConfig, ProviderConfig};

/// Examples shown at the bottom of `llm-bench --help`.
const HELP_EXAMPLES: &str = "Examples:
  llm-bench models
      List the models llm-bench knows prices for.

  export OPENAI_API_KEY=sk-...   (PowerShell: $env:OPENAI_API_KEY=\"sk-...\")
  llm-bench run --models gpt-4o-mini --prompts \"Say hi\"
      One model, one prompt, 3 runs, results table.

  llm-bench run --models gpt-4o-mini,claude-haiku-4-5 --prompt-file prompts.txt --repeat 5 --stream
      Compare two providers on your own prompts: TTFT, p50/p95 latency, tok/s, errors, cost.

  llm-bench run --models groq:llama-3.3-70b-versatile,together:meta-llama/Llama-3.3-70B-Instruct-Turbo,fireworks:accounts/fireworks/models/llama-v3p3-70b-instruct --prompt-file prompts.txt --stream --concurrency 8
      The same open model on three hosts (GROQ_API_KEY, TOGETHER_API_KEY, FIREWORKS_API_KEY).

  llm-bench run --models deepseek:deepseek-v4-flash,openrouter:openai/gpt-4o-mini --prompts \"Say hi\" --output markdown
      DeepSeek and OpenRouter, as a Markdown table for a PR or an issue.

  llm-bench run --config llm-bench.toml --models vllm:my-model --prompts \"Say hi\" --stream
      Any OpenAI-compatible server defined in a config file:
        [providers.vllm]
        base_url = \"http://localhost:8000/v1\"
        [prices.my-model]          # optional, USD per 1M tokens
        input = 0.10
        output = 0.40

  llm-bench run --models ollama:llama3.2 --prompts \"Say hi\" --stream
      A local Ollama model, no key needed.";

//  Top-level CLI

/// Universal LLM provider benchmark CLI.
///
/// Compare providers and models on your own prompts: time to first token,
/// p50/p95 latency, tokens per second, error rate and cost.
#[derive(Debug, Parser)]
#[command(
    name = "llm-bench",
    version,
    about = "Compare LLM providers on your own prompts (OpenAI, Anthropic, OpenRouter, Groq, Together, Fireworks, DeepSeek, Gemini, Ollama, any OpenAI-compatible URL): TTFT, p50/p95 latency, tokens/s, error rate and cost",
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
#[derive(Debug, Default, Parser)]
pub struct RunArgs {
    /// TOML file with extra OpenAI-compatible providers (`[providers.NAME]
    /// base_url = ...`, `api_key_env = ...`) and prices (`[prices."MODEL"]
    /// input = ..., output = ...`, USD per 1M tokens).
    #[arg(long, env = "LLM_BENCH_CONFIG")]
    pub config: Option<PathBuf>,

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

    /// Groq API key, for `groq:` models.
    #[arg(long, env = "GROQ_API_KEY", hide_env_values = true)]
    pub groq_key: Option<String>,

    /// Together AI API key, for `together:` models.
    #[arg(long, env = "TOGETHER_API_KEY", hide_env_values = true)]
    pub together_key: Option<String>,

    /// Fireworks AI API key, for `fireworks:` models.
    #[arg(long, env = "FIREWORKS_API_KEY", hide_env_values = true)]
    pub fireworks_key: Option<String>,

    /// DeepSeek API key, for `deepseek:` models.
    #[arg(long, env = "DEEPSEEK_API_KEY", hide_env_values = true)]
    pub deepseek_key: Option<String>,

    /// Ollama server for `ollama:` models (default http://localhost:11434).
    /// `OLLAMA_HOST` is read too, with or without `http://`.
    #[arg(long, env = "OLLAMA_HOST")]
    pub ollama_url: Option<String>,

    /// Comma-separated list of model identifiers to benchmark.
    ///
    /// Prefix with the provider: `openai:`, `anthropic:`, `openrouter:`,
    /// `groq:`, `together:`, `fireworks:`, `deepseek:`, `gemini:`,
    /// `ollama:` or a provider from `--config`. Bare `gpt-*`, `o1*`, `o3*`,
    /// `o4*`, `claude-*`, `gemini-*` and `deepseek-*` names are matched
    /// automatically.
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

    /// Number of times each prompt is sent to each model (`--repeat` is
    /// the same flag).
    #[arg(long, visible_alias = "repeat", default_value_t = 3)]
    pub runs: u32,

    /// Maximum number of requests in flight at once, across all models.
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,

    /// Output format.
    #[arg(long, default_value = "table")]
    pub output: OutputFormat,

    /// Save full JSON results to this file.
    #[arg(long)]
    pub output_file: Option<PathBuf>,

    /// Also write the comparison table as Markdown to this file.
    #[arg(long)]
    pub markdown_file: Option<PathBuf>,

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
#[derive(Debug, Clone, Default, ValueEnum)]
pub enum OutputFormat {
    /// Comparison table for the terminal (default).
    #[default]
    Table,
    /// The comparison table as GitHub-flavoured Markdown.
    #[value(alias = "md")]
    Markdown,
    /// JSON array of all BenchResult records.
    Json,
    /// CSV, one row per successful request.
    Csv,
}

//  Config builder

/// Build a `BenchConfig` from parsed `RunArgs`; see [`build_plan`], which
/// also returns the `[prices]` from the `--config` file.
///
/// # Errors
/// As [`build_plan`].
///
/// # Panics
/// This function never panics.
pub fn build_config(args: &RunArgs) -> Result<BenchConfig, BenchError> {
    build_plan(args).map(|(config, _)| config)
}

/// Build the benchmark from parsed `RunArgs`: prompts from `--prompts` and
/// `--prompt-file`, one `ProviderConfig` per `--models` entry (keys from
/// the flags, their environment variables or the `--config` file), and the
/// `[prices]` table from the `--config` file.
///
/// # Errors
/// Returns `BenchError::InvalidConfig` if:
/// - No prompts are provided via either flag.
/// - A model string cannot be resolved to a known provider.
/// - An API key is missing for a hosted provider.
/// - The `--config` file is invalid.
///
/// Returns `BenchError::Io` if the prompt or config file cannot be read.
///
/// # Panics
/// This function never panics.
pub fn build_plan(args: &RunArgs) -> Result<(BenchConfig, PriceTable), BenchError> {
    let file = match &args.config {
        Some(path) => FileConfig::load(path)?,
        None => FileConfig::default(),
    };

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

        let (provider_name, model) = resolve_model_with(model_str, &file)?;
        let builtin = BUILTIN_PROVIDERS
            .iter()
            .find(|(n, _)| *n == provider_name)
            .map(|(_, env)| *env);

        let key = |k: &Option<String>| k.as_deref().unwrap_or("").trim().to_owned();
        let flag_key = match provider_name.as_str() {
            "openai" => key(&args.openai_key),
            "anthropic" => key(&args.anthropic_key),
            "openrouter" => key(&args.openrouter_key),
            "gemini" => key(&args.gemini_key),
            "groq" => key(&args.groq_key),
            "together" => key(&args.together_key),
            "fireworks" => key(&args.fireworks_key),
            "deepseek" => key(&args.deepseek_key),
            _ => String::new(),
        };
        let api_key = if flag_key.is_empty() {
            file.api_key(&provider_name)
        } else {
            flag_key
        };
        // The config file asked for a key from a variable that is not set:
        // say so now instead of sending every request without a key.
        if api_key.is_empty() {
            if let Some(var) = file.missing_key_env(&provider_name) {
                return Err(BenchError::InvalidConfig {
                    reason: format!(
                        "provider '{provider_name}' in the --config file reads its key from \
                         api_key_env = \"{var}\", but {var} is not set (needed for model '{model}').\n  \
                         Fix: set {var}, or remove the api_key_env line if the server needs no key."
                    ),
                });
            }
        }

        // A flag wins over the config file. OpenAI and Anthropic URLs drop a
        // trailing /v1 (the request path adds it); other OpenAI-compatible
        // URLs are used as given, so a path such as Groq's /openai/v1 stays.
        let flag_url = match provider_name.as_str() {
            "openai" => args.openai_base_url.as_deref(),
            "anthropic" => args.anthropic_base_url.as_deref(),
            "ollama" => args.ollama_url.as_deref(),
            _ => None,
        }
        .filter(|u| !u.trim().is_empty());
        let raw_url = flag_url.or_else(|| file.base_url(&provider_name));
        let base_url = match provider_name.as_str() {
            "openai" | "anthropic" => normalize_base_url(raw_url),
            "ollama" => normalize_ollama_url(raw_url),
            _ => raw_url.unwrap_or("").trim().trim_end_matches('/').to_owned(),
        };

        match builtin {
            // Hosted endpoints need a key; a custom base URL (local server,
            // proxy) and Ollama may not.
            Some(env_var) => {
                if api_key.is_empty() && base_url.is_empty() && provider_name != "ollama" {
                    return Err(BenchError::InvalidConfig {
                        reason: format!(
                            "no API key for {provider_name} (needed for model '{model}').\n  \
                             Fix: set {env_var} (bash: export {env_var}=...  PowerShell: $env:{env_var}=\"...\")\n  \
                             or pass --{provider_name}-key <KEY>, or leave '{model}' out of --models."
                        ),
                    });
                }
            }
            None => {
                if base_url.is_empty() {
                    return Err(BenchError::InvalidConfig {
                        reason: format!(
                            "provider '{provider_name}' has no base_url in the --config file"
                        ),
                    });
                }
            }
        }

        providers.push(ProviderConfig {
            name: provider_name.clone(),
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

    let config = BenchConfig {
        prompts,
        runs_per_prompt: args.runs,
        concurrency: args.concurrency,
        providers,
        stream: args.stream,
        retries: args.retries,
        warmup: args.warmup,
    };
    Ok((config, file.price_table()))
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

/// Built-in provider prefixes accepted in `--models`.
const PREFIXES: [&str; 9] = [
    "openai",
    "anthropic",
    "ollama",
    "openrouter",
    "gemini",
    "groq",
    "together",
    "fireworks",
    "deepseek",
];

/// [`resolve_model`], plus the provider names defined in a config file.
///
/// Order: an explicit built-in prefix (`groq:...`), then a provider named
/// in the config file (`my-server:...`), then the bare-name rules
/// (`gpt-*`, `claude-*`, ...). Config names come before the bare-name rules
/// so that `[providers.deepseek-local]` or `[providers.gpt-proxy]` is not
/// mistaken for a `deepseek-*` or `gpt-*` model on the vendor's own API.
fn resolve_model_with<'a>(s: &'a str, file: &FileConfig) -> Result<(String, &'a str), BenchError> {
    if let Some((p, m)) = resolve_builtin_prefix(s) {
        return Ok((p.to_owned(), m));
    }
    if let Some((p, rest)) = s.split_once(':') {
        if file.providers.contains_key(p) && !rest.is_empty() {
            return Ok((p.to_owned(), rest));
        }
    }
    resolve_model(s).map(|(p, m)| (p.to_owned(), m))
}

/// `"groq:llama"` -> `("groq", "llama")` for a built-in prefix followed by a
/// non-empty model id.
fn resolve_builtin_prefix(s: &str) -> Option<(&'static str, &str)> {
    PREFIXES.into_iter().find_map(|p| {
        s.strip_prefix(p)
            .and_then(|r| r.strip_prefix(':'))
            .filter(|rest| !rest.is_empty())
            .map(|rest| (p, rest))
    })
}

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
    if let Some(hit) = resolve_builtin_prefix(s) {
        return Ok(hit);
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
    if s.starts_with("deepseek-") {
        return Ok(("deepseek", s));
    }

    Err(BenchError::InvalidConfig {
        reason: format!(
            "Cannot determine provider for model '{s}'. \
             Prefix it with a provider ({}) or one from --config, e.g. ollama:{s}",
            PREFIXES.join(", ")
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
            ..RunArgs::default()
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
            ..RunArgs::default()
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
            ..RunArgs::default()
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
            ..RunArgs::default()
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
            ..RunArgs::default()
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
            ..RunArgs::default()
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
            ..RunArgs::default()
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
            ..RunArgs::default()
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

    //  0.4: new providers, --config, --repeat

    fn args(models: &[&str]) -> RunArgs {
        RunArgs {
            models: models.iter().map(|m| (*m).to_owned()).collect(),
            prompts: vec!["hi".into()],
            runs: 1,
            concurrency: 1,
            max_tokens: 16,
            ..RunArgs::default()
        }
    }

    fn write_temp(name: &str, text: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "llm_bench_{name}_{}_{}.toml",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&path, text).unwrap();
        path
    }

    #[test]
    fn test_new_hosted_providers_resolve_and_need_keys() {
        for (prefix, env) in [
            ("groq", "GROQ_API_KEY"),
            ("together", "TOGETHER_API_KEY"),
            ("fireworks", "FIREWORKS_API_KEY"),
            ("deepseek", "DEEPSEEK_API_KEY"),
        ] {
            let mut a = args(&[&format!("{prefix}:some/model:tag")]);
            let err = build_config(&a).err().map(|e| e.to_string()).unwrap_or_default();
            assert!(err.contains(env), "{prefix}: {err}");
            assert!(err.contains(&format!("--{prefix}-key")), "{prefix}: {err}");
            let k = Some(format!("{prefix}-secret"));
            match prefix {
                "groq" => a.groq_key = k,
                "together" => a.together_key = k,
                "fireworks" => a.fireworks_key = k,
                _ => a.deepseek_key = k,
            }
            let c = build_config(&a).unwrap();
            assert_eq!(c.providers[0].name, prefix);
            assert_eq!(c.providers[0].model, "some/model:tag");
            assert_eq!(c.providers[0].api_key, format!("{prefix}-secret"));
            assert_eq!(c.providers[0].base_url, "");
        }
        assert_eq!(resolve_model("deepseek-v4-flash").ok(), Some(("deepseek", "deepseek-v4-flash")));
    }

    #[test]
    fn test_config_file_custom_provider_overrides_and_prices() {
        let path = write_temp(
            "cfg",
            r#"
[providers.vllm]
base_url = "http://127.0.0.1:8000/v1/"

[providers.groq]
base_url = "https://gateway.example/groq/openai/v1"
api_key = "from-file"

[prices."vllm:qwen2.5:0.5b"]
input = 0.1
output = 0.2
"#,
        );
        let mut a = args(&["vllm:qwen2.5:0.5b", "groq:llama-3.3-70b-versatile"]);
        a.config = Some(path.clone());
        let (c, prices) = build_plan(&a).unwrap();
        // Custom provider: no key needed, URL kept with its /v1.
        assert_eq!(c.providers[0].name, "vllm");
        assert_eq!(c.providers[0].model, "qwen2.5:0.5b");
        assert_eq!(c.providers[0].base_url, "http://127.0.0.1:8000/v1");
        assert!(c.providers[0].api_key.is_empty());
        // Built-in provider pointed elsewhere, key from the file.
        assert_eq!(c.providers[1].base_url, "https://gateway.example/groq/openai/v1");
        assert_eq!(c.providers[1].api_key, "from-file");
        assert_eq!(prices.get("vllm", "qwen2.5:0.5b").map(|p| p.output), Some(0.2));
        // A flag key beats the file.
        a.groq_key = Some("from-flag".into());
        assert_eq!(build_config(&a).unwrap().providers[1].api_key, "from-flag");
        // Unknown prefix without a config entry.
        let mut b = args(&["nope:model"]);
        b.config = Some(path.clone());
        assert!(build_config(&b).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_config_provider_named_like_a_model_family_wins_over_bare_name_rules() {
        // Names that start like deepseek-*, gpt-*, claude-*, gemini-* or o3*
        // models must still route to the configured URL, not the vendor.
        let path = write_temp(
            "lookalike",
            r#"
[providers.deepseek-local]
base_url = "http://127.0.0.1:9/v1"

[providers.gpt-proxy]
base_url = "http://127.0.0.1:9/proxy/v1"

[providers.claude-gw]
base_url = "http://127.0.0.1:9/claude"

[providers.o3-mirror]
base_url = "http://127.0.0.1:9/o3"
"#,
        );
        let mut a = args(&["deepseek-local:foo", "gpt-proxy:gpt-4o", "claude-gw:bar", "o3-mirror:o3"]);
        a.config = Some(path.clone());
        // No DEEPSEEK/OPENAI/ANTHROPIC key is set on `a`; this used to fail
        // with "no API key for deepseek".
        let c = build_config(&a).unwrap();
        let got: Vec<_> = c
            .providers
            .iter()
            .map(|p| (p.name.as_str(), p.model.as_str(), p.base_url.as_str()))
            .collect();
        assert_eq!(
            got,
            [
                ("deepseek-local", "foo", "http://127.0.0.1:9/v1"),
                ("gpt-proxy", "gpt-4o", "http://127.0.0.1:9/proxy/v1"),
                ("claude-gw", "bar", "http://127.0.0.1:9/claude"),
                ("o3-mirror", "o3", "http://127.0.0.1:9/o3"),
            ]
        );
        // Bare names and explicit built-in prefixes are unchanged.
        let file = FileConfig::parse("[providers.deepseek-local]
base_url = \"http://x\"").unwrap();
        assert_eq!(
            resolve_model_with("deepseek-chat", &file).ok(),
            Some(("deepseek".to_owned(), "deepseek-chat"))
        );
        assert_eq!(
            resolve_model_with("deepseek:deepseek-local:foo", &file).ok(),
            Some(("deepseek".to_owned(), "deepseek-local:foo"))
        );
        assert_eq!(
            resolve_model_with("deepseek-local:foo", &file).ok(),
            Some(("deepseek-local".to_owned(), "foo"))
        );
        // Without the config entry the old bare-name rule still applies.
        assert_eq!(
            resolve_model_with("deepseek-local:foo", &FileConfig::default()).ok(),
            Some(("deepseek".to_owned(), "deepseek-local:foo"))
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_config_api_key_env_unset_is_a_setup_error() {
        let path = write_temp(
            "keyenv",
            r#"
[providers.myhost]
base_url = "http://127.0.0.1:9/v1"
api_key_env = "LLM_BENCH_TEST_UNSET_VARIABLE_2"

[providers.groq]
base_url = "http://127.0.0.1:9/groq"
api_key_env = "LLM_BENCH_TEST_UNSET_VARIABLE_3"
"#,
        );
        let mut a = args(&["myhost:m"]);
        a.config = Some(path.clone());
        let err = build_config(&a).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(err.contains("LLM_BENCH_TEST_UNSET_VARIABLE_2"), "{err}");
        assert!(err.contains("myhost"), "{err}");
        // A built-in provider redirected by the file: the flag key (or its
        // usual variable) still satisfies it; without one, the same error.
        let mut g = args(&["groq:llama"]);
        g.config = Some(path.clone());
        let err = build_config(&g).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(err.contains("LLM_BENCH_TEST_UNSET_VARIABLE_3"), "{err}");
        g.groq_key = Some("gk".into());
        assert_eq!(build_config(&g).unwrap().providers[0].api_key, "gk");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_config_file_errors_name_the_file() {
        let path = write_temp("bad", "[providers.x]\nbase_urll = 'typo'\n");
        let mut a = args(&["gpt-4o-mini"]);
        a.openai_key = Some("k".into());
        a.config = Some(path.clone());
        let err = build_config(&a).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(err.contains("base_urll"), "{err}");
        assert!(err.contains(&path.display().to_string()), "{err}");
        let _ = std::fs::remove_file(&path);
        a.config = Some(PathBuf::from("/nonexistent/llm-bench.toml"));
        assert!(matches!(build_config(&a), Err(BenchError::Io(_))));
    }

    #[test]
    fn test_repeat_alias_markdown_output_and_flags_parse() {
        let cli = Cli::try_parse_from([
            "llm-bench", "run", "--models", "ollama:x", "--prompts", "hi", "--repeat", "7",
            "--concurrency", "5", "--output", "md", "--markdown-file", "out.md",
        ])
        .unwrap();
        let Command::Run(a) = cli.command else { unreachable!() };
        assert_eq!((a.runs, a.concurrency), (7, 5));
        assert!(matches!(a.output, OutputFormat::Markdown));
        assert_eq!(a.markdown_file, Some(PathBuf::from("out.md")));
        let c = build_config(&a).unwrap();
        assert_eq!((c.runs_per_prompt, c.concurrency), (7, 5));
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
