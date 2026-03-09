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

//  Top-level CLI 

/// Universal LLM provider benchmark CLI.
///
/// Compare OpenAI and Anthropic models on latency, cost, quality, and
/// token throughput from the command line.
#[derive(Debug, Parser)]
#[command(
    name = "llm-bench",
    version,
    about = "Benchmark OpenAI and Anthropic models on latency, cost, and throughput",
    long_about = None
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

/// Available sub-commands.
#[derive(Debug, Subcommand)]
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

    /// Comma-separated list of model identifiers to benchmark.
    ///
    /// Prefix with provider: `openai:gpt-4o-mini` or `anthropic:claude-3-5-haiku-20241022`.
    /// Bare names (no prefix) are matched against known models automatically.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "gpt-4o-mini,claude-3-5-haiku-20241022"
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
}

/// Output format selector.
#[derive(Debug, Clone, ValueEnum)]
pub enum OutputFormat {
    /// Pretty ASCII table (default).
    Table,
    /// JSON array of all BenchResult records.
    Json,
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

        let api_key = match provider_name {
            "openai" => args.openai_key.as_deref().unwrap_or("").to_owned(),
            "anthropic" => args.anthropic_key.as_deref().unwrap_or("").to_owned(),
            other => {
                return Err(BenchError::InvalidConfig {
                    reason: format!("unknown provider '{other}'"),
                })
            }
        };

        if api_key.is_empty() {
            return Err(BenchError::InvalidConfig {
                reason: format!(
                    "No API key for provider '{provider_name}'. \
                     Pass --{provider_name}-key or set the environment variable."
                ),
            });
        }

        providers.push(ProviderConfig {
            name: provider_name.to_owned(),
            model: model.to_owned(),
            api_key,
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
    })
}

/// Resolve a model string to `(provider_name, model_id)`.
///
/// Accepts:
/// - `"openai:gpt-4o-mini"` → `("openai", "gpt-4o-mini")`
/// - `"anthropic:claude-3-5-haiku-20241022"` → `("anthropic", "claude-3-5-haiku-20241022")`
/// - Bare names are matched against known model prefixes.
///
/// # Panics
/// This function never panics.
fn resolve_model(s: &str) -> Result<(&'static str, &str), BenchError> {
    if let Some(rest) = s.strip_prefix("openai:") {
        return Ok(("openai", rest));
    }
    if let Some(rest) = s.strip_prefix("anthropic:") {
        return Ok(("anthropic", rest));
    }

    // Auto-detect from known model name prefixes
    if s.starts_with("gpt-") || s.starts_with("o1") || s.starts_with("o3") {
        return Ok(("openai", s));
    }
    if s.starts_with("claude-") {
        return Ok(("anthropic", s));
    }

    Err(BenchError::InvalidConfig {
        reason: format!(
            "Cannot determine provider for model '{s}'. \
             Prefix with 'openai:' or 'anthropic:', e.g. openai:{s}"
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
        let (provider, model) =
            resolve_model("anthropic:claude-3-5-haiku-20241022").unwrap_or(("", ""));
        assert_eq!(provider, "anthropic");
        assert_eq!(model, "claude-3-5-haiku-20241022");
    }

    #[test]
    fn test_resolve_model_bare_gpt_autodetects_openai() {
        let (provider, model) = resolve_model("gpt-4o-mini").unwrap_or(("", ""));
        assert_eq!(provider, "openai");
        assert_eq!(model, "gpt-4o-mini");
    }

    #[test]
    fn test_resolve_model_bare_claude_autodetects_anthropic() {
        let (provider, model) =
            resolve_model("claude-3-5-haiku-20241022").unwrap_or(("", ""));
        assert_eq!(provider, "anthropic");
        assert_eq!(model, "claude-3-5-haiku-20241022");
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
        assert!(matches!(result.unwrap_err(), BenchError::InvalidConfig { .. }));
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
            models: vec![model.to_owned()],
            prompts: vec!["Say hi".to_owned()],
            prompt_file: None,
            runs: 2,
            concurrency: 3,
            output: OutputFormat::Table,
            output_file: None,
            max_tokens: 128,
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
        let args = minimal_run_args(
            "claude-3-5-haiku-20241022",
            "anthropic",
            "sk-ant-test",
        );
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
            models: vec!["gpt-4o-mini".into()],
            prompts: vec![],
            prompt_file: None,
            runs: 1,
            concurrency: 1,
            output: OutputFormat::Table,
            output_file: None,
            max_tokens: 128,
        };
        let result = build_config(&args);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BenchError::InvalidConfig { .. }));
    }

    #[test]
    fn test_build_config_missing_openai_key_returns_error() {
        let args = RunArgs {
            openai_key: None,
            anthropic_key: None,
            models: vec!["gpt-4o-mini".into()],
            prompts: vec!["hello".into()],
            prompt_file: None,
            runs: 1,
            concurrency: 1,
            output: OutputFormat::Table,
            output_file: None,
            max_tokens: 128,
        };
        let result = build_config(&args);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BenchError::InvalidConfig { .. }));
    }

    #[test]
    fn test_build_config_missing_anthropic_key_returns_error() {
        let args = RunArgs {
            openai_key: None,
            anthropic_key: None,
            models: vec!["claude-3-5-haiku-20241022".into()],
            prompts: vec!["hello".into()],
            prompt_file: None,
            runs: 1,
            concurrency: 1,
            output: OutputFormat::Table,
            output_file: None,
            max_tokens: 128,
        };
        let result = build_config(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_config_explicit_openai_prefix() {
        let args = RunArgs {
            openai_key: Some("sk-test".into()),
            anthropic_key: None,
            models: vec!["openai:gpt-4o".into()],
            prompts: vec!["test".into()],
            prompt_file: None,
            runs: 1,
            concurrency: 1,
            output: OutputFormat::Table,
            output_file: None,
            max_tokens: 256,
        };
        let config = build_config(&args).unwrap_or_else(|_| unreachable!());
        assert_eq!(config.providers[0].model, "gpt-4o");
    }

    #[test]
    fn test_build_config_multiple_models_produces_multiple_providers() {
        let args = RunArgs {
            openai_key: Some("sk-openai".into()),
            anthropic_key: Some("sk-ant".into()),
            models: vec![
                "gpt-4o-mini".into(),
                "claude-3-5-haiku-20241022".into(),
            ],
            prompts: vec!["hello".into()],
            prompt_file: None,
            runs: 1,
            concurrency: 2,
            output: OutputFormat::Table,
            output_file: None,
            max_tokens: 128,
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
            models: vec!["gpt-4o-mini".into()],
            prompts: vec![],
            prompt_file: Some(PathBuf::from("/nonexistent/path/prompts.txt")),
            runs: 1,
            concurrency: 1,
            output: OutputFormat::Table,
            output_file: None,
            max_tokens: 128,
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
            models: vec!["gpt-4o-mini".into()],
            prompts: vec![],
            prompt_file: Some(path.clone()),
            runs: 1,
            concurrency: 1,
            output: OutputFormat::Table,
            output_file: None,
            max_tokens: 128,
        };
        let config = build_config(&args).unwrap_or_else(|_| unreachable!());
        assert_eq!(config.prompts.len(), 2);
        assert_eq!(config.prompts[0], "First prompt");

        // Cleanup
        let _ = std::fs::remove_file(&path);
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
