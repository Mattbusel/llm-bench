//! # Module: Config file
//!
//! `--config llm-bench.toml` adds OpenAI-compatible providers under names of
//! your choosing, points built-in providers at another URL, and sets prices
//! for models that have no published list price (Groq, Together, Fireworks,
//! a self-hosted server).
//!
//! ```toml
//! # A self-hosted vLLM, used as `--models vllm:meta-llama/Llama-3.1-8B-Instruct`.
//! [providers.vllm]
//! base_url = "http://localhost:8000/v1"
//! api_key_env = "VLLM_API_KEY"   # optional; local servers often need no key
//!
//! # Send groq: models through a gateway instead of api.groq.com.
//! [providers.groq]
//! base_url = "https://gateway.example.com/groq/openai/v1"
//!
//! # USD per 1 million tokens. A key is either "provider:model" or "model".
//! [prices."groq:llama-3.3-70b-versatile"]
//! input = 0.59
//! output = 0.79
//! ```
//!
//! Every provider defined here speaks the OpenAI Chat Completions format,
//! except an entry named `anthropic`, which only overrides the Anthropic URL.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::BenchError;

/// The parsed contents of a `--config` TOML file.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct FileConfig {
    /// Providers by name: new OpenAI-compatible ones, or overrides for the
    /// built-in ones (`openai`, `groq`, ...).
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderEntry>,
    /// Prices by `"provider:model"` or `"model"`.
    #[serde(default)]
    pub prices: BTreeMap<String, ModelPrice>,
}

/// One `[providers.<name>]` table.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
#[non_exhaustive]
pub struct ProviderEntry {
    /// Base URL of the API, with or without the trailing version segment
    /// (`http://host:8000` and `http://host:8000/v1` both work).
    #[serde(default)]
    pub base_url: Option<String>,
    /// Environment variable that holds the API key.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// The API key itself. Prefer `api_key_env` so the key stays out of the
    /// file.
    #[serde(default)]
    pub api_key: Option<String>,
}

/// A model's price in USD per 1 million tokens.
///
/// Prompt tokens served from the provider's prompt cache are billed at
/// `cached_input` when it is set and at `input` otherwise. Anthropic cache
/// writes (`cache_creation_input_tokens`) are billed at `input`; Anthropic
/// itself charges more for them (1.25x for the 5-minute cache), so a
/// `[prices]` entry for an Anthropic model slightly undercounts requests that
/// write to the cache. Without a `[prices]` entry, Anthropic models use the
/// llm-prices list price, which bills cache writes correctly.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPrice {
    /// USD per 1M prompt (input) tokens.
    pub input: f64,
    /// USD per 1M completion (output) tokens.
    pub output: f64,
    /// USD per 1M prompt tokens read from the provider's prompt cache.
    /// Defaults to `input` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input: Option<f64>,
}

impl ModelPrice {
    /// A price with no separate cache price.
    pub fn new(input: f64, output: f64) -> Self {
        Self {
            input,
            output,
            cached_input: None,
        }
    }

    /// Cost in USD of `prompt_tokens` in and `completion_tokens` out, every
    /// prompt token at the `input` price.
    pub fn cost(&self, prompt_tokens: u32, completion_tokens: u32) -> f64 {
        self.cost_cached(prompt_tokens, 0, completion_tokens)
    }

    /// Cost in USD of `uncached_prompt_tokens` at `input`,
    /// `cached_prompt_tokens` at `cached_input` (or `input` when unset) and
    /// `completion_tokens` at `output`.
    pub fn cost_cached(
        &self,
        uncached_prompt_tokens: u32,
        cached_prompt_tokens: u32,
        completion_tokens: u32,
    ) -> f64 {
        (f64::from(uncached_prompt_tokens) * self.input
            + f64::from(cached_prompt_tokens) * self.cached_input.unwrap_or(self.input)
            + f64::from(completion_tokens) * self.output)
            / 1_000_000.0
    }
}

/// Prices from the config file, looked up by provider and model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PriceTable {
    prices: BTreeMap<String, ModelPrice>,
}

impl PriceTable {
    /// A table from `"provider:model"` or `"model"` keys.
    pub fn new(prices: BTreeMap<String, ModelPrice>) -> Self {
        Self { prices }
    }

    /// The price for `model` on `provider`: the `"provider:model"` entry
    /// first, then the bare `"model"` entry.
    pub fn get(&self, provider: &str, model: &str) -> Option<ModelPrice> {
        self.prices
            .get(&format!("{provider}:{model}"))
            .or_else(|| self.prices.get(model))
            .copied()
    }

    /// Whether the table has no prices.
    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }
}

impl FileConfig {
    /// Read and parse a config file.
    ///
    /// # Errors
    /// `BenchError::Io` when the file cannot be read, and
    /// `BenchError::InvalidConfig` when it is not valid TOML, has unknown
    /// keys, a provider name that cannot be used as a `--models` prefix, or
    /// a negative or non-finite price.
    pub fn load(path: &Path) -> Result<Self, BenchError> {
        let text = std::fs::read_to_string(path).map_err(BenchError::Io)?;
        Self::parse(&text).map_err(|e| BenchError::InvalidConfig {
            reason: format!("{}: {e}", path.display()),
        })
    }

    /// Parse config text (see [`FileConfig::load`]).
    ///
    /// # Errors
    /// A message saying what is wrong.
    pub fn parse(text: &str) -> Result<Self, String> {
        let cfg: FileConfig = toml::from_str(text).map_err(|e| {
            let msg = e.to_string();
            if msg.contains("expected one of `input`, `output`") {
                // [prices.gpt-4.1] is a nested table in TOML, not a model id.
                format!("{msg}\nhint: quote model ids that contain dots: [prices.\"gpt-4.1\"]")
            } else {
                msg
            }
        })?;
        for name in cfg.providers.keys() {
            let ok = !name.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_');
            if !ok {
                return Err(format!(
                    "provider name '{name}' must be lowercase letters, digits, '-' or '_' \
                     (it is used as the prefix in --models {name}:<model>)"
                ));
            }
        }
        for (model, p) in &cfg.prices {
            let ok = |v: f64| v.is_finite() && v >= 0.0;
            if !(ok(p.input) && ok(p.output) && p.cached_input.is_none_or(ok)) {
                return Err(format!(
                    "price for '{model}' must be finite and not negative"
                ));
            }
        }
        Ok(cfg)
    }

    /// The `[prices]` section as a [`PriceTable`].
    pub fn price_table(&self) -> PriceTable {
        PriceTable::new(self.prices.clone())
    }

    /// The API key for provider `name` from `api_key`, or from the variable
    /// named by `api_key_env`. Empty when neither is set, and also when
    /// `api_key_env` names a variable that is unset or empty (see
    /// [`FileConfig::missing_key_env`], which the CLI turns into an error).
    pub fn api_key(&self, name: &str) -> String {
        let Some(p) = self.providers.get(name) else {
            return String::new();
        };
        if let Some(k) = p
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|k| !k.is_empty())
        {
            return k.to_owned();
        }
        p.api_key_env
            .as_deref()
            .and_then(|var| std::env::var(var).ok())
            .map(|k| k.trim().to_owned())
            .unwrap_or_default()
    }

    /// The variable named by provider `name`'s `api_key_env` when that
    /// variable is unset or empty and no `api_key` is given in the file.
    /// `None` when the file supplies a key or does not ask for one.
    pub fn missing_key_env(&self, name: &str) -> Option<&str> {
        let p = self.providers.get(name)?;
        if p.api_key.as_deref().is_some_and(|k| !k.trim().is_empty()) {
            return None;
        }
        let var = p.api_key_env.as_deref().map(str::trim).filter(|v| !v.is_empty())?;
        match std::env::var(var) {
            Ok(v) if !v.trim().is_empty() => None,
            _ => Some(var),
        }
    }

    /// The configured base URL for provider `name`, if any.
    pub fn base_url(&self, name: &str) -> Option<&str> {
        self.providers
            .get(name)
            .and_then(|p| p.base_url.as_deref())
            .map(str::trim)
            .filter(|u| !u.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
[providers.vllm]
base_url = "http://localhost:8000/v1"
api_key = "local-key"

[providers.groq]
base_url = "http://127.0.0.1:9999/openai/v1"
api_key_env = "LLM_BENCH_TEST_UNSET_VARIABLE"

[prices."groq:llama-3.3-70b-versatile"]
input = 0.59
output = 0.79

[prices.my-model]
input = 1.0
output = 2.0
"#;

    #[test]
    fn parses_providers_and_prices() {
        let c = FileConfig::parse(SAMPLE).unwrap();
        assert_eq!(c.base_url("vllm"), Some("http://localhost:8000/v1"));
        assert_eq!(c.api_key("vllm"), "local-key");
        assert_eq!(c.api_key("groq"), "");
        assert_eq!(c.api_key("nope"), "");
        let t = c.price_table();
        assert_eq!(
            t.get("groq", "llama-3.3-70b-versatile").map(|p| p.input),
            Some(0.59)
        );
        assert!(t.get("together", "llama-3.3-70b-versatile").is_none());
        // A bare model key applies to every provider.
        assert_eq!(t.get("anything", "my-model").map(|p| p.output), Some(2.0));
        let p = t.get("groq", "llama-3.3-70b-versatile").unwrap();
        assert!((p.cost(1_000_000, 1_000_000) - 1.38).abs() < 1e-12);
    }

    #[test]
    fn example_config_parses() {
        let c = FileConfig::parse(include_str!("../examples/llm-bench.toml")).unwrap();
        assert_eq!(c.base_url("vllm"), Some("http://localhost:8000/v1"));
        let t = c.price_table();
        assert!(t.get("groq", "llama-3.3-70b-versatile").is_some());
        assert!(t.get("vllm", "meta-llama/Llama-3.1-8B-Instruct").is_some());
    }

    #[test]
    fn rejects_bad_names_unknown_keys_and_bad_prices() {
        assert!(FileConfig::parse("[providers.\"My Server\"]\nbase_url='x'").is_err());
        assert!(FileConfig::parse("[providers.a]\nbaseurl='x'").is_err());
        assert!(FileConfig::parse("[prices.m]\ninput=-1\noutput=1").is_err());
        assert!(FileConfig::parse("[prices.m]\ninput=1").is_err());
        assert!(FileConfig::parse("nonsense = = 1").is_err());
        // A dotted bare key is a nested table in TOML; the error says to quote it.
        let dotted = FileConfig::parse("[prices.gpt-4.1]\ninput = 1\noutput = 2").unwrap_err();
        assert!(dotted.contains("[prices.\"gpt-4.1\"]"), "{dotted}");
        let quoted = FileConfig::parse("[prices.\"gpt-4.1\"]\ninput = 1\noutput = 2").unwrap();
        assert!(quoted.price_table().get("openai", "gpt-4.1").is_some());
        assert_eq!(FileConfig::parse("").unwrap(), FileConfig::default());
        assert!(FileConfig::parse("[prices.m]
input=1
output=1
cached_input=-0.5").is_err());
    }

    #[test]
    fn cached_input_price_defaults_to_input() {
        let c = FileConfig::parse("[prices.m]
input = 2.0
output = 4.0
cached_input = 0.5").unwrap();
        let p = c.price_table().get("x", "m").unwrap();
        assert_eq!(p.cached_input, Some(0.5));
        // 1M uncached at $2 + 1M cached at $0.50 + 1M out at $4.
        assert!((p.cost_cached(1_000_000, 1_000_000, 1_000_000) - 6.5).abs() < 1e-12);
        let flat = ModelPrice::new(2.0, 4.0);
        assert!((flat.cost_cached(1_000_000, 1_000_000, 0) - 4.0).abs() < 1e-12);
        assert_eq!(flat.cost(1_000_000, 0), flat.cost_cached(1_000_000, 0, 0));
    }

    #[test]
    fn missing_key_env_names_the_unset_variable() {
        let c = FileConfig::parse(SAMPLE).unwrap();
        // groq asks for LLM_BENCH_TEST_UNSET_VARIABLE, which is not set.
        assert_eq!(c.missing_key_env("groq"), Some("LLM_BENCH_TEST_UNSET_VARIABLE"));
        // vllm has a key in the file; unknown providers ask for nothing.
        assert_eq!(c.missing_key_env("vllm"), None);
        assert_eq!(c.missing_key_env("nope"), None);
        let no_env = FileConfig::parse("[providers.lm]
base_url = \"http://x\"").unwrap();
        assert_eq!(no_env.missing_key_env("lm"), None);
        // PATH is set on every platform the tests run on.
        let set = FileConfig::parse("[providers.lm]
base_url = \"http://x\"
api_key_env = \"PATH\"").unwrap();
        assert_eq!(set.missing_key_env("lm"), None);
    }
}
