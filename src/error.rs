//! # Module: Error
//!
//! ## Responsibility
//! Centralised, typed error surface for the entire `llm-bench` binary.
//! Every failure mode that crosses a module boundary lives here.
//!
//! ## Guarantees
//! - Every variant is named and carries structured context.
//! - No `Box<dyn Error>` escapes as a library-facing type.
//! - All variants implement `std::error::Error` via `thiserror`.
//! - This module never panics.

use thiserror::Error;

/// All errors that can be produced by `llm-bench`.
#[derive(Debug, Error)]
pub enum BenchError {
    /// The upstream API returned a non-2xx status code.
    #[error("API error from '{provider}' (model '{model}'): HTTP {status} — {body}")]
    ApiError {
        provider: String,
        model: String,
        status: u16,
        body: String,
    },

    /// The API responded with HTTP 429 (rate-limited).
    #[error("Rate limited by '{provider}': retry after {retry_after_secs:?}s")]
    RateLimited {
        provider: String,
        retry_after_secs: Option<u64>,
    },

    /// A configuration value is invalid or missing.
    #[error("Invalid configuration: {reason}")]
    InvalidConfig { reason: String },

    /// An I/O error (file read/write, stdin).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialisation or deserialisation failed.
    #[error("JSON error: {0}")]
    SerdeJson(#[from] serde_json::Error),

    /// An HTTP transport-level error (connection refused, timeout, etc.).
    #[error("HTTP transport error: {0}")]
    Http(#[from] reqwest::Error),

    /// A semaphore or task join error during concurrent execution.
    #[error("Concurrency error: {reason}")]
    Concurrency { reason: String },
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Display formatting ────────────────────────────────────────────────────

    #[test]
    fn test_api_error_display_contains_provider_model_status() {
        let err = BenchError::ApiError {
            provider: "openai".into(),
            model: "gpt-4o".into(),
            status: 401,
            body: "Unauthorized".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("openai"), "expected 'openai' in: {msg}");
        assert!(msg.contains("gpt-4o"), "expected 'gpt-4o' in: {msg}");
        assert!(msg.contains("401"), "expected '401' in: {msg}");
        assert!(msg.contains("Unauthorized"), "expected body in: {msg}");
    }

    #[test]
    fn test_rate_limited_display_with_retry_after() {
        let err = BenchError::RateLimited {
            provider: "anthropic".into(),
            retry_after_secs: Some(30),
        };
        let msg = err.to_string();
        assert!(msg.contains("anthropic"), "expected 'anthropic' in: {msg}");
        assert!(msg.contains("30"), "expected retry delay in: {msg}");
    }

    #[test]
    fn test_rate_limited_display_without_retry_after() {
        let err = BenchError::RateLimited {
            provider: "openai".into(),
            retry_after_secs: None,
        };
        let msg = err.to_string();
        assert!(msg.contains("openai"), "expected 'openai' in: {msg}");
    }

    #[test]
    fn test_invalid_config_display_contains_reason() {
        let err = BenchError::InvalidConfig {
            reason: "API key is empty".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("API key is empty"), "expected reason in: {msg}");
    }

    #[test]
    fn test_io_error_from_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let bench_err = BenchError::from(io_err);
        let msg = bench_err.to_string();
        assert!(msg.contains("file missing"), "expected io message in: {msg}");
    }

    #[test]
    fn test_serde_json_from_conversion() {
        let raw = "{ invalid json }}}";
        let json_err = serde_json::from_str::<serde_json::Value>(raw).unwrap_err();
        let bench_err = BenchError::from(json_err);
        let msg = bench_err.to_string();
        assert!(!msg.is_empty(), "expected non-empty error message");
    }

    #[test]
    fn test_concurrency_error_display() {
        let err = BenchError::Concurrency {
            reason: "task join failed".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("task join failed"), "expected reason in: {msg}");
    }

    // ── Debug trait ───────────────────────────────────────────────────────────

    #[test]
    fn test_all_variants_are_debug_printable() {
        let variants: Vec<String> = vec![
            format!(
                "{:?}",
                BenchError::ApiError {
                    provider: "p".into(),
                    model: "m".into(),
                    status: 500,
                    body: "err".into()
                }
            ),
            format!(
                "{:?}",
                BenchError::RateLimited {
                    provider: "p".into(),
                    retry_after_secs: None
                }
            ),
            format!(
                "{:?}",
                BenchError::InvalidConfig {
                    reason: "r".into()
                }
            ),
            format!(
                "{:?}",
                BenchError::Concurrency {
                    reason: "c".into()
                }
            ),
        ];
        for v in &variants {
            assert!(!v.is_empty(), "debug output should not be empty");
        }
    }

    // ── Source chain ──────────────────────────────────────────────────────────

    #[test]
    fn test_io_error_has_source() {
        use std::error::Error as StdError;
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let bench_err = BenchError::from(io_err);
        assert!(
            bench_err.source().is_some(),
            "Io variant should expose source"
        );
    }

    #[test]
    fn test_serde_json_error_has_source() {
        use std::error::Error as StdError;
        let json_err = serde_json::from_str::<serde_json::Value>("{bad}").unwrap_err();
        let bench_err = BenchError::from(json_err);
        assert!(
            bench_err.source().is_some(),
            "SerdeJson variant should expose source"
        );
    }
}
