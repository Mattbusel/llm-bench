# Changelog

## [0.4.1] - 2026-11-16

- Package metadata only: `repository` points at https://github.com/Mattbusel/llm-bench (the GitHub repository, where issues, releases and the dependents graph are), no code change from 0.4.0.

## [0.4.0] - 2026-11-14

### Added

- **Providers:** `groq:` (`GROQ_API_KEY`), `together:` (`TOGETHER_API_KEY`), `fireworks:` (`FIREWORKS_API_KEY`) and `deepseek:` (`DEEPSEEK_API_KEY`), each with a `--<name>-key` flag. Bare `deepseek-*` names are recognised.
- **`--config FILE`** (or `LLM_BENCH_CONFIG`), a TOML file: `[providers.NAME] base_url = ...` adds any OpenAI-compatible server as `--models NAME:<model>` (vLLM, LM Studio, a gateway), with `api_key_env` or `api_key`; the same table points a built-in provider at another URL; `[prices."provider:model"]` or `[prices."model"]` sets `input` / `output` in USD per 1M tokens, and optionally `cached_input` for prompt tokens served from the provider's cache (otherwise billed at `input`). A provider name from the file is matched before the bare-name rules, so `[providers.deepseek-local]` or `[providers.gpt-proxy]` is used as `--models deepseek-local:<model>` instead of being sent to DeepSeek or OpenAI. When `api_key_env` names a variable that is not set (and no flag or built-in variable supplies a key), the run stops with an error naming the variable instead of sending every request without a key. Unknown keys, bad provider names and negative prices are rejected with the file name, and a dotted bare key such as `[prices.gpt-4.1]` gets a hint to quote it. See `examples/llm-bench.toml`.
- **Comparison table:** one row per model with successful/attempted requests, **error rate**, **time to first token p50** (with `--stream`), **total latency p50 and p95**, median **tokens per second**, **cost per request** and total cost, and a "Fastest / Highest throughput / Cheapest" line under it.
- **`--output markdown`** (alias `md`) prints that table as GitHub Markdown; **`--markdown-file PATH`** writes it alongside any other output. `--output json` is unchanged (one record per request).
- **`--repeat N`**, an alias of `--runs`. `--concurrency C` caps requests in flight across all models; an end-to-end test now checks that a mock server never sees more than C at once.
- `--summary-file` reports gain `error_rate` and `cost_per_request_usd` (the mean over requests that have a price; unpriced requests are left out rather than counted as $0); per-request JSON gains `cost_source` (`provider`, `config` or `llm-prices`), `cached_prompt_tokens` and `cache_write_prompt_tokens` (Anthropic cache writes), each omitted when empty.
- Library: `llm_bench::config` (`FileConfig` with `missing_key_env`, `PriceTable`, `ModelPrice` with `new` and `cost_cached`), `BenchRunner::with_prices`, `runner::BUILTIN_PROVIDERS`, `report::{comparison_rows, markdown_table, comparison_verdict, print_comparison}`, `providers::{list_price, prices_date, LIST_PRICED_PROVIDERS}`.
- `tests/cli_mock_server.rs` runs the built binary against a local OpenAI-compatible mock (streaming and non-streaming, one model that always returns HTTP 500): Markdown and JSON output, `--markdown-file`, `--summary-file`, a config-defined provider and `groq:` redirected by the config file (checks the URL path and the `GROQ_API_KEY` bearer token), configured prices, the error rate, `--concurrency 1` and `3`, exit status 1 when every request fails, config providers named `deepseek-local` and `gpt-proxy` reaching their own URLs with vendor keys set, and the unset-`api_key_env` error.

### Changed

- **Prices come from the [llm-prices](https://crates.io/crates/llm-prices) crate** (dated, source-cited list prices for OpenAI, Anthropic, Gemini and DeepSeek; table checked 2026-10-23) instead of a hand-kept table of 15 models. `llm-bench models` lists every listed model in USD per 1M tokens with the table date.
- Cached prompt tokens are billed at the cache price: OpenAI `prompt_tokens_details.cached_tokens`, DeepSeek `prompt_cache_hit_tokens`, Anthropic `cache_read_input_tokens` and `cache_creation_input_tokens`. Repeated long prompts on OpenAI were overcharged before.
- List prices apply only to models served by their own vendor (openai, anthropic, gemini, deepseek) or by OpenRouter. The same model id on Groq, Together, Fireworks, Ollama or a custom URL shows `n/a` unless a `[prices]` entry says what that host charges, instead of being billed at the vendor's price. Cost precedence: the provider's reported cost (OpenRouter), then `[prices]`, then llm-prices. A `[prices]` entry bills cached prompt tokens at `cached_input` (or `input`) and counts them once: OpenAI-compatible hosts include them in `prompt_tokens`, Anthropic reports cache reads and writes on top of it. Anthropic cache writes are billed at `input` under a `[prices]` entry (Anthropic charges 1.25x for them); without one, the llm-prices list price bills them exactly.
- The terminal table replaces the 0.3 columns (P50, P99, Tok/s, Avg Cost, Total Cost, Success). `report::print_table`, `generate_summary` and `BenchSummary` remain for library users.
- `providers::ModelInfo` rows are built from llm-prices; `gpt-4-turbo` and `gpt-4-turbo-preview` map to `gpt-4-turbo-2024-04-09`.
- `RunArgs` and `OutputFormat` derive `Default`; `cli::build_plan` returns the `BenchConfig` and the config file's prices (`build_config` still returns only the former).

### Not yet verified

- The new hosted providers (Groq, Together, Fireworks, DeepSeek) were tested only against the local mock server, not their real endpoints. Streamed requests ask for usage with `stream_options.include_usage`; a host that leaves usage out of streamed chunks reports 0 completion tokens, so its tok/s is left out and its cost is understated. Run without `--stream` if a host's numbers look wrong.

## [0.3.0] - 2026-10-04

### Added

- **Providers:** `ollama:` (local, no key; `--ollama-url` or `OLLAMA_HOST`), `openrouter:` (`OPENROUTER_API_KEY`) and `gemini:` (`GEMINI_API_KEY`, Google's OpenAI-compatible endpoint). Bare `gemini-*` and `o4*` names are recognised. OpenRouter's own per-request cost (`usage.cost`) is used when it is reported.
- **`--stream`:** responses are read as server-sent events with [eventsource-stream](https://crates.io/crates/eventsource-stream), and latency becomes time to first token (`ttft_ms` in the JSON). Tested end to end against the bundled mock server, which now streams too and splits a two-byte character across network writes.
- **`--retries N`** (default 0, so results are unchanged unless asked): retries on 408/409/429/5xx, timeouts and connection errors with [backon](https://crates.io/crates/backon), exponential backoff with jitter; `Retry-After` on a 429 (seconds or HTTP date) sets the wait, up to 60 s.
- **`--output csv`**, written with the [csv](https://crates.io/crates/csv) crate so prompts and answers with commas, quotes or newlines stay intact.
- **`--summary-file`**: per-model count, mean, stddev, min, p50/p90/p95/p99 and max of latency, time to first token, total time and **time per output token** (`tpot_ms`, new in the JSON for streamed requests). Property-tested (proptest): percentiles are ordered and are always measured values.
- **`--warmup N`**: requests sent to each model first and discarded.
- Transport errors include their cause ("tcp connect error: ... actively refused it") instead of only "error sending request for url".
- The crate is now also a library (`llm_bench::runner`, `providers`, `stats`, `report`), documented on docs.rs with the README as its front page; the CLI dependencies (clap, indicatif, tabled, colored, tracing-subscriber) are the default `cli` feature.
- `tests/timing.rs`: a local server streams with known delays (300 ms to the first token, 200 ms more to the end) and the test checks the reported TTFT, total and time per token against them. Against a fixed 100 ms server, 0.3.0 reports p50 100 ms and p99 102 ms over 100 requests (0.2.4: 100 / 103).
- Examples: `local_server` (no key needed), `latency_gate`, `compare_providers`.
- Prices for claude-opus-5-5 ($4 / $20) and claude-sonnet-5-5 ($2 / $10).

### Fixed

- **OpenAI reasoning models failed:** `o1`, `o3` and `o4-mini` reject `max_tokens` with HTTP 400. The official OpenAI API now gets `max_completion_tokens`; other servers still get `max_tokens`.
- **`--output json` printed status lines into stdout** ("Benchmarking ...", "Results: ..."), so piping it into a JSON parser failed. With `--output json` or `csv`, status lines now go to stderr.
- **Invented prices removed:** an unlisted model on the official endpoints was charged a made-up flat $0.002 per 1 000 tokens; it now shows `n/a`, as unlisted models on custom URLs already did.
- **gpt-4o price** corrected to $2.50 / $10 per million tokens (was $5 / $15). Added gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, o3, o4-mini, claude-opus-4-6 and claude-sonnet-4-6.
- `cargo binstall` metadata pointed at GitHub-style release URLs; it now uses the GitLab release downloads for Linux and Windows x86_64.
- `cargo clippy --all-targets` failed (the crate denies `unwrap_used`, and the tests used it); tests are now exempt.

### Changed

- The JSON results keep every 0.2 field with the same meaning; `ttft_ms` and `retries` appear only when set.
- `--concurrency 0` is treated as 1 instead of hanging.
- Minimum Rust version 1.87 for the library and binary (checked with `cargo +1.87.0 check`; the test-only wiremock needs 1.88). GitLab CI runs tests with all and with no default features, clippy and rustdoc with `-D warnings`, and an MSRV check. The dead GitHub workflows are gone.

## [0.2.4] - 2026-09-30

- HTTPS through rustls instead of the system OpenSSL. The 0.2.3 Linux download needed OpenSSL 1.1, which Ubuntu 22.04 and newer do not ship; this one runs on any glibc 2.31+ distribution.

## [0.2.3] - 2026-09-30

- Links point at GitLab.
- Linux x86_64 release builds from GitLab CI; downloads and links now point at GitLab.

## 0.2.2 (2026-09-25)

- Models that are not in the price table and are served from a custom base URL (Ollama, vLLM, LM Studio) now show `n/a` for cost instead of a made-up fallback price, and the "Cheapest" verdict only compares models that have a price.
- README: a real benchmark of two local Ollama models.

## 0.2.1 (2026-09-25)

- `--openai-base-url` / `OPENAI_BASE_URL` and `--anthropic-base-url` / `ANTHROPIC_BASE_URL`: benchmark any OpenAI-compatible server (Ollama, vLLM, LM Studio, a proxy). With a custom URL the API key is optional.
- The Success column is now real: failed requests are counted per model (it used to always show 100%), models whose every request failed still get a row, and the reasons are listed under the table with a hint on how to fix common ones. A run where every request fails exits with status 1.
- A "Fastest / Cheapest" line under the table, a rounded table with right-aligned numbers, a cleaner progress bar and `llm-bench models` layout.
- Missing-key errors say exactly which variable to set (bash and PowerShell) and `--help` ends with examples.
- `scripts/mock_server.py`: a local stand-in for both APIs (canned replies, simulated latency) for trying the tool without keys.
- Install one-liners: `install.sh`, `install.ps1`, Homebrew, Scoop and `cargo binstall` metadata.

## 0.2.0 (2026-09-25)

- Anthropic model table updated to current models (`claude-haiku-4-5`, `claude-sonnet-5`, `claude-opus-5`) with current prices. The Claude 3.x models previously listed, including the old default `claude-3-5-haiku-20241022`, have been retired by Anthropic, so the default `llm-bench run` failed on the Anthropic side. The default is now `gpt-4o-mini,claude-haiku-4-5`.
- Prebuilt binaries for Windows, macOS (Apple Silicon and Intel) and Linux on every GitHub Release, with SHA256SUMS.txt.
- Published on crates.io: `cargo install llm-bench`.

## 0.1.0

- Initial version: concurrent OpenAI and Anthropic benchmarking with p50/p99 latency, tokens per second, cost and success rate.
