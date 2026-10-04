# Changelog

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
