# llm-bench

[![CI](https://github.com/Mattbusel/llm-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/llm-bench/actions/workflows/ci.yml)

A command-line benchmark for LLM APIs: send the same prompts to OpenAI and Anthropic models concurrently and compare p50/p99 latency, tokens per second, cost per request and success rate in one table or JSON file.

Vendor latency numbers rarely match what you see from your own network with your own prompts. `llm-bench` runs your prompts N times against each model with a bounded number of requests in flight, then aggregates the results per model, so you can pick a model on measured speed and cost instead of a pricing page.

---

## Installation

### Download (no Rust needed)

Grab the file for your system from the [latest release](https://github.com/Mattbusel/llm-bench/releases/latest):

| System | File |
|--------|------|
| Windows | `llm-bench-vX.Y.Z-x86_64-pc-windows-msvc.zip` |
| macOS (Apple Silicon) | `llm-bench-vX.Y.Z-aarch64-apple-darwin.tar.gz` |
| macOS (Intel) | `llm-bench-vX.Y.Z-x86_64-apple-darwin.tar.gz` |
| Linux (x86_64) | `llm-bench-vX.Y.Z-x86_64-unknown-linux-gnu.tar.gz` |

Unzip it and run `llm-bench --help` (`llm-bench.exe` on Windows) from a terminal. `SHA256SUMS.txt` in the release lets you verify the download.

The binaries are not code-signed. Windows SmartScreen may say "unknown publisher": click **More info**, then **Run anyway**. On macOS, if it is blocked, right-click the file and choose **Open** (or run `xattr -d com.apple.quarantine llm-bench`).

### With Cargo

```bash
cargo install llm-bench
```

### From source

```bash
git clone https://github.com/Mattbusel/llm-bench
cd llm-bench
cargo build --release
./target/release/llm-bench --help
```

---

## Quick Start

```bash
# Set API keys once
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Run a benchmark with the defaults (gpt-4o-mini vs claude-3-5-haiku)
llm-bench run --prompts "Explain Rust in one sentence,Write a haiku about latency"

# Compare flagship models with 5 runs per prompt
llm-bench run \
 --models gpt-4o,claude-sonnet-5 \
 --prompts "Summarise the CAP theorem" \
 --runs 5 \
 --concurrency 8

# Use a file of prompts and save JSON output
llm-bench run \
 --prompt-file prompts.txt \
 --output json \
 --output-file results.json
```

---

## Output

A progress bar while requests run, then one row per model. The layout looks like this (values depend entirely on your prompts, network and the providers at the time):

```
Results: 12 succeeded, 0 failed (100% success rate)

| Provider  | Model                     | P50 (ms) | P99 (ms) | Tok/s | Avg Cost | Total Cost | Success |
| anthropic | claude-haiku-4-5 |      ... |      ... |   ... |      ... |        ... |     ... |
| openai    | gpt-4o-mini               |      ... |      ... |   ... |      ... |        ... |     ... |
```

- **P50 / P99**: latency percentiles over all runs of that model (full response time; requests are not streamed).
- **Tok/s**: completion tokens divided by response time, averaged.
- **Avg / Total Cost**: from the provider's token counts and the built-in price table (`llm-bench models`).

Results are printed when the whole run finishes; Ctrl+C aborts the run without a report.

---

## CLI Reference

### `llm-bench run`

| Flag | Default | Description |
|------|---------|-------------|
| `--openai-key <KEY>` | `$OPENAI_API_KEY` | OpenAI API key |
| `--anthropic-key <KEY>` | `$ANTHROPIC_API_KEY` | Anthropic API key |
| `--models <MODEL,...>` | `gpt-4o-mini,claude-haiku-4-5` | Comma-separated model IDs. Prefix with `openai:` or `anthropic:` to disambiguate, or use bare names for known models |
| `--prompts <PROMPT,...>` | - | Inline prompts (comma-separated) |
| `--prompt-file <FILE>` | - | Path to a file with one prompt per line |
| `--runs <N>` | `3` | Runs per prompt (for statistical stability) |
| `--concurrency <N>` | `4` | Maximum concurrent in-flight API calls |
| `--output [table\|json]` | `table` | Output format |
| `--output-file <FILE>` | - | Save full JSON results to a file |
| `--max-tokens <N>` | `512` | Maximum completion tokens per request |

### `llm-bench models`

Lists all supported models with prompt and completion pricing (USD / 1 000 tokens).

```
Supported models and pricing (USD per 1 000 tokens):

Provider     Model                                  Prompt/1k   Completion/1k
openai       gpt-4o                                 $0.005000       $0.015000
openai       gpt-4o-mini                            $0.000150       $0.000600
openai       gpt-4-turbo                            $0.010000       $0.030000
anthropic    claude-sonnet-5                        $0.002000       $0.010000
anthropic    claude-haiku-4-5                       $0.001000       $0.005000
anthropic    claude-opus-5                          $0.005000       $0.025000
```

### `llm-bench version`

```
llm-bench 0.2.0
```

---

## Model Selection

Model strings are resolved in order:

1. **Explicit prefix** - `openai:gpt-4o`, `anthropic:claude-haiku-4-5`
2. **Auto-detect** - `gpt-*` and `o1*`/`o3*` go to OpenAI; `claude-*` goes to Anthropic
3. **Error** - anything else; disambiguate with a prefix

---

## JSON Output Schema

Each element in the output array is a `BenchResult`:

```json
{
 "provider": "openai",
 "model": "gpt-4o-mini",
 "prompt": "Explain Rust in one sentence",
 "latency_ms": 634,
 "total_ms": 634,
 "prompt_tokens": 12,
 "completion_tokens": 47,
 "cost_usd": 0.0000298,
 "tokens_per_second": 74.2,
 "response_text": "Rust is a systems programming language...",
 "run_index": 0
}
```

---

## Development

```bash
# Build
cargo build --release

# Run tests (126 tests; providers are tested against a local wiremock server, no API keys needed)
cargo test

# Check lint
cargo clippy --all-features -- -D warnings
```

---

## Architecture

```
src/
 main.rs - CLI wiring, progress bar, Ctrl+C handler
 cli.rs - clap argument structs + BenchConfig builder
 runner.rs - concurrent task dispatch (semaphore-bounded)
 providers.rs - OpenAI + Anthropic HTTP calls, cost calculation
 report.rs - p50/p99 aggregation, table + JSON rendering
 types.rs - shared domain types (BenchResult, BenchConfig, …)
 error.rs - typed BenchError enum (thiserror)
```

---

## Limitations

- Two providers only (OpenAI Chat Completions and Anthropic Messages). Other OpenAI-compatible endpoints are not configurable yet.
- The price table is built in and covers six models (the Claude 3.x models it used to list have been retired by Anthropic and were replaced in 0.2.0); update `src/providers.rs` when prices change.
- `--prompts` splits on commas, so use `--prompt-file` for prompts that contain commas.
- Each request is a single non-streaming call, so time to first token is not measured separately.

---

Related: [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator), a Rust orchestration layer for LLM pipelines, and the [rust-crates](https://github.com/Mattbusel/rust-crates) index.
