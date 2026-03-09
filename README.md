# llm-bench

[![Rust](https://img.shields.io/badge/rust-1.78%2B-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Crates.io](https://img.shields.io/crates/v/llm-bench.svg)](https://crates.io/crates/llm-bench)
[![Tests](https://img.shields.io/badge/tests-126%20passing-brightgreen.svg)](#)

**Universal LLM provider benchmark CLI.**
Compare OpenAI and Anthropic models side-by-side on latency (p50/p99), cost per request, and token throughput - all from a single command.

---

## Installation

```bash
cargo install llm-bench
```

Or build from source:

```bash
git clone https://github.com/you/llm-bench
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
 --models gpt-4o,claude-3-5-sonnet-20241022 \
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

## Example Output

```
→ Benchmarking 2 model(s) × 2 prompt(s) × 3 run(s) = 12 total requests
 [00:00:08] [========================================] 12/12 (0s)

Results: 12 succeeded, 0 failed (100% success rate)

+---------------------+------------------------------+----------+----------+-------+------------+------------+---------+
| Provider | Model | P50 (ms) | P99 (ms) | Tok/s | Avg Cost | Total Cost | Success |
+---------------------+------------------------------+----------+----------+-------+------------+------------+---------+
| anthropic | claude-3-5-haiku-20241022 | 812 | 1203 | 78.4 | $0.000184 | $0.001104 | 100% |
| openai | gpt-4o-mini | 634 | 987 | 92.1 | $0.000023 | $0.000138 | 100% |
+---------------------+------------------------------+----------+----------+-------+------------+------------+---------+
```

---

## CLI Reference

### `llm-bench run`

| Flag | Default | Description |
|------|---------|-------------|
| `--openai-key <KEY>` | `$OPENAI_API_KEY` | OpenAI API key |
| `--anthropic-key <KEY>` | `$ANTHROPIC_API_KEY` | Anthropic API key |
| `--models <MODEL,...>` | `gpt-4o-mini,claude-3-5-haiku-20241022` | Comma-separated model IDs. Prefix with `openai:` or `anthropic:` to disambiguate, or use bare names for known models |
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

Provider Model Prompt/1k Completion/1k
openai gpt-4o $0.005000 $0.015000
openai gpt-4o-mini $0.000150 $0.000600
openai gpt-4-turbo $0.010000 $0.030000
anthropic claude-3-5-sonnet-20241022 $0.003000 $0.015000
anthropic claude-3-5-haiku-20241022 $0.000800 $0.004000
anthropic claude-3-opus-20240229 $0.015000 $0.075000
```

### `llm-bench version`

```
llm-bench 0.1.0
```

---

## Model Selection

Model strings are resolved in order:

1. **Explicit prefix** - `openai:gpt-4o`, `anthropic:claude-3-5-haiku-20241022`
2. **Auto-detect** - `gpt-*` and `o1*`/`o3*` → OpenAI; `claude-*` → Anthropic
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

# Run tests (126 tests)
cargo test

# Check lint
cargo clippy --all-features -- -D warnings

# Ratio check (must be >= 1.5:1)
# Production LOC: 855 | Test LOC: 1319 | Ratio: 1.543:1
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

## Connection to tokio-prompt-orchestrator

`llm-bench` is a standalone diagnostic tool for the
[tokio-prompt-orchestrator](https://github.com/you/tokio-prompt-orchestrator) project.
Use it to:
- **Select the fastest model** for a given prompt class before wiring it into the pipeline.
- **Validate cost envelopes** against the per-stage latency budgets defined in the orchestrator's `ARCHITECTURE.md`.
- **Regression-test provider SLAs** as part of CI by saving JSON results and diffing percentiles over time.

---

## License

MIT © 2026
