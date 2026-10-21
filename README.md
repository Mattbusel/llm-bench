# llm-bench

<!-- outcome-block -->
**Pick the fastest, cheapest LLM for your prompts from measurements on your own machine: time to first token, p95 latency, tokens per second, cost and success rate.**

**Why this over the alternatives:** Leaderboards measure someone else's prompts on someone else's network. This runs yours against OpenAI, Anthropic, Ollama, Gemini, OpenRouter or any OpenAI-compatible server.

**Guarantee:** if it doesn't do what this README says, [open an issue](https://github.com/Mattbusel/llm-bench/issues) and I'll fix it.
<!-- outcome-block -->

A command-line tool and Rust library for anyone choosing an LLM: it sends the same prompts to several models and measures time to first token, latency percentiles, tokens per second, cost and success rate, from your own machine.

<p align="center">
  <img src="https://gitlab.com/mattbusel/llm-bench/-/raw/main/assets/banner.png" alt="llm-bench: send the same prompts to several models and compare p50/p99 latency, tokens per second and cost" width="100%">
</p>

It talks to OpenAI, Anthropic, a local Ollama, OpenRouter, Google Gemini, or any OpenAI-compatible server (vLLM, LM Studio, llama.cpp server, a proxy).

```bash
llm-bench run --models gpt-4o-mini,claude-haiku-4-5,ollama:qwen2.5:7b --prompts "Say hi" --stream --summary-file summary.json
```

As a library (10 lines, `cargo run --example local_server` runs a complete version against a built-in server):

```rust,no_run
use llm_bench::{runner::BenchRunner, stats::model_reports, types::{BenchConfig, ProviderConfig}};
# async fn demo() -> Result<(), llm_bench::error::BenchError> {
let ollama = ProviderConfig { name: "ollama".into(), model: "qwen2.5:7b".into(), api_key: String::new(), base_url: String::new(), max_tokens: 128 };
let config = BenchConfig { prompts: vec!["Say hi".into()], runs_per_prompt: 10, concurrency: 2,
    providers: vec![ollama], stream: true, retries: 2, warmup: 1 };
let (results, failures) = BenchRunner::new()?.run_detailed(&config, |_, _| {}).await?;
for m in model_reports(&results, &failures) {
    println!("{}: TTFT p95 {:?} ms", m.model, m.ttft_ms.map(|s| s.p95));
}
# Ok(()) }
```

## Why this tool

| You want | llm-bench | Alternatives |
|---|---|---|
| Compare hosted and local models in one run | OpenAI, Anthropic, Ollama, OpenRouter, Gemini and any OpenAI-compatible URL, side by side, one table | [LLMPerf](https://github.com/ray-project/llmperf) and vLLM's `benchmark_serving.py` are Python and aimed at one serving endpoint at a time |
| Streaming metrics | Time to first token, time per output token, p50/p90/p95/p99, mean and stddev (`--summary-file`) | LLMPerf and GenAI-Perf report the same metrics; [oha](https://crates.io/crates/oha) measures HTTP latency but does not parse tokens |
| Cost per request | Built-in prices for 15 OpenAI and Anthropic models, or OpenRouter's own per-request cost | Most load tools do not price requests |
| A single static binary for CI | Linux and Windows downloads, `cargo binstall`, a [GitLab CI/CD component](https://gitlab.com/explore/catalog/mattbusel/llm-ci) | Python tools need an environment |
| A library | `runner`, `providers`, `stats` and `report` are public; `default-features = false` leaves out the CLI dependencies | |

What it does not do: sustained load tests at a target request rate (use oha, goose or GenAI-Perf), and quality or accuracy evaluation.

<p align="center">
  <img src="https://gitlab.com/mattbusel/llm-bench/-/raw/main/assets/demo.gif" alt="llm-bench models, then llm-bench run comparing gpt-4o-mini, gpt-4o and claude-haiku-4-5 with a live progress bar and a results table" width="100%">
</p>
<p align="center"><sub>Real recording of llm-bench 0.2.1 on 2026-09-25. It is pointed at the bundled <a href="https://gitlab.com/mattbusel/llm-bench/-/blob/main/scripts/mock_server.py">local mock server</a> (canned replies, simulated latency), so no API keys were used and the numbers are not real model speeds.</sub></p>

## Install

**Linux** (x86_64, Ubuntu 20.04+ / Debian 11+). One line, no dependencies, installs to `~/.local/bin`:

```sh
mkdir -p ~/.local/bin && curl -fsSL https://gitlab.com/mattbusel/llm-bench/-/releases/permalink/latest/downloads/llm-bench-linux-x86_64.tar.gz | tar xz --strip-components=1 -C ~/.local/bin --wildcards '*/llm-bench'
```

| Other systems | |
|---|---|
| **Windows** | [Download llm-bench-windows-x86_64.exe](https://gitlab.com/mattbusel/llm-bench/-/releases/permalink/latest/downloads/llm-bench-windows-x86_64.exe) and run it. (Unsigned, so SmartScreen may ask: *More info*, then *Run anyway*.) |
| **macOS, or from source** | `cargo install --locked llm-bench` (or `cargo binstall llm-bench` for the prebuilt Linux/Windows binary) |

As a library: `cargo add llm-bench --no-default-features` (the CLI dependencies are the `cli` feature).

Needs a key for the hosted providers you benchmark, or a local Ollama. Every release, with SHA-256 checksums: [Releases](https://gitlab.com/mattbusel/llm-bench/-/releases).

**In GitLab CI:** benchmark models on every pipeline and fail when p99 latency or the success rate slips with the [`llm-benchmark`](https://gitlab.com/explore/catalog/mattbusel/llm-ci) CI/CD component.

## Use it in 3 steps

**1. See which models it knows prices for**

```bash
llm-bench models
```

You get a table of models with prompt and completion prices per 1 000 tokens.

**2. Give it a key** (only the providers you benchmark need one)

```bash
export OPENAI_API_KEY=sk-...           # PowerShell: $env:OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY=sk-ant-...    # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY=sk-or-...    # for openrouter: models
export GEMINI_API_KEY=...              # for gemini: models
```

**3. Run a benchmark**

```bash
llm-bench run --models gpt-4o-mini,claude-haiku-4-5 --prompts "Explain Rust in one sentence,Write a haiku about latency" --runs 4
```

A progress bar counts the requests, then one row per model appears, followed by the fastest and cheapest model.

Add `--stream` to measure time to first token, and `--retries 2` if your account hits rate limits.

No key? Try it against a local model or the bundled mock server:

```bash
# A local Ollama; no key needed (OLLAMA_HOST or --ollama-url for another address)
llm-bench run --models ollama:llama3.2 --prompts "Say hi" --stream

# The mock server from the recording (Python 3, no dependencies; download scripts/mock_server.py from this repo)
python scripts/mock_server.py
llm-bench run --openai-base-url http://127.0.0.1:8787 --anthropic-base-url http://127.0.0.1:8787 \
  --models gpt-4o-mini,gpt-4o,claude-haiku-4-5 --prompts "Say hi"
```

## Providers

| Prefix in `--models` | Provider | Key | Default address |
|---|---|---|---|
| `openai:` (or bare `gpt-*`, `o1*`, `o3*`, `o4*`) | OpenAI, or any OpenAI-compatible server with `--openai-base-url` | `OPENAI_API_KEY` (optional with a custom URL) | `https://api.openai.com` |
| `anthropic:` (or bare `claude-*`) | Anthropic | `ANTHROPIC_API_KEY` | `https://api.anthropic.com` |
| `ollama:` | Ollama on your machine | none | `http://localhost:11434` (`--ollama-url`, `OLLAMA_HOST`) |
| `openrouter:` | OpenRouter | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| `gemini:` (or bare `gemini-*`) | Google Gemini (OpenAI-compatible endpoint) | `GEMINI_API_KEY` | `https://generativelanguage.googleapis.com/v1beta/openai` |

Mix them in one run: `--models gpt-4o-mini,claude-haiku-4-5,ollama:qwen2.5:7b,gemini:gemini-2.5-flash`.

## Results

This is the exact output of the run in the recording (llm-bench 0.2.1, local mock server, 2026-09-25):

```text
→ Benchmarking 3 model(s) × 2 prompt(s) × 4 run(s) = 24 total requests

Results: 24 succeeded, 0 failed (100% success rate)

╭───────────┬──────────────────┬──────────┬──────────┬───────┬───────────┬────────────┬─────────╮
│ Provider  │ Model            │ P50 (ms) │ P99 (ms) │ Tok/s │  Avg Cost │ Total Cost │ Success │
├───────────┼──────────────────┼──────────┼──────────┼───────┼───────────┼────────────┼─────────┤
│ anthropic │ claude-haiku-4-5 │      488 │      559 │  92.2 │ $0.000224 │  $0.001788 │    100% │
│ openai    │ gpt-4o           │      734 │      845 │  74.9 │ $0.000804 │  $0.006435 │    100% │
│ openai    │ gpt-4o-mini      │      294 │      442 │ 113.0 │ $0.000023 │  $0.000184 │    100% │
╰───────────┴──────────────────┴──────────┴──────────┴───────┴───────────┴────────────┴─────────╯
Fastest: gpt-4o-mini (p50 294 ms)   Cheapest: gpt-4o-mini ($0.000023 per request)
```

Latency comes from the mock's simulated delays, and cost is the built-in price table applied to the token counts the mock reported. (gpt-4o has since been repriced to $2.50 / $10 per million tokens, so 0.3.0 shows a lower cost for it.)

With `--stream` the latency columns are time to first token. The same mock (which now also streams) with llm-bench 0.3.0, 3 runs of 2 prompts:

```text
╭───────────┬──────────────────┬──────────┬──────────┬───────┬───────────┬────────────┬─────────╮
│ Provider  │ Model            │ P50 (ms) │ P99 (ms) │ Tok/s │  Avg Cost │ Total Cost │ Success │
├───────────┼──────────────────┼──────────┼──────────┼───────┼───────────┼────────────┼─────────┤
│ anthropic │ claude-haiku-4-5 │      162 │      188 │  98.8 │ $0.000222 │  $0.001332 │    100% │
│ openai    │ gpt-4o           │      267 │      329 │  75.3 │ $0.000525 │  $0.003150 │    100% │
│ openai    │ gpt-4o-mini      │      138 │      175 │ 107.3 │ $0.000023 │  $0.000139 │    100% │
╰───────────┴──────────────────┴──────────┴──────────┴───────┴───────────┴────────────┴─────────╯
Fastest: gpt-4o-mini (p50 138 ms)   Cheapest: gpt-4o-mini ($0.000023 per request)

Latency columns are time to first token (--stream).
```

And a real run against two small local models served by [Ollama](https://ollama.com) on this Windows PC (2026-09-25, llm-bench 0.2.2, `--concurrency 1 --max-tokens 128`):

```text
$ llm-bench run --openai-base-url http://localhost:11434 --models openai:qwen2.5:0.5b,openai:qwen2.5:1.5b \
    --prompts "Explain Rust in one sentence,Write a haiku about latency" --runs 5 --concurrency 1 --max-tokens 128

Results: 20 succeeded, 0 failed (100% success rate)

╭──────────┬──────────────┬──────────┬──────────┬───────┬──────────┬────────────┬─────────╮
│ Provider │ Model        │ P50 (ms) │ P99 (ms) │ Tok/s │ Avg Cost │ Total Cost │ Success │
├──────────┼──────────────┼──────────┼──────────┼───────┼──────────┼────────────┼─────────┤
│ openai   │ qwen2.5:0.5b │       54 │      648 │ 371.2 │      n/a │        n/a │    100% │
│ openai   │ qwen2.5:1.5b │       76 │      131 │ 255.5 │      n/a │        n/a │    100% │
╰──────────┴──────────────┴──────────┴──────────┴───────┴──────────┴────────────┴─────────╯
Fastest: qwen2.5:0.5b (p50 54 ms)
```

The p99 of the 0.5b model is one slow request (648 ms) among fast ones, which is exactly what a p50-only number would hide. Local models have no price, so cost shows `n/a`. Against the real APIs the same table shows what you actually get from your network, with your prompts, today.

What the columns mean:

- **P50 / P99**: median and 99th-percentile response time over all runs of that model: time to the full response, or time to first token with `--stream`.
- **Tok/s**: completion tokens divided by response time, averaged.
- **Avg / Total Cost**: the provider's token counts times the built-in price table (`llm-bench models`), or the cost OpenRouter reports for the request. Models without a price show `n/a`.
- **Success**: share of requests that returned a response. Failed requests are listed under the table with a hint (for example "the API key was rejected" or "could not reach the server"), and if every request fails the command exits with status 1.

<details>
<summary><b>All options</b></summary>

### `llm-bench run`

| Flag | Default | Description |
|------|---------|-------------|
| `--models <MODEL,...>` | `gpt-4o-mini,claude-haiku-4-5` | Comma-separated model IDs, with a provider prefix from the table above where needed |
| `--prompts <PROMPT,...>` | - | Inline prompts, comma-separated |
| `--prompt-file <FILE>` | - | A file with one prompt per line (use this for prompts that contain commas) |
| `--runs <N>` | `3` | Times each prompt is sent to each model |
| `--concurrency <N>` | `4` | Maximum requests in flight at once |
| `--max-tokens <N>` | `512` | Maximum completion tokens per request |
| `--output table\|json\|csv` | `table` | Print the summary table, or every individual result as JSON or CSV (status lines then go to stderr, so stdout is clean data) |
| `--stream` | off | Stream responses and measure time to first token |
| `--warmup <N>` | `0` | Requests sent to each model first and discarded (cold starts, local models loading) |
| `--summary-file <FILE>` | - | Per-model statistics as JSON: count, mean, stddev, min, p50/p90/p95/p99, max of latency, TTFT, total time and time per output token |
| `--retries <N>` | `0` | Retry a request on HTTP 408/409/429/5xx, timeouts and connection errors, with exponential backoff and jitter; `Retry-After` on a 429 is honoured (up to 60 s). Latency is that of the attempt that succeeded |
| `--output-file <FILE>` | - | Also save every individual result as JSON |
| `--openai-key <KEY>` | `$OPENAI_API_KEY` | OpenAI API key |
| `--anthropic-key <KEY>` | `$ANTHROPIC_API_KEY` | Anthropic API key |
| `--openai-base-url <URL>` | `$OPENAI_BASE_URL`, else `https://api.openai.com` | Any OpenAI-compatible server. `http://host:port` and `http://host:port/v1` both work. With a custom URL the key is optional |
| `--anthropic-base-url <URL>` | `$ANTHROPIC_BASE_URL`, else `https://api.anthropic.com` | A proxy or gateway for the Anthropic Messages API |
| `--openrouter-key <KEY>` | `$OPENROUTER_API_KEY` | OpenRouter API key |
| `--gemini-key <KEY>` | `$GEMINI_API_KEY` | Google Gemini API key |
| `--ollama-url <URL>` | `$OLLAMA_HOST`, else `http://localhost:11434` | Ollama server for `ollama:` models |

### `llm-bench models`

Lists the built-in price table (USD per 1 000 tokens) used for cost estimates.

```text
Provider    Model                Prompt/1k   Completion/1k
──────────────────────────────────────────────────────────
openai      gpt-4o               $0.002500       $0.010000
openai      gpt-4o-mini          $0.000150       $0.000600
openai      gpt-4.1              $0.002000       $0.008000
openai      gpt-4.1-mini         $0.000400       $0.001600
openai      gpt-4.1-nano         $0.000100       $0.000400
openai      gpt-4-turbo          $0.010000       $0.030000
openai      o3                   $0.002000       $0.008000
openai      o4-mini              $0.001100       $0.004400
anthropic   claude-opus-5-5      $0.004000       $0.020000
anthropic   claude-sonnet-5-5    $0.002000       $0.010000
anthropic   claude-opus-5        $0.005000       $0.025000
anthropic   claude-sonnet-5      $0.002000       $0.010000
anthropic   claude-opus-4-6      $0.005000       $0.025000
anthropic   claude-sonnet-4-6    $0.003000       $0.015000
anthropic   claude-haiku-4-5     $0.001000       $0.005000
```

A model that is not in the table shows `n/a` for cost (before 0.3.0, unlisted models on the official endpoints were charged a made-up flat $0.002 per 1 000 tokens).

### `llm-bench version` / `--version`

Prints the version. `llm-bench --help` shows examples.

</details>

<details>
<summary><b>JSON output</b></summary>

`--output json` or `--output-file results.json` writes one object per successful request (`ttft_ms` and `tpot_ms` appear only with `--stream`, `retries` only when a request needed one):

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

</details>

<details>
<summary><b>How it works, and limits</b></summary>

```text
src/
  lib.rs        the library (docs.rs/llm-bench)
  main.rs       CLI wiring, progress bar, failure report
  cli.rs        clap arguments, providers, base URL handling, BenchConfig builder
  runner.rs     concurrent dispatch bounded by a semaphore, retries (backon)
  providers.rs  OpenAI-compatible + Anthropic calls, streaming (eventsource-stream), price table
  report.rs     p50/p99 aggregation, success rates, table, JSON and CSV (csv crate)
  types.rs      BenchResult, BenchFailure, BenchConfig, BenchSummary
  error.rs      typed BenchError
```

- Two API shapes: OpenAI Chat Completions (OpenAI, Ollama, OpenRouter, Gemini and other compatible servers) and Anthropic Messages.
- On the official OpenAI API the token limit is sent as `max_completion_tokens`, which reasoning models (o1, o3, o4-mini) require; other servers get `max_tokens`.
- `--prompts` splits on commas; use `--prompt-file` for prompts with commas.
- The price table is built in; update `src/providers.rs` when prices change.
- Results are printed when the whole run finishes; Ctrl+C stops the run without a report.

</details>

## Cargo features

| Feature | Default | What it adds |
|---|---|---|
| `cli` | on | The `llm-bench` binary: clap, indicatif progress bar, tabled table, colored output, tracing-subscriber |

`llm-bench = { version = "0.3", default-features = false }` gives the library (runner, providers, stats, report, CSV/JSON output) with tokio, reqwest (rustls), serde, backon, eventsource-stream and csv.

## Examples

| Example | What it shows |
|---|---|
| `local_server` | Measures a streaming server started inside the example; no key or network needed |
| `latency_gate` | Exit 1 when p95 time to first token is over a limit (a CI gate) |
| `compare_providers` | Every provider you have a key for, plus Ollama if it is running |

## Accuracy

`tests/timing.rs` serves a stream with a known 300 ms wait before the first token and 200 ms before the rest, and checks that llm-bench reports time to first token between 300 and 450 ms, total time between 500 and 650 ms, and time per output token from the same numbers. Against a server that answers every request after exactly 100 ms (100 requests, concurrency 4, Windows 11, i7-13700KF), llm-bench 0.3.0 reported p50 100 ms, p99 102 ms, mean 100.1 ms; 0.2.4 reported p50 100 ms, p99 103 ms.

## Development

```bash
cargo test     # providers run against local mock servers, no API keys needed
cargo clippy -- -D warnings
cargo fmt --check
```

MIT licensed. Related: [tokio-prompt-orchestrator](https://gitlab.com/mattbusel/tokio-prompt-orchestrator), a Rust orchestration layer for LLM pipelines, and the [rust-crates](https://gitlab.com/mattbusel/rust-crates) index.
