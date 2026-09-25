<p align="center">
  <img src="https://raw.githubusercontent.com/Mattbusel/llm-bench/main/assets/banner.png" alt="llm-bench: send the same prompts to several models and compare p50/p99 latency, tokens per second and cost" width="100%">
</p>

# llm-bench

**Find out which LLM is actually fastest and cheapest for your prompts, from your own machine, in one command.**

It sends the same prompts to several models at once (OpenAI, Anthropic, or any OpenAI-compatible server such as Ollama, vLLM or LM Studio) and prints one table: p50/p99 latency, tokens per second, cost per request and success rate.

<p align="center">
  <img src="https://raw.githubusercontent.com/Mattbusel/llm-bench/main/assets/demo.gif" alt="llm-bench models, then llm-bench run comparing gpt-4o-mini, gpt-4o and claude-haiku-4-5 with a live progress bar and a results table" width="100%">
</p>
<p align="center"><sub>Real recording of llm-bench 0.2.1 on 2026-09-25. It is pointed at the bundled <a href="https://github.com/Mattbusel/llm-bench/blob/main/scripts/mock_server.py">local mock server</a> (canned replies, simulated latency), so no API keys were used and the numbers are not real model speeds.</sub></p>

## Install

| Platform | Command |
|----------|---------|
| Windows (PowerShell) | `irm https://raw.githubusercontent.com/Mattbusel/llm-bench/main/install.ps1 \| iex` |
| macOS / Linux | `curl -fsSL https://raw.githubusercontent.com/Mattbusel/llm-bench/main/install.sh \| sh` |
| Homebrew | `brew install mattbusel/tap/llm-bench` |
| Scoop | `scoop bucket add mattbusel https://github.com/Mattbusel/scoop-bucket` then `scoop install llm-bench` |
| Rust, prebuilt | `cargo binstall llm-bench` |
| Rust, from source | `cargo install llm-bench` |
| Manual | Download a zip or tarball from [Releases](https://github.com/Mattbusel/llm-bench/releases/latest) (Windows, macOS Intel and Apple Silicon, Linux x86_64) |

The scripts verify the download against the release's `SHA256SUMS.txt`. The Windows script installs to `%LOCALAPPDATA%\Programs\llm-bench` and adds it to your user PATH; the shell script installs to `~/.local/bin`. The binaries are not code-signed: Windows SmartScreen may say "unknown publisher" (More info, then Run anyway), and on macOS a manually downloaded file may need `xattr -d com.apple.quarantine llm-bench`.

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
```

**3. Run a benchmark**

```bash
llm-bench run --models gpt-4o-mini,claude-haiku-4-5 --prompts "Explain Rust in one sentence,Write a haiku about latency" --runs 4
```

A progress bar counts the requests, then one row per model appears, followed by the fastest and cheapest model.

No key? Try it against a local model or the bundled mock server:

```bash
# Ollama (or any OpenAI-compatible server); no key needed
llm-bench run --openai-base-url http://localhost:11434 --models openai:llama3.2 --prompts "Say hi"

# The mock server from the recording (Python 3, no dependencies; download scripts/mock_server.py from this repo)
python scripts/mock_server.py
llm-bench run --openai-base-url http://127.0.0.1:8787 --anthropic-base-url http://127.0.0.1:8787 \
  --models gpt-4o-mini,gpt-4o,claude-haiku-4-5 --prompts "Say hi"
```

## Results

This is the exact output of the run in the recording (local mock server, 2026-09-25):

```
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

Latency comes from the mock's simulated delays, and cost is the built-in price table applied to the token counts the mock reported.

And a real run against two small local models served by [Ollama](https://ollama.com) on this Windows PC (2026-09-25, llm-bench 0.2.2, `--concurrency 1 --max-tokens 128`):

```
$ llm-bench run --openai-base-url http://localhost:11434 --models openai:qwen2.5:0.5b,openai:qwen2.5:1.5b \n    --prompts "Explain Rust in one sentence,Write a haiku about latency" --runs 5 --concurrency 1 --max-tokens 128

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

- **P50 / P99**: median and 99th-percentile response time over all runs of that model (full response; requests are not streamed).
- **Tok/s**: completion tokens divided by response time, averaged.
- **Avg / Total Cost**: the provider's token counts times the built-in price table (`llm-bench models`).
- **Success**: share of requests that returned a response. Failed requests are listed under the table with a hint (for example "the API key was rejected" or "could not reach the server"), and if every request fails the command exits with status 1.

<details>
<summary><b>All options</b></summary>

### `llm-bench run`

| Flag | Default | Description |
|------|---------|-------------|
| `--models <MODEL,...>` | `gpt-4o-mini,claude-haiku-4-5` | Comma-separated model IDs. `gpt-*`, `o1*`, `o3*` go to OpenAI and `claude-*` to Anthropic; prefix with `openai:` or `anthropic:` for anything else (for example `openai:llama3.2`) |
| `--prompts <PROMPT,...>` | - | Inline prompts, comma-separated |
| `--prompt-file <FILE>` | - | A file with one prompt per line (use this for prompts that contain commas) |
| `--runs <N>` | `3` | Times each prompt is sent to each model |
| `--concurrency <N>` | `4` | Maximum requests in flight at once |
| `--max-tokens <N>` | `512` | Maximum completion tokens per request |
| `--output table\|json` | `table` | Print the summary table or every individual result as JSON |
| `--output-file <FILE>` | - | Also save every individual result as JSON |
| `--openai-key <KEY>` | `$OPENAI_API_KEY` | OpenAI API key |
| `--anthropic-key <KEY>` | `$ANTHROPIC_API_KEY` | Anthropic API key |
| `--openai-base-url <URL>` | `$OPENAI_BASE_URL`, else `https://api.openai.com` | Any OpenAI-compatible server. `http://host:port` and `http://host:port/v1` both work. With a custom URL the key is optional |
| `--anthropic-base-url <URL>` | `$ANTHROPIC_BASE_URL`, else `https://api.anthropic.com` | A proxy or gateway for the Anthropic Messages API |

### `llm-bench models`

Lists the built-in price table (USD per 1 000 tokens) used for cost estimates.

```
Provider    Model                Prompt/1k   Completion/1k
──────────────────────────────────────────────────────────
openai      gpt-4o               $0.005000       $0.015000
openai      gpt-4o-mini          $0.000150       $0.000600
openai      gpt-4-turbo          $0.010000       $0.030000
anthropic   claude-sonnet-5      $0.002000       $0.010000
anthropic   claude-haiku-4-5     $0.001000       $0.005000
anthropic   claude-opus-5        $0.005000       $0.025000
```

A model that is not in the table shows `n/a` for cost when it is served from a custom base URL (Ollama, vLLM and so on). On the official endpoints an unlisted model is estimated at a flat $0.002 per 1 000 tokens, so treat that cost as a rough placeholder.

### `llm-bench version` / `--version`

Prints the version. `llm-bench --help` shows examples.

</details>

<details>
<summary><b>JSON output</b></summary>

`--output json` or `--output-file results.json` writes one object per successful request:

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

```
src/
  main.rs       CLI wiring, progress bar, failure report
  cli.rs        clap arguments, base URL handling, BenchConfig builder
  runner.rs     concurrent dispatch bounded by a semaphore; collects results and failures
  providers.rs  OpenAI Chat Completions + Anthropic Messages calls, price table
  report.rs     p50/p99 aggregation, success rates, table and JSON rendering
  types.rs      BenchResult, BenchFailure, BenchConfig, BenchSummary
  error.rs      typed BenchError
```

- Two API shapes: OpenAI Chat Completions (also used by OpenAI-compatible servers) and Anthropic Messages.
- Each request is a single non-streaming call, so time to first token is not measured separately.
- `--prompts` splits on commas; use `--prompt-file` for prompts with commas.
- The price table is built in; update `src/providers.rs` when prices change.
- Results are printed when the whole run finishes; Ctrl+C stops the run without a report.

</details>

## Development

```bash
cargo test     # 137 tests; providers run against a local wiremock server, no API keys needed
cargo clippy -- -D warnings
cargo fmt --check
```

MIT licensed. Related: [tokio-prompt-orchestrator](https://github.com/Mattbusel/tokio-prompt-orchestrator), a Rust orchestration layer for LLM pipelines, and the [rust-crates](https://github.com/Mattbusel/rust-crates) index.
