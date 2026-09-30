# Changelog

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
