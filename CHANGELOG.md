# Changelog

## 0.2.0 (2026-09-25)

- Anthropic model table updated to current models (`claude-haiku-4-5`, `claude-sonnet-5`, `claude-opus-5`) with current prices. The Claude 3.x models previously listed, including the old default `claude-3-5-haiku-20241022`, have been retired by Anthropic, so the default `llm-bench run` failed on the Anthropic side. The default is now `gpt-4o-mini,claude-haiku-4-5`.
- Prebuilt binaries for Windows, macOS (Apple Silicon and Intel) and Linux on every GitHub Release, with SHA256SUMS.txt.
- Published on crates.io: `cargo install llm-bench`.

## 0.1.0

- Initial version: concurrent OpenAI and Anthropic benchmarking with p50/p99 latency, tokens per second, cost and success rate.
