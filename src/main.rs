//! # llm-bench
//!
//! Compare LLM providers on your own prompts from the command line: time
//! to first token (`--stream`), p50/p95 total latency, tokens per second,
//! error rate and cost per request, as a table, Markdown, JSON or CSV.
//!
//! ## Entry points
//! - `llm-bench run`      -  execute a benchmark
//! - `llm-bench models`   -  list supported models and pricing
//! - `llm-bench version`  -  print version

#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use clap::Parser;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use tracing::info;

use llm_bench::cli::{self, Cli, Command, OutputFormat};
use llm_bench::error::BenchError;
use llm_bench::{providers, report, runner};

#[tokio::main]
async fn main() {
    // Initialise tracing  -  respects RUST_LOG env var
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    if let Err(e) = run().await {
        eprintln!("{} {e}", "error:".red().bold());
        std::process::exit(1);
    }
}

async fn run() -> Result<(), BenchError> {
    let cli = Cli::parse();

    match cli.command {
        Command::Version => {
            println!("llm-bench {}", env!("CARGO_PKG_VERSION"));
        }

        Command::Models => {
            print_models_table();
        }

        Command::Run(args) => {
            run_benchmark(args).await?;
        }
    }

    Ok(())
}

async fn run_benchmark(args: cli::RunArgs) -> Result<(), BenchError> {
    // Validate configuration before touching the network
    let (config, prices) = cli::build_plan(&args)?;

    let total_tasks =
        config.providers.len() * config.prompts.len() * config.runs_per_prompt as usize;

    // With --output markdown, json or csv, stdout carries only the data, so
    // status lines go to stderr (before 0.3.0 they were mixed into the JSON).
    let status_line = |line: String| {
        if matches!(args.output, OutputFormat::Table) {
            println!("{line}");
        } else {
            eprintln!("{line}");
        }
    };

    status_line(format!(
        "{} Benchmarking {} model(s) × {} prompt(s) × {} run(s) = {} total requests",
        "→".cyan().bold(),
        config.providers.len(),
        config.prompts.len(),
        config.runs_per_prompt,
        total_tasks
    ));

    //  Progress bar
    let pb = ProgressBar::new(total_tasks as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} {bar:40.cyan/blue} {pos}/{len} requests  {elapsed_precise}  eta {eta}",
        )
        .unwrap_or_else(|_| ProgressStyle::default_bar())
        .progress_chars("━╸─"),
    );

    let pb_arc = Arc::new(pb);
    let pb_cb = Arc::clone(&pb_arc);
    let completed = Arc::new(AtomicUsize::new(0));
    let completed_cb = Arc::clone(&completed);

    //  Ctrl+C handler
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            eprintln!(
                "\n{}",
                "Interrupted  -  partial results may have been collected.".yellow()
            );
            std::process::exit(130);
        }
    });

    //  Run benchmark
    let bench_runner = runner::BenchRunner::new()?.with_prices(prices);

    let (results, failures) = bench_runner
        .run_detailed(&config, move |done, _total| {
            completed_cb.store(done, Ordering::Relaxed);
            pb_cb.set_position(done as u64);
        })
        .await?;

    pb_arc.finish_and_clear();

    info!(result_count = results.len(), "benchmark complete");

    let success = results.len();
    let failed = failures.len();
    let rate = (success * 100).checked_div(success + failed).unwrap_or(0);
    let status = format!("{success} succeeded, {failed} failed ({rate}% success rate)");
    status_line(format!(
        "
{} {}",
        "Results:".bold(),
        if failed == 0 {
            status.green()
        } else {
            status.yellow()
        }
    ));

    //  Output
    let reports = llm_bench::stats::model_reports(&results, &failures);
    match args.output {
        OutputFormat::Table => report::print_comparison(&reports),
        OutputFormat::Markdown => print!("{}", report::markdown_table(&reports)),
        OutputFormat::Json => {
            let json = report::print_results_json(&results)?;
            println!("{json}");
        }
        OutputFormat::Csv => {
            print!("{}", report::results_csv(&results)?);
        }
    }

    let lines = report::failure_lines(&failures);
    if !lines.is_empty() {
        eprintln!("{}", "Failed requests:".yellow().bold());
        for line in lines {
            eprintln!("  {line}");
        }
        eprintln!();
    }

    //  Save to file
    if let Some(ref path) = args.output_file {
        let json = report::print_results_json(&results)?;
        std::fs::write(path, &json).map_err(BenchError::Io)?;
        status_line(format!("{} Results saved to {}", "→".cyan(), path.display()));
    }

    if let Some(ref path) = args.markdown_file {
        std::fs::write(path, report::markdown_table(&reports)).map_err(BenchError::Io)?;
        status_line(format!("{} Markdown table saved to {}", "→".cyan(), path.display()));
    }

    if let Some(ref path) = args.summary_file {
        let json = serde_json::to_string_pretty(&reports).map_err(BenchError::SerdeJson)?;
        std::fs::write(path, json).map_err(BenchError::Io)?;
        status_line(format!("{} Summary saved to {}", "→".cyan(), path.display()));
    }

    if success == 0 && failed > 0 {
        return Err(BenchError::InvalidConfig {
            reason: "every request failed (reasons listed above)".into(),
        });
    }

    Ok(())
}

fn print_models_table() {
    println!(
        "\n{}\n",
        format!(
            "List prices, USD per 1M tokens (llm-prices, checked {})",
            providers::prices_date()
        )
        .bold()
    );
    let width = providers::supported_models()
        .iter()
        .map(|m| m.model.len())
        .max()
        .unwrap_or(18)
        .max(5);
    let header = format!(
        "{:<10} {:<width$} {:>10} {:>10}",
        "Provider", "Model", "Input/1M", "Output/1M"
    );
    println!("{}", header.bold());
    println!("{}", "─".repeat(header.chars().count()).dimmed());

    for m in providers::supported_models().iter() {
        println!(
            "{:<10} {:<width$} {:>10} {:>10}",
            m.provider,
            m.model,
            format!("${:.3}", m.prompt_per_1k * 1000.0),
            format!("${:.3}", m.completion_per_1k * 1000.0)
        );
    }
    let names: Vec<&str> = runner::BUILTIN_PROVIDERS.iter().map(|(n, _)| *n).collect();
    println!(
        "\nProviders: {}, or any OpenAI-compatible URL from --config.\n\
         Cost: the provider's own reported cost (OpenRouter), else a [prices] entry in --config,\n\
         else these list prices for openai, anthropic, gemini, deepseek and openrouter models.\n\
         Models hosted elsewhere (Groq, Together, Fireworks, Ollama, custom) show n/a without a [prices] entry.\n\
         {} llm-bench run --models gpt-4o-mini,claude-haiku-4-5 --prompts \"Say hi\" --stream\n",
        names.join(", "),
        "Next:".cyan().bold()
    );
}
