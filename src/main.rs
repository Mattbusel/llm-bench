//! # llm-bench
//!
//! Universal LLM provider benchmark CLI.
//!
//! Compares OpenAI and Anthropic models on latency (p50/p99), cost per
//! request, and token throughput.  Runs prompts concurrently against all
//! configured models and renders results as a coloured ASCII table or JSON.
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

mod cli;
mod error;
mod providers;
mod report;
mod runner;
mod types;

use cli::{Cli, Command, OutputFormat};
use error::BenchError;

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
    let config = cli::build_config(&args)?;

    let total_tasks =
        config.providers.len() * config.prompts.len() * config.runs_per_prompt as usize;

    println!(
        "{} Benchmarking {} model(s) × {} prompt(s) × {} run(s) = {} total requests",
        "→".cyan().bold(),
        config.providers.len(),
        config.prompts.len(),
        config.runs_per_prompt,
        total_tasks
    );

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
    let bench_runner = runner::BenchRunner::new()?;

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
    println!(
        "
{} {}",
        "Results:".bold(),
        if failed == 0 {
            status.green()
        } else {
            status.yellow()
        }
    );

    //  Output
    match args.output {
        OutputFormat::Table => {
            let summaries = report::generate_summary_with_failures(&results, &failures);
            report::print_table(&summaries);
        }
        OutputFormat::Json => {
            let json = report::print_results_json(&results)?;
            println!("{json}");
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
        println!("{} Results saved to {}", "→".cyan(), path.display());
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
        "Supported models and pricing (USD per 1 000 tokens)".bold()
    );
    let header = format!(
        "{:<11} {:<18} {:>11} {:>15}",
        "Provider", "Model", "Prompt/1k", "Completion/1k"
    );
    println!("{}", header.bold());
    println!("{}", "─".repeat(header.chars().count()).dimmed());

    for m in providers::supported_models() {
        println!(
            "{:<11} {:<18} {:>11} {:>15}",
            m.provider,
            m.model,
            format!("${:.6}", m.prompt_per_1k),
            format!("${:.6}", m.completion_per_1k)
        );
    }
    println!(
        "\nCost is estimated from each response's token counts and these prices.\n{} llm-bench run --models gpt-4o-mini,claude-haiku-4-5 --prompts \"Say hi\"\n",
        "Next:".cyan().bold()
    );
}
