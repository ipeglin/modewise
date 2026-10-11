//! The `modewise` command. Also reachable from the Python package.

mod commands;

use std::ffi::OsString;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// Exit code when every recording succeeded.
pub const OK: u8 = 0;
/// Exit code for invalid arguments, specs or folders.
pub const ERROR: u8 = 1;
/// Exit code when the run finished but some recordings failed.
pub const PARTIAL: u8 = 2;

#[derive(Parser)]
#[command(
    name = "modewise",
    version,
    about = "Data-adaptive signal decomposition pipelines"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run a pipeline over every recording of a dataset.
    Run {
        /// Pipeline spec (TOML).
        pipeline: PathBuf,
        /// Dataset spec (TOML) describing where recordings are.
        #[arg(long)]
        dataset: PathBuf,
        /// Results folder. Unchanged steps from earlier runs are reused.
        #[arg(long)]
        out: PathBuf,
        /// Run only part of the recordings, as `i/n` (1-based), for example one Slurm array task.
        #[arg(long, value_parser = parse_shard)]
        shard: Option<(usize, usize)>,
        /// Recordings processed at once. 0 uses all cores.
        #[arg(long, default_value_t = 0)]
        workers: usize,
    },
    /// List the recordings a dataset spec finds.
    Inspect {
        dataset: PathBuf,
        /// Also load the first recording and show its shape and sampling rate.
        #[arg(long)]
        load: bool,
    },
    /// List available operations with their inputs, outputs and summaries.
    Ops {
        /// Print every operation with its parameter JSON Schema, for tools and agents.
        #[arg(long, conflicts_with = "markdown")]
        json: bool,
        /// Print the reference page (docs/reference/operations.md).
        #[arg(long)]
        markdown: bool,
    },
    /// Write saved results as one HDF5 file per recording.
    Export {
        /// Results folder written by `modewise run`.
        results: PathBuf,
        /// Pipeline name. Defaults to every pipeline in the folder.
        #[arg(long)]
        pipeline: Option<String>,
        /// Target folder.
        #[arg(long)]
        to: PathBuf,
    },
}

fn parse_shard(text: &str) -> Result<(usize, usize), String> {
    let (i, n) = text
        .split_once('/')
        .ok_or("expected i/n, for example 2/8")?;
    let (i, n): (usize, usize) = (
        i.parse().map_err(|_| "i is not a number")?,
        n.parse().map_err(|_| "n is not a number")?,
    );
    if n == 0 || i == 0 || i > n {
        return Err(format!("need 1 <= i <= n, got {i}/{n}"));
    }
    Ok((i, n))
}

/// Runs the CLI with process-style arguments (the first is the program name). Returns the exit code.
pub fn main(args: impl IntoIterator<Item = impl Into<OsString> + Clone>) -> u8 {
    let cli = match Cli::try_parse_from(args) {
        Ok(cli) => cli,
        Err(e) => {
            let _ = e.print();
            return if e.use_stderr() { ERROR } else { OK };
        }
    };
    let outcome = match cli.command {
        Command::Run {
            pipeline,
            dataset,
            out,
            shard,
            workers,
        } => commands::run(&pipeline, &dataset, &out, shard, workers),
        Command::Inspect { dataset, load } => commands::inspect(&dataset, load).map(|_| OK),
        Command::Ops { json, markdown } => {
            commands::ops(json, markdown);
            Ok(OK)
        }
        Command::Export {
            results,
            pipeline,
            to,
        } => commands::export(&results, pipeline.as_deref(), &to).map(|_| OK),
    };
    outcome.unwrap_or_else(|e| {
        eprintln!("error: {e:#}");
        ERROR
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_must_be_one_based_and_within_count() {
        assert_eq!(parse_shard("1/1"), Ok((1, 1)));
        assert_eq!(parse_shard("3/8"), Ok((3, 8)));
        for bad in ["0/2", "2/0", "3/2", "1", "a/2", "1/b", "0/0"] {
            assert!(parse_shard(bad).is_err(), "accepted {bad}");
        }
    }
}
