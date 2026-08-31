//! Production entry point for the standalone SGLang-Omni Rust router.

use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{CommandFactory, Parser};
use sgl_omni_router::{RunOutcome, execute};

#[derive(Debug, Parser)]
#[command(
    name = "sgl-omni-router",
    version,
    about = "Standalone SGLang-Omni Rust router process"
)]
struct Cli {
    /// Path to the one strict TOML configuration file.
    #[arg(long, value_name = "PATH", required = true)]
    config: PathBuf,
    /// Validate configuration without creating a runtime or listener.
    #[arg(long)]
    check_config: bool,
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error) => {
            let code = if error.use_stderr() { 2 } else { 0 };
            let _write_result = error.print();
            return ExitCode::from(code);
        }
    };

    match execute(&cli.config, cli.check_config) {
        Ok(RunOutcome::ConfigValid) => {
            let _write_result = writeln!(io::stdout().lock(), "configuration valid");
            ExitCode::SUCCESS
        }
        Ok(RunOutcome::CleanShutdown) => ExitCode::SUCCESS,
        Err(error) => {
            let exit_code = error.exit_code();
            let mut command = Cli::command();
            let _write_result = command.error(clap::error::ErrorKind::Io, error).print();
            ExitCode::from(exit_code)
        }
    }
}
