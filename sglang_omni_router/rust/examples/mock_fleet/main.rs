mod config;
mod runner;
mod worker;

use std::io::Write;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use clap::{Parser, ValueEnum};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::{Child, Command};

use config::{Fleet, Result};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mode {
    Run,
    Serve,
}

#[derive(Parser)]
#[command(about = "Model-free modality workers and verification/load harness for the Rust router")]
struct Options {
    #[arg(long)]
    config: PathBuf,
    #[arg(long, default_value = "target/debug/sgl-omni-router")]
    router_bin: PathBuf,
    #[arg(long, value_enum, default_value = "run")]
    mode: Mode,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long, default_value_t = 8, value_parser = clap::value_parser!(u32).range(1..=10000))]
    requests: u32,
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=256))]
    concurrency: u32,
    #[arg(long, default_value_t = 300, value_parser = clap::value_parser!(u64).range(1..=86400))]
    deadline_secs: u64,
    #[arg(long, hide = true)]
    worker: Option<usize>,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<()> {
    let options = Options::parse();
    let fleet = Fleet::parse(&std::fs::read_to_string(&options.config)?)?;
    if let Some(index) = options.worker {
        let worker = fleet
            .workers
            .get(index)
            .ok_or("worker ordinal out of range")?
            .clone();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        println!(
            "{}",
            json!({"worker":worker.id,"url":format!("http://{}/",listener.local_addr()?)})
        );
        std::io::stdout().flush()?;
        axum::serve(listener,worker::app(worker)).with_graceful_shutdown(async {
			let mut input = tokio::io::stdin();
			let mut byte = [0];
			tokio::select! { _signal = shutdown_signal() => {}, _eof = input.read(&mut byte) => {} }
		}).await?;
        return Ok(());
    }
    let output = options.output.clone().unwrap_or_else(|| {
        PathBuf::from("target").join(format!(
            "mock-fleet-{}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            std::process::id()
        ))
    });
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::create_dir(&output)?;
    let output = std::fs::canonicalize(output)?;
    let mut children: Vec<Child> = Vec::new();
    let mut result = tokio::select! {
        result = supervise(&options,&fleet,&output,&mut children) => result,
        signal = shutdown_signal() => signal.and_then(|()|match options.mode { Mode::Serve=>Ok(()),Mode::Run=>Err("interrupted; owned processes stopped".into()) }),
    };
    for child in children.iter_mut().rev() {
        if child.try_wait()?.is_none() {
            let _kill = child.start_kill();
            if let Err(error) = child.wait().await {
                result = Err(error.into());
            }
        }
    }
    if let Err(error) = &result
        && !output.join("report.json").exists()
    {
        std::fs::write(
            output.join("report.json"),
            serde_json::to_vec_pretty(&json!({"passed":false,"error":error.to_string()}))?,
        )?;
    }
    eprintln!("Artifacts: {}", output.display());
    result
}

async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! { result = tokio::signal::ctrl_c() => { result?; }, _signal = terminate.recv() => {} }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await?;
    Ok(())
}

async fn supervise(
    options: &Options,
    fleet: &Fleet,
    output: &std::path::Path,
    children: &mut Vec<Child>,
) -> Result<()> {
    let router_binary = std::fs::canonicalize(&options.router_bin)?;
    let config_path = std::fs::canonicalize(&options.config)?;
    let executable = std::env::current_exe()?;
    let _available = std::net::TcpListener::bind(fleet.listen)?;
    let mut urls = Vec::new();
    for (index, worker) in fleet.workers.iter().enumerate() {
        let log = std::fs::File::create(output.join(format!("{}.log", worker.id)))?;
        let mut child = Command::new(&executable)
            .arg("--config")
            .arg(&config_path)
            .arg("--worker")
            .arg(index.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(log)
            .kill_on_drop(true)
            .spawn()?;
        let stdout = child.stdout.take().ok_or("worker readiness pipe missing")?;
        children.push(child);
        let mut line = String::new();
        tokio::time::timeout(
            Duration::from_secs(10),
            BufReader::new(stdout).read_line(&mut line),
        )
        .await??;
        let announced: Value = serde_json::from_str(&line)?;
        if announced["worker"] != worker.id {
            return Err("worker readiness identity mismatch".into());
        }
        let url = announced["url"]
            .as_str()
            .ok_or("worker URL missing")?
            .to_owned();
        eprintln!("{}: {url}", worker.id);
        urls.push(url);
    }
    let router_config = output.join("router.toml");
    std::fs::write(&router_config, fleet.router_config(&urls)?)?;
    let log = std::fs::File::create(output.join("config-check.log"))?;
    let mut checker = Command::new(&router_binary)
        .arg("--config")
        .arg(&router_config)
        .arg("--check-config")
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .kill_on_drop(true)
        .spawn()?;
    let checked = tokio::time::timeout(Duration::from_secs(10), checker.wait()).await;
    match checked {
        Ok(Ok(status)) if status.success() => {}
        _ => {
            let _kill = checker.start_kill();
            let _reaped = checker.wait().await;
            return Err("router rejected generated config; see config-check.log".into());
        }
    }
    drop(_available);
    let log = std::fs::File::create(output.join("router.log"))?;
    let router = Command::new(&router_binary)
        .arg("--config")
        .arg(&router_config)
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .kill_on_drop(true)
        .spawn()?;
    children.push(router);
    let router_url = format!("http://{}", fleet.listen);
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(1))
        .build()?;
    let mut ticks = tokio::time::interval(Duration::from_millis(100));
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            for child in children.iter_mut() {
                if let Some(code) = child.try_wait()? {
                    return Err(format!("child exited during startup: {code}; inspect logs").into());
                }
            }
            if client
                .get(format!("{router_url}/ready"))
                .send()
                .await
                .is_ok_and(|response| response.status().is_success())
            {
                return Ok::<_, Box<dyn std::error::Error + Send + Sync>>(());
            }
            ticks.tick().await;
        }
    })
    .await??;
    std::fs::write(
        output.join("endpoints.json"),
        serde_json::to_vec_pretty(
            &json!({"router":router_url,"workers":fleet.workers.iter().zip(&urls).map(|(worker,url)|json!({"id":worker.id,"url":url})).collect::<Vec<_>>()}),
        )?,
    )?;
    println!("Router ready: {router_url}");
    match options.mode {
        Mode::Serve => {
            shutdown_signal().await?;
            Ok(())
        }
        Mode::Run => {
            let report = tokio::time::timeout(
                Duration::from_secs(options.deadline_secs),
                runner::run(
                    fleet,
                    &urls,
                    &router_url,
                    options.requests as usize,
                    options.concurrency as usize,
                ),
            )
            .await??;
            std::fs::write(
                output.join("report.json"),
                serde_json::to_vec_pretty(&report)?,
            )?;
            println!(
                "passed={} workloads={} checks={}",
                report["passed"],
                report["workloads"].as_array().map_or(0, Vec::len),
                report["checks"].as_array().map_or(0, Vec::len)
            );
            if report["passed"] != true {
                return Err("mock-fleet checks failed; see report.json".into());
            }
            Ok(())
        }
    }
}
