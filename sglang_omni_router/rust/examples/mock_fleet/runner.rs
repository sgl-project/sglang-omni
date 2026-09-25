use std::collections::{BTreeMap, HashSet};
use std::time::{Duration, Instant};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use futures_util::{SinkExt, StreamExt, stream};
use serde::Serialize;
use serde_json::{Value, json};
use tokio_tungstenite::tungstenite::Message;

use crate::config::{Fleet, Result};
use crate::worker::{contains, supports, wav};

#[derive(Clone)]
struct Probe {
    name: String,
    service: String,
    path: String,
    body: Value,
    eligible: Vec<usize>,
}

#[derive(Serialize)]
struct Sample {
    ok: bool,
    status: Option<u16>,
    worker: String,
    bytes: usize,
    elapsed_ms: f64,
    first_body_ms: Option<f64>,
    error: Option<String>,
}

fn percentile(sorted: &[f64], percent: usize) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    sorted
        .get((sorted.len() * percent).div_ceil(100).saturating_sub(1))
        .copied()
}

fn percentiles(mut values: Vec<f64>) -> Value {
    values.sort_by(f64::total_cmp);
    json!({"p50":percentile(&values,50),"p95":percentile(&values,95),"p99":percentile(&values,99)})
}

fn complete(content_type: &str, tail: &[u8]) -> bool {
    !tail.is_empty()
        && (!content_type.starts_with("text/event-stream")
            || tail.windows(12).any(|window| window == b"data: [DONE]"))
}

fn strings(profile: &Value, field: &str) -> Vec<String> {
    profile[field]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

fn route_domain<'a>(fleet: &'a Fleet, profile: &Value) -> Option<&'a str> {
    let service = profile["service"].as_str()?;
    let section = match service {
        "generation_http" => fleet.document.get("http_generation")?,
        "speech_websocket" => fleet.document.get("websocket")?.get("speech")?,
        "realtime_websocket" => fleet.document.get("websocket")?.get("realtime")?,
        _ => {
            let section = fleet.document.get("http_media")?;
            let route = match service {
                "speech_http" => "speech",
                "speech_batch" => "speech_batch",
                "transcription_http" if profile["task"] == "translate" => "translation",
                "transcription_http" => "transcription",
                _ => return None,
            };
            if !section
                .get("routes")?
                .as_array()?
                .iter()
                .any(|value| value.as_str() == Some(route))
            {
                return None;
            }
            section
        }
    };
    section.get("trust_domain")?.as_str()
}

fn oversized_limit(fleet: &Fleet) -> Option<usize> {
    let section = fleet.document.get("http_generation")?;
    let buffered = section
        .get("buffered_request_max_bytes")
        .and_then(toml::Value::as_integer)
        .unwrap_or(8 * 1024 * 1024);
    let streamed = section
        .get("streamed_request_max_bytes")
        .and_then(toml::Value::as_integer)
        .unwrap_or(512 * 1024 * 1024);
    let limit = buffered.max(streamed);
    (1..=2 * 1024 * 1024)
        .contains(&limit)
        .then_some(limit as usize)
}

fn probes(fleet: &Fleet) -> Result<Vec<Probe>> {
    let mut probes = Vec::new();
    let mut seen = HashSet::new();
    for worker in &fleet.workers {
        for profile in &worker.profiles {
            let Some(domain) =
                route_domain(fleet, profile).filter(|domain| *domain == worker.trust_domain)
            else {
                continue;
            };
            let service = profile["service"]
                .as_str()
                .ok_or("profile service missing")?;
            let models = if service == "realtime_websocket" {
                vec![worker.default_model.clone()]
            } else {
                strings(profile, "model_ids")
            };
            let streams = if service == "speech_batch" || service == "realtime_websocket" {
                vec!["non_streaming".to_owned()]
            } else {
                strings(profile, "stream_modes")
            };
            for model in models {
                for mode in &streams {
                    let streaming = mode == "streaming";
                    let mut body = json!({"model":model,"stream":streaming});
                    let (path, variants) = match service {
                        "generation_http" => {
                            let inputs = strings(profile, "input_modalities");
                            let mut variants = Vec::new();
                            for input in &inputs {
                                let mut request = body.clone();
                                if input == "text"
                                    && contains(profile, "message_content_forms", "string")
                                {
                                    request["messages"] =
                                        json!([{"role":"user","content":"mock request"}]);
                                } else if contains(profile, "message_content_forms", "typed_parts")
                                    && (input == "text"
                                        || contains(profile, "media_placements", "typed_parts"))
                                {
                                    let part = match input.as_str() {
                                        "image" => {
                                            json!({"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}})
                                        }
                                        "audio" => {
                                            json!({"type":"input_audio","input_audio":{"data":"AAA=","format":"wav"}})
                                        }
                                        "video" => {
                                            json!({"type":"video_url","video_url":{"url":"data:video/mp4;base64,AA=="}})
                                        }
                                        _ => json!({"type":"text","text":"mock request"}),
                                    };
                                    request["messages"] = json!([{"role":"user","content":[part]}]);
                                } else if contains(profile, "media_placements", "top_level") {
                                    request[format!("{input}s")] = json!(["mock bytes"]);
                                    request["messages"] = json!([]);
                                } else {
                                    return Err(format!(
                                        "cannot generate {input} request for {}",
                                        worker.id
                                    )
                                    .into());
                                }
                                for output in strings(profile, "output_modalities") {
                                    request["modalities"] = json!([output]);
                                    if output == "audio" {
                                        for format in strings(profile, "chat_audio_formats") {
                                            request["audio"] = json!({"format":format});
                                            variants.push(request.clone());
                                        }
                                    } else {
                                        request
                                            .as_object_mut()
                                            .map(|object| object.remove("audio"));
                                        variants.push(request.clone());
                                    }
                                }
                            }
                            ("/v1/chat/completions", variants)
                        }
                        "speech_http" | "speech_batch" | "speech_websocket" => {
                            body["input"] = json!("mock speech");
                            if service == "speech_websocket" {
                                body["type"] = json!("session.config");
                                body["stream_audio"] = json!(streaming);
                                body.as_object_mut().map(|object| object.remove("stream"));
                            }
                            if let Some(task) = strings(profile, "tasks").first() {
                                body["task_type"] = json!(match task.as_str() {
                                    "voice_clone" => "Base",
                                    "voice_design" => "VoiceDesign",
                                    _ => "CustomVoice",
                                });
                            }
                            if !contains(profile, "reference_forms", "none") {
                                if contains(profile, "reference_forms", "direct") {
                                    body["ref_audio"] = json!("data:audio/wav;base64,AAA=");
                                } else if contains(profile, "reference_forms", "list") {
                                    body["references"] = json!([{"audio":"AAA=","text":"mock"}]);
                                } else {
                                    body["references"] = json!([{"vq_codes":[[1]],"text":"mock"}]);
                                }
                            }
                            if service == "speech_batch" {
                                let count = profile["max_batch_size"].as_u64().unwrap_or(0).min(2);
                                body["items"] = json!(
                                    (0..count)
                                        .map(|index| json!({"input":format!("item {index}")}))
                                        .collect::<Vec<_>>()
                                );
                                body.as_object_mut().map(|object| object.remove("stream"));
                            }
                            let variants = strings(profile, "response_formats")
                                .into_iter()
                                .map(|format| {
                                    let mut body = body.clone();
                                    body["response_format"] = json!(format);
                                    body
                                })
                                .collect();
                            (
                                match service {
                                    "speech_batch" => "/v1/audio/speech/batch",
                                    "speech_websocket" => "/v1/audio/speech/stream",
                                    _ => "/v1/audio/speech",
                                },
                                variants,
                            )
                        }
                        "transcription_http" => {
                            body["task"] = profile["task"].clone();
                            let variants = strings(profile, "response_formats")
                                .into_iter()
                                .map(|format| {
                                    let mut body = body.clone();
                                    body["response_format"] = json!(format);
                                    body
                                })
                                .collect();
                            (
                                if profile["task"] == "translate" {
                                    "/v1/audio/translations"
                                } else {
                                    "/v1/audio/transcriptions"
                                },
                                variants,
                            )
                        }
                        "realtime_websocket" => {
                            if model.is_empty() {
                                body.as_object_mut().map(|object| object.remove("model"));
                            }
                            ("/v1/realtime", vec![body])
                        }
                        _ => return Err(format!("unimplemented probe service {service}").into()),
                    };
                    for body in variants {
                        if !supports(worker, service, &body) {
                            return Err(format!(
                                "generated probe incompatible with {}: {body}",
                                worker.id
                            )
                            .into());
                        }
                        let key = format!("{path}:{body}");
                        if !seen.insert(key) {
                            continue;
                        }
                        let eligible = fleet
                            .workers
                            .iter()
                            .enumerate()
                            .filter_map(|(index, worker)| {
                                (worker.trust_domain == domain && supports(worker, service, &body))
                                    .then_some(index)
                            })
                            .collect();
                        probes.push(Probe {
                            name: format!("{service}/{}", probes.len()),
                            service: service.to_owned(),
                            path: path.to_owned(),
                            body,
                            eligible,
                        });
                    }
                }
            }
        }
    }
    if probes.is_empty() || probes.len() > 2048 {
        return Err("expected 1..=2048 generated workload variants".into());
    }
    Ok(probes)
}

async fn send(
    client: &reqwest::Client,
    base: &str,
    probe: &Probe,
    request_id: &str,
) -> Result<reqwest::Response> {
    let request = client
        .post(format!("{}{}", base.trim_end_matches('/'), probe.path))
        .header("x-request-id", request_id);
    Ok(if probe.service == "transcription_http" {
        let mut form = reqwest::multipart::Form::new().part(
            "file",
            reqwest::multipart::Part::bytes(wav(&[0; 128]))
                .file_name("mock.wav")
                .mime_str("audio/wav")?,
        );
        for field in ["model", "response_format", "stream"] {
            if let Some(value) = probe.body.get(field) {
                form = form.text(
                    field,
                    value
                        .as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| value.to_string()),
                );
            }
        }
        request.multipart(form).send().await?
    } else {
        request.json(&probe.body).send().await?
    })
}

async fn measure(
    client: &reqwest::Client,
    base: &str,
    probe: &Probe,
    fleet: &Fleet,
    sequence: usize,
    routed: bool,
) -> Sample {
    let started = Instant::now();
    let mut sample = Sample {
        ok: false,
        status: None,
        worker: String::new(),
        bytes: 0,
        elapsed_ms: 0.0,
        first_body_ms: None,
        error: None,
    };
    let result: Result<()> = async {
        let request_id = format!("mock-{}-{sequence}", probe.name.replace('/', "-"));
        let mut response = send(client, base, probe, &request_id).await?;
        sample.status = Some(response.status().as_u16());
        sample.worker = response
            .headers()
            .get("x-mock-worker")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        let echoed = response
            .headers()
            .get("x-mock-request-id")
            .and_then(|value| value.to_str().ok())
            == Some(request_id.as_str());
        let returned = !routed
            || response
                .headers()
                .get("x-request-id")
                .and_then(|value| value.to_str().ok())
                == Some(request_id.as_str());
        let mut tail = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            if chunk.is_empty() {
                continue;
            }
            sample
                .first_body_ms
                .get_or_insert_with(|| started.elapsed().as_secs_f64() * 1000.0);
            sample.bytes += chunk.len();
            if sample.bytes > 128 * 1024 * 1024 {
                return Err("mock response exceeds 128 MiB measurement bound".into());
            }
            tail.extend_from_slice(&chunk);
            if tail.len() > 4096 {
                tail.drain(..tail.len() - 4096);
            }
        }
        if sample.status != Some(200) {
            return Err(format!(
                "unexpected HTTP status {:?}: {}",
                sample.status,
                String::from_utf8_lossy(&tail)
            )
            .into());
        }
        if !probe
            .eligible
            .iter()
            .any(|index| fleet.workers[*index].id == sample.worker)
        {
            return Err("response came from an ineligible or unidentified worker".into());
        }
        if !echoed || !returned {
            return Err("request ID not preserved upstream or returned downstream".into());
        }
        if !complete(&content_type, &tail) {
            return Err("empty or incomplete streaming response".into());
        }
        Ok(())
    }
    .await;
    sample.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    match result {
        Ok(()) => sample.ok = true,
        Err(error) => sample.error = Some(error.to_string()),
    }
    sample
}

async fn load(
    client: &reqwest::Client,
    fleet: &Fleet,
    urls: &[String],
    router: &str,
    probe: &Probe,
    counts: (usize, usize),
    direct: bool,
) -> Value {
    let (requests, concurrency) = counts;
    let started = Instant::now();
    let samples: Vec<Sample> = stream::iter(0..requests)
        .map(|sequence| {
            let base = if direct {
                &urls[probe.eligible[sequence % probe.eligible.len()]]
            } else {
                router
            };
            measure(client, base, probe, fleet, sequence, !direct)
        })
        .buffer_unordered(concurrency)
        .collect()
        .await;
    let seconds = started.elapsed().as_secs_f64();
    let successes = samples.iter().filter(|sample| sample.ok).count();
    let bytes: usize = samples
        .iter()
        .filter(|sample| sample.ok)
        .map(|sample| sample.bytes)
        .sum();
    let mut distribution = BTreeMap::new();
    for sample in &samples {
        *distribution.entry(sample.worker.clone()).or_insert(0_usize) += 1;
    }
    json!({"requests":requests,"successes":successes,"failures":requests-successes,"wall_seconds":seconds,
        "successful_requests_per_second":successes as f64/seconds,"successful_bytes_per_second":bytes as f64/seconds,
        "latency_ms":percentiles(samples.iter().filter(|sample|sample.ok).map(|sample|sample.elapsed_ms).collect()),
        "first_body_byte_ms":percentiles(samples.iter().filter(|sample|sample.ok).filter_map(|sample|sample.first_body_ms).collect()),
        "worker_distribution":distribution,"errors":samples.iter().filter(|sample|!sample.ok).take(10).collect::<Vec<_>>()})
}

async fn websocket_probe(base: &str, probe: &Probe, fleet: &Fleet) -> Result<Value> {
    let mut url = reqwest::Url::parse(&format!(
        "{}{}",
        base.trim_end_matches('/').replacen("http://", "ws://", 1),
        probe.path
    ))?;
    if probe.service == "realtime_websocket"
        && let Some(model) = probe.body["model"].as_str()
    {
        url.query_pairs_mut().append_pair("model", model);
    }
    let (mut socket, _) = tokio_tungstenite::connect_async(url.as_str()).await?;
    if probe.service == "speech_websocket" {
        socket
            .send(Message::Text(probe.body.to_string().into()))
            .await?;
    }
    let first = socket
        .next()
        .await
        .ok_or("missing websocket setup event")??;
    let event: Value = serde_json::from_str(first.to_text()?)
        .map_err(|error| format!("invalid setup frame {first:?}: {error}"))?;
    let expected = if probe.service == "speech_websocket" {
        "session.configured"
    } else {
        "session.created"
    };
    if event["type"] != expected {
        return Err(format!("unexpected websocket setup: {event}").into());
    }
    let worker = event["mock"]["worker_id"]
        .as_str()
        .ok_or("missing session identity")?
        .to_owned();
    if !probe
        .eligible
        .iter()
        .any(|index| fleet.workers[*index].id == worker)
    {
        return Err("ineligible websocket worker".into());
    }
    socket.send(Message::Text(json!({"type":if probe.service=="speech_websocket" {"input.text"} else {"response.create"},"text":"mock"}).to_string().into())).await?;
    let mut audio_bytes = 0;
    let mut prefix: Vec<u8> = Vec::new();
    let mut done = false;
    while let Some(message) = socket.next().await {
        match message? {
            Message::Binary(bytes) => {
                audio_bytes += bytes.len();
                prefix.extend(bytes.iter().take(44_usize.saturating_sub(prefix.len())));
            }
            Message::Text(text) => {
                let event: Value = serde_json::from_str(&text)?;
                if event["mock"]["worker_id"] != worker {
                    return Err("websocket identity changed during session".into());
                }
                if event["type"] == "response.audio.delta" {
                    audio_bytes += STANDARD
                        .decode(event["delta"].as_str().ok_or("audio delta missing")?)?
                        .len();
                }
                if event["type"] == "audio.done" || event["type"] == "response.done" {
                    done = true;
                    break;
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }
    socket.close(None).await?;
    if !done || audio_bytes == 0 {
        return Err("websocket audio incomplete".into());
    }
    let selected = fleet
        .workers
        .iter()
        .find(|candidate| candidate.id == worker)
        .ok_or("unknown audio worker")?;
    let pcm_bytes = selected.behavior.chunk_bytes * selected.behavior.chunks;
    if probe.service == "speech_websocket" && probe.body["response_format"] == "wav" {
        let expected_header = wav(&vec![0; pcm_bytes]);
        if audio_bytes != pcm_bytes + 44 || prefix != expected_header[..44] {
            return Err("WAV header or length does not match negotiated format".into());
        }
    } else if audio_bytes != pcm_bytes {
        return Err("PCM byte count does not match configured payload".into());
    }
    Ok(json!({"worker":worker,"received_audio_payload_bytes":audio_bytes}))
}

fn check(name: &str, result: Result<Value>) -> Value {
    match result {
        Ok(details) => json!({"name":name,"passed":true,"details":details}),
        Err(error) => json!({"name":name,"passed":false,"error":error.to_string()}),
    }
}

async fn control(client: &reqwest::Client, url: &str, field: &str, status: u16) -> Result<()> {
    let mut body = json!({});
    body[field] = json!(status);
    client
        .post(format!("{}__mock/control", url))
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .bytes()
        .await?;
    Ok(())
}

async fn wait_ready(client: &reqwest::Client, router: &str, status: u16) -> Result<()> {
    let mut tick = tokio::time::interval(Duration::from_millis(50));
    tokio::time::timeout(Duration::from_secs(20), async {
        loop {
            let response = client.get(format!("{router}/ready")).send().await?;
            if response.status().as_u16() == status {
                return Ok::<_, Box<dyn std::error::Error + Send + Sync>>(());
            }
            tick.tick().await;
        }
    })
    .await??;
    Ok(())
}

async fn worker_total(client: &reqwest::Client, urls: &[String]) -> Result<u64> {
    let mut total = 0;
    for url in urls {
        let stats: Value = client
            .get(format!("{}__mock/stats", url))
            .send()
            .await?
            .error_for_status()?
            .json()
            .await?;
        total += stats["accepted"]
            .as_u64()
            .ok_or("accepted counter missing")?
            + stats["rejected"]
                .as_u64()
                .ok_or("rejected counter missing")?;
    }
    Ok(total)
}

async fn operational_checks(
    client: &reqwest::Client,
    fleet: &Fleet,
    urls: &[String],
    router: &str,
    probes: &[Probe],
) -> Vec<Value> {
    let mut checks = Vec::new();
    if let Some(probe) = probes
        .iter()
        .find(|probe| !probe.service.ends_with("websocket"))
    {
        let result: Result<Value> = async {
            for url in urls {
                control(client, url, "response_status", 503).await?;
            }
            let response = send(client, router, probe, "mock-injected-failure").await?;
            let status = response.status();
            let identity = response.headers().contains_key("x-mock-worker");
            response.bytes().await?;
            if status.as_u16() != 503 || !identity {
                return Err("upstream 503 was not relayed with mock identity".into());
            }
            Ok(json!({"status":503}))
        }
        .await;
        checks.push(check("upstream_503_relay", result));
        for (url, worker) in urls.iter().zip(&fleet.workers) {
            if let Err(error) = control(
                client,
                url,
                "response_status",
                worker.behavior.response_status,
            )
            .await
            {
                checks.push(check("restore_response_status", Err(error)));
            }
        }
        let result: Result<Value> = async {
            for url in urls {
                control(client, url, "health_status", 503).await?;
            }
            wait_ready(client, router, 503).await?;
            let before = worker_total(client, urls).await?;
            let response = send(client, router, probe, "mock-unhealthy").await?;
            let status = response.status();
            response.bytes().await?;
            let after = worker_total(client, urls).await?;
            if status.as_u16() != 503 || before != after {
                return Err("unhealthy request was dispatched or did not return 503".into());
            }
            Ok(json!({"status":503,"upstream_requests":after-before}))
        }
        .await;
        checks.push(check("all_workers_unhealthy", result));
        for (url, worker) in urls.iter().zip(&fleet.workers) {
            if let Err(error) =
                control(client, url, "health_status", worker.behavior.health_status).await
            {
                checks.push(check("restore_worker_health", Err(error)));
            }
        }
        checks.push(check(
            "readiness_recovers",
            wait_ready(client, router, 200)
                .await
                .map(|()| json!({"status":200})),
        ));
    }
    if let Some(probe) = probes
        .iter()
        .find(|probe| probe.service == "realtime_websocket")
    {
        let limit = fleet
            .document
            .get("admission")
            .and_then(|value| value.get("realtime_websocket"))
            .and_then(toml::Value::as_integer)
            .unwrap_or(0);
        if (1..=32).contains(&limit) {
            let result = tokio::time::timeout(Duration::from_secs(15), async {
                let mut sockets = Vec::new();
                let mut rejected = false;
                let mut url = reqwest::Url::parse(&format!(
                    "{}{}",
                    router.replacen("http://", "ws://", 1),
                    probe.path
                ))?;
                url.query_pairs_mut()
                    .append_pair("model", probe.body["model"].as_str().unwrap_or_default());
                for _attempt in 0..=limit {
                    match tokio_tungstenite::connect_async(url.as_str()).await {
                        Ok((mut socket, _)) => {
                            let event = socket
                                .next()
                                .await
                                .ok_or("missing saturation setup event")??;
                            let event: Value = serde_json::from_str(event.to_text()?)?;
                            if event["type"] != "session.created" {
                                return Err("unexpected saturation setup event".into());
                            }
                            sockets.push(socket);
                        }
                        Err(tokio_tungstenite::tungstenite::Error::Http(response))
                            if response.status().as_u16() == 429 =>
                        {
                            rejected = true;
                            break;
                        }
                        Err(error) => return Err(error.into()),
                    }
                }
                let admitted = sockets.len();
                for mut socket in sockets {
                    let _closed = socket.close(None).await;
                }
                if !rejected {
                    return Err(
                        "realtime admission did not return 429 at its configured limit".into(),
                    );
                }
                Ok::<_, Box<dyn std::error::Error + Send + Sync>>(
                    json!({"admitted_sessions":admitted,"status":429}),
                )
            })
            .await;
            checks.push(check(
                "realtime_session_saturation",
                match result {
                    Ok(result) => result,
                    Err(error) => Err(error.into()),
                },
            ));
        }
    }
    checks
}

pub(super) async fn run(
    fleet: &Fleet,
    urls: &[String],
    router: &str,
    requests: usize,
    concurrency: usize,
) -> Result<Value> {
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .timeout(Duration::from_secs(15))
        .build()?;
    let probes = probes(fleet)?;
    let mut skipped = Vec::new();
    if oversized_limit(fleet).is_none() {
        skipped.push("oversized generation: route disabled or maximum of streamed/buffered limits exceeds the 2 MiB test budget");
    }
    if !probes
        .iter()
        .any(|probe| probe.service == "realtime_websocket")
    {
        skipped.push("realtime session saturation: no routed realtime profile");
    } else if !fleet
        .document
        .get("admission")
        .and_then(|value| value.get("realtime_websocket"))
        .and_then(toml::Value::as_integer)
        .is_some_and(|limit| (1..=32).contains(&limit))
    {
        skipped.push(
            "realtime session saturation: requires explicit session admission limit of 1..=32",
        );
    }
    if !fleet.workers.iter().any(|worker| worker.voice_owner) {
        skipped.push("voice CRUD and named synthesis: no voice owner configured");
    }
    let mut workloads = Vec::new();
    let mut checks = Vec::new();
    for probe in &probes {
        eprintln!("checking {} {}", probe.name, probe.body);
        if probe.service.ends_with("websocket") {
            let result = tokio::time::timeout(
                Duration::from_secs(15),
                websocket_probe(router, probe, fleet),
            )
            .await;
            checks.push(check(
                &probe.name,
                match result {
                    Ok(result) => result,
                    Err(error) => Err(error.into()),
                },
            ));
            continue;
        }
        let direct = load(
            &client,
            fleet,
            urls,
            router,
            probe,
            (requests, concurrency),
            true,
        )
        .await;
        let routed = load(
            &client,
            fleet,
            urls,
            router,
            probe,
            (requests, concurrency),
            false,
        )
        .await;
        workloads.push(json!({"name":probe.name,"service":probe.service,"request":probe.body,"eligible_workers":probe.eligible.iter().map(|index|&fleet.workers[*index].id).collect::<Vec<_>>(),"direct":direct,"router":routed}));
        let mut bad = probe.clone();
        bad.body["model"] = json!("__unsupported_mock_model__");
        for (name, base) in [
            ("direct_model_rejection", urls[probe.eligible[0]].as_str()),
            ("routed_model_rejection", router),
        ] {
            let result: Result<Value> = async {
                let response = send(&client, base, &bad, "mock-negative").await?;
                let status = response.status();
                let _body = response.bytes().await?;
                if !status.is_client_error() || status.as_u16() == 429 {
                    return Err(format!("expected capability rejection, received {status}").into());
                }
                Ok(json!({"status":status.as_u16()}))
            }
            .await;
            checks.push(check(&format!("{}/{name}", probe.name), result));
        }
    }
    if let Some(probe) = probes
        .iter()
        .find(|probe| probe.service == "generation_http")
    {
        let mut incompatible = probe.clone();
        incompatible.body["images"] = json!(["mock image"]);
        if !fleet
            .workers
            .iter()
            .any(|worker| supports(worker, "generation_http", &incompatible.body))
        {
            for (label, base) in [
                ("direct", urls[probe.eligible[0]].as_str()),
                ("router", router),
            ] {
                let result: Result<Value> = async {
                    let response =
                        send(&client, base, &incompatible, "mock-wrong-modality").await?;
                    let status = response.status();
                    response.bytes().await?;
                    if !status.is_client_error() || status.as_u16() == 429 {
                        return Err(
                            format!("unsupported modality was not rejected: {status}").into()
                        );
                    }
                    Ok(json!({"status":status.as_u16()}))
                }
                .await;
                checks.push(check(&format!("{label}_unsupported_modality"), result));
            }
        }
        if let Some(limit) = oversized_limit(fleet) {
            let mut oversized = probe.clone();
            oversized.body["messages"] = json!([{"role":"user","content":"x".repeat(limit+1)}]);
            let result: Result<Value> = async {
                let before = worker_total(&client, urls).await?;
                let response = send(&client, router, &oversized, "mock-oversized").await?;
                let status = response.status();
                let _body = response.bytes().await?;
                if status.as_u16() != 413 {
                    return Err(format!("expected complete 413 response, got {status}").into());
                }
                let after = worker_total(&client, urls).await?;
                if before != after {
                    return Err("oversized request reached a mock worker".into());
                }
                Ok(json!({"status":413,"upstream_requests":after-before}))
            }
            .await;
            checks.push(check("oversized_generation_response", result));
        }
    }
    for worker in fleet.workers.iter().filter(|worker| worker.voice_owner) {
        let result: Result<Value> = async {
            let endpoint = format!("{router}/v1/audio/voices");
            let response = client
                .post(&endpoint)
                .multipart(
                    reqwest::multipart::Form::new()
                        .text("name", "mock-test-voice")
                        .part(
                            "file",
                            reqwest::multipart::Part::bytes(wav(&[0; 128])).file_name("voice.wav"),
                        ),
                )
                .send()
                .await?
                .error_for_status()?;
            if response
                .headers()
                .get("x-mock-worker")
                .and_then(|value| value.to_str().ok())
                != Some(worker.id.as_str())
            {
                return Err("voice mutation went to wrong owner".into());
            }
            let _body = response.bytes().await?;
            let listed: Value = client
                .get(format!("{endpoint}?names_only=true"))
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;
            if !listed["voices"]
                .as_array()
                .is_some_and(|voices| voices.iter().any(|voice| voice == "mock-test-voice"))
            {
                return Err("created voice not listed".into());
            }
            if let Some(probe) = probes.iter().find(|probe| {
                probe.service == "speech_http" && supports(worker, "speech_http", &probe.body)
            }) {
                let mut named = probe.clone();
                named.body["voice"] = json!("mock-test-voice");
                let response = send(&client, router, &named, "mock-named-voice")
                    .await?
                    .error_for_status()?;
                if response
                    .headers()
                    .get("x-mock-worker")
                    .and_then(|value| value.to_str().ok())
                    != Some(worker.id.as_str())
                {
                    return Err("named voice synthesis went to wrong owner".into());
                }
                response.bytes().await?;
            }
            let _deleted = client
                .delete(format!("{endpoint}/mock-test-voice"))
                .send()
                .await?
                .error_for_status()?
                .bytes()
                .await?;
            Ok(json!({"owner":worker.id}))
        }
        .await;
        checks.push(check("voice_owner_crud", result));
    }
    checks.extend(operational_checks(&client, fleet, urls, router, &probes).await);
    let mut stats = Vec::new();
    for url in urls {
        stats.push(
            client
                .get(format!("{}__mock/stats", url))
                .send()
                .await?
                .error_for_status()?
                .json::<Value>()
                .await?,
        );
    }
    let metrics = client
        .get(format!("{router}/metrics"))
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    let diagnostics: Value = client
        .get(format!("{router}/diagnostics"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let passed = checks.iter().all(|check| check["passed"] == true)
        && workloads.iter().all(|workload| {
            workload["direct"]["failures"] == 0 && workload["router"]["failures"] == 0
        });
    Ok(
        json!({"schema_version":1,"passed":passed,"requests_per_workload_per_path":requests,"concurrency":concurrency,
        "checks":checks,"skipped":skipped,"workloads":workloads,"worker_stats":stats,"router_diagnostics":diagnostics,"router_metrics":metrics,
        "unsupported_router_features":["CP snapshots and generation fencing","connect retries","circuit breakers","router-internal TTFP"],
        "not_tested":["real-model quality and GPU/MPS scaling","encoded audio other than PCM/WAV","dynamic membership","exhaustive protocol fuzzing","per-core CPU profiling","slow consumers and upload cancellation","restart-safe session affinity","WebSocket throughput","worker saturation instrumentation","priority and cache-aware scheduling","global and HTTP-class overload bounds","graceful router drain with active streams","raw malformed/duplicate-header and rejected-media connection-reset regressions","multipart streaming memory usage","voice-owner failure during mutation","route-hint conflicts","frame-size boundary enforcement"],
        "measurement_notes":["Synthetic closed-loop workload; first_body_byte_ms is client-observed, not router-internal audio TTFP.","Direct baseline distributes over the same eligible workers; routed workload uses configured policy.","Metrics include direct and routed requests in worker stats; router metrics include routed requests only.","No CPU affinity is applied; these numbers do not establish a per-core Rust/Python speedup."]}),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn oversize_probe_exceeds_both_direct_and_classified_limits() -> Result<()> {
        let mut fleet = Fleet::parse(include_str!("fleet.toml"))?;
        fleet.document["http_generation"]["streamed_request_max_bytes"] =
            toml::Value::Integer(1024 * 1024);
        assert_eq!(oversized_limit(&fleet), Some(1024 * 1024));
        fleet.document["http_generation"]["streamed_request_max_bytes"] =
            toml::Value::Integer(4 * 1024 * 1024);
        assert_eq!(oversized_limit(&fleet), None);
        Ok(())
    }

    #[test]
    fn single_item_batch_and_route_scopes_are_respected() -> Result<()> {
        let mut document: toml::Value = toml::from_str(include_str!("fleet.toml"))?;
        document["workers"][4]["service_profiles"][2]["max_batch_size"] = toml::Value::Integer(1);
        let fleet = Fleet::parse(&toml::to_string(&document)?)?;
        assert!(
            probes(&fleet)?
                .iter()
                .filter(|probe| probe.service == "speech_batch")
                .all(|probe| probe.body["items"]
                    .as_array()
                    .is_some_and(|items| items.len() == 1))
        );
        document["workers"][0]["trust_domain"] = toml::Value::String("other".to_owned());
        let fleet = Fleet::parse(&toml::to_string(&document)?)?;
        assert!(
            probes(&fleet)?
                .iter()
                .all(|probe| !probe.eligible.contains(&0))
        );
        document
            .as_table_mut()
            .ok_or("table")?
            .remove("http_generation");
        let fleet = Fleet::parse(&toml::to_string(&document)?)?;
        assert!(
            probes(&fleet)?
                .iter()
                .all(|probe| probe.service != "generation_http")
        );
        Ok(())
    }

    #[tokio::test]
    async fn realtime_counts_decoded_audio_bytes() -> Result<()> {
        let fleet = Fleet::parse(include_str!("fleet.toml"))?;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let address = listener.local_addr()?;
        let worker = fleet.workers[3].clone();
        let expected = worker.behavior.chunks * worker.behavior.chunk_bytes;
        let task =
            tokio::spawn(async move { axum::serve(listener, crate::worker::app(worker)).await });
        let cases = probes(&fleet)?;
        let probe = cases
            .iter()
            .find(|probe| probe.service == "realtime_websocket")
            .ok_or("realtime missing")?;
        let result = websocket_probe(&format!("http://{address}"), probe, &fleet).await;
        task.abort();
        let _stopped = task.await;
        assert_eq!(result?["received_audio_payload_bytes"], expected);
        Ok(())
    }

    #[test]
    fn example_probes_preserve_websocket_stream_field_and_shared_model_routing() -> Result<()> {
        let fleet = Fleet::parse(include_str!("fleet.toml"))?;
        let cases = probes(&fleet)?;
        let speech = cases
            .iter()
            .find(|probe| probe.service == "speech_websocket")
            .ok_or("speech probe missing")?;
        assert_eq!(speech.body["stream_audio"], true);
        assert!(speech.body.get("stream").is_none());
        assert!(
            cases
                .iter()
                .any(|probe| probe.body["model"] == "shared" && probe.eligible.len() > 1)
        );
        Ok(())
    }

    use super::*;

    #[test]
    fn percentile_uses_nearest_rank_without_panicking_on_empty_samples() {
        assert_eq!(percentile(&[], 99), None);
        assert_eq!(percentile(&[1.0, 2.0, 9.0, 10.0], 50), Some(2.0));
        assert_eq!(percentile(&[1.0, 2.0, 9.0, 10.0], 99), Some(10.0));
    }

    #[test]
    fn streaming_completion_requires_a_done_marker() {
        assert!(complete(
            "text/event-stream",
            b"data: {}\n\ndata: [DONE]\n\n"
        ));
        assert!(!complete("text/event-stream", b"data: {}\n\n"));
        assert!(complete("audio/pcm", &[0, 0]));
    }
}
