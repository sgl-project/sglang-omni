use std::fmt;

use serde::de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};

use crate::error::HttpFault;
use crate::speech_facts::{
    BatchSpeechFields, SpeechFields, effective_reference_forms,
    named_voice as classify_named_voice, read_batch_field, read_field as read_speech_field,
    read_stream as read_speech_stream, reference_forms,
    response_format as classify_response_format, task as classify_task,
};
use crate::worker_pool::{
    ModelSelection, ProfileRequirement, ReferenceForm, RouteRequirement, SpeechResponseFormat,
    SpeechToTextTask, StreamMode, TranscriptionResponseFormat, TrustDomain,
};

use super::multipart;

pub(super) struct Classified {
    pub(super) requirement: RouteRequirement,
    pub(super) credits: u32,
}

#[cfg(test)]
pub(super) fn speech(
    bytes: &[u8],
    _pool: &crate::worker_pool::WorkerPool,
    trust: &TrustDomain,
) -> Result<Classified, HttpFault> {
    speech_with_hints(bytes, None, None, trust)
}

pub(super) fn speech_with_hints(
    bytes: &[u8],
    route_model: Option<&str>,
    route_stream: Option<bool>,
    trust: &TrustDomain,
) -> Result<Classified, HttpFault> {
    let fields = parse_speech(bytes)?;
    let model = model_selection(fields.model.clone().flatten(), route_model)?;
    let format = classify_response_format(
        fields
            .response_format
            .as_ref()
            .and_then(Option::as_deref)
            .unwrap_or("wav"),
    )
    .ok_or(HttpFault::MalformedRequest)?;
    let body_stream = fields.stream.as_ref().and_then(|value| *value);
    let stream = merge_stream(body_stream, route_stream)?;
    if stream && format != SpeechResponseFormat::Pcm {
        return Err(HttpFault::MalformedRequest);
    }
    let stream_mode = if stream {
        StreamMode::Streaming
    } else {
        StreamMode::NonStreaming
    };
    let task = fields
        .task
        .as_ref()
        .and_then(Option::as_deref)
        .map(|value| classify_task(value).ok_or(HttpFault::MalformedRequest))
        .transpose()?;
    let mut references = reference_forms(&fields);
    let named_voice = classify_named_voice(&fields, &references);
    if named_voice {
        references.clear();
    }
    Ok(Classified {
        requirement: RouteRequirement::new(
            ProfileRequirement::SpeechHttp {
                model,
                response_format: format,
                stream_mode,
                task,
                reference_forms: references,
                named_voice,
            },
            trust.clone(),
        ),
        credits: 1,
    })
}

#[cfg(test)]
pub(super) fn batch(
    bytes: &[u8],
    _pool: &crate::worker_pool::WorkerPool,
    trust: &TrustDomain,
) -> Result<Classified, HttpFault> {
    batch_with_hints(bytes, None, None, trust)
}

pub(super) fn batch_with_hints(
    bytes: &[u8],
    route_model: Option<&str>,
    route_stream: Option<bool>,
    trust: &TrustDomain,
) -> Result<Classified, HttpFault> {
    let (defaults, items) = parse_batch(bytes)?;
    if merge_stream(
        defaults.stream.as_ref().and_then(|value| *value),
        route_stream,
    )? {
        return Err(HttpFault::MalformedRequest);
    }
    if items.is_empty() || items.len() > usize::from(u16::MAX) {
        return Err(HttpFault::MalformedRequest);
    }
    let item_count = items.len();
    let mut models = Vec::with_capacity(item_count);
    let mut default_model = defaults.model.clone().flatten();
    let mut default_model_added = false;
    let mut tasks = Vec::new();
    let mut references = Vec::new();
    let default_format = classify_response_format(
        defaults
            .response_format
            .as_ref()
            .and_then(Option::as_deref)
            .unwrap_or("wav"),
    )
    .ok_or(HttpFault::MalformedRequest)?;
    let mut formats = vec![default_format];
    let default_task = defaults
        .task
        .as_ref()
        .and_then(Option::as_deref)
        .map(|value| classify_task(value).ok_or(HttpFault::MalformedRequest))
        .transpose()?;
    let default_references = reference_forms(&defaults);
    let mut named_voice = classify_named_voice(&defaults, &default_references);
    for item in items {
        if !item.routing_fields_valid() {
            continue;
        }
        let mut item = item.fields;
        if let Some(model) = item.model.take().flatten() {
            models.push(model_selection(Some(model), route_model)?);
        } else if !default_model_added {
            models.push(model_selection(default_model.take(), route_model)?);
            default_model_added = true;
        }
        match item.response_format.as_ref().and_then(Option::as_deref) {
            Some(value) => {
                if let Some(format) = classify_response_format(value) {
                    insert_once(&mut formats, format);
                }
            }
            None => insert_once(&mut formats, default_format),
        }
        match item.task.as_ref().and_then(Option::as_deref) {
            Some(value) => {
                if let Some(task) = classify_task(value) {
                    insert_once(&mut tasks, task);
                }
            }
            None => {
                if let Some(task) = default_task {
                    insert_once(&mut tasks, task);
                }
            }
        }
        let effective_references = effective_reference_forms(&defaults, &item);
        let explicit_reference = effective_references != [ReferenceForm::None];
        let voice = item
            .voice
            .as_ref()
            .and_then(Option::as_deref)
            .or_else(|| defaults.voice.as_ref().and_then(Option::as_deref));
        let item_named_voice = !explicit_reference
            && voice
                .is_some_and(|value| !value.is_empty() && !value.eq_ignore_ascii_case("default"));
        if item_named_voice {
            named_voice = true;
        } else {
            for form in effective_references {
                insert_once(&mut references, form);
            }
        }
    }
    models.sort_unstable();
    models.dedup();
    let batch_size = u16::try_from(item_count).map_err(|_| HttpFault::MalformedRequest)?;
    Ok(Classified {
        requirement: RouteRequirement::new(
            ProfileRequirement::SpeechBatch {
                models,
                response_formats: formats,
                tasks,
                reference_forms: references,
                named_voice,
                batch_size,
            },
            trust.clone(),
        ),
        credits: u32::from(batch_size),
    })
}

pub(super) fn transcription_with_hints(
    bytes: &[u8],
    boundary: &[u8],
    route_model: Option<&str>,
    route_stream: Option<bool>,
    trust: &TrustDomain,
) -> Result<Classified, HttpFault> {
    speech_to_text(
        bytes,
        boundary,
        route_model,
        route_stream,
        trust,
        SpeechToTextTask::Transcribe,
    )
}

pub(super) fn translation_with_hints(
    bytes: &[u8],
    boundary: &[u8],
    route_model: Option<&str>,
    route_stream: Option<bool>,
    trust: &TrustDomain,
) -> Result<Classified, HttpFault> {
    speech_to_text(
        bytes,
        boundary,
        route_model,
        route_stream,
        trust,
        SpeechToTextTask::Translate,
    )
}

fn speech_to_text(
    bytes: &[u8],
    boundary: &[u8],
    route_model: Option<&str>,
    route_stream: Option<bool>,
    trust: &TrustDomain,
    task: SpeechToTextTask,
) -> Result<Classified, HttpFault> {
    let facts = multipart::scan(bytes, boundary)?;
    let model = model_selection(facts.model, route_model)?;
    let stream = merge_stream(facts.stream, route_stream)?;
    let format = if stream {
        if !matches!(
            facts
                .response_format
                .as_deref()
                .unwrap_or("json")
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "json" | "text"
        ) {
            return Err(HttpFault::MalformedRequest);
        }
        TranscriptionResponseFormat::Sse
    } else {
        match facts
            .response_format
            .as_deref()
            .unwrap_or("json")
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "json" => TranscriptionResponseFormat::Json,
            "text" => TranscriptionResponseFormat::Text,
            "verbose_json" => TranscriptionResponseFormat::VerboseJson,
            "srt" => TranscriptionResponseFormat::Srt,
            "vtt" => TranscriptionResponseFormat::Vtt,
            _ => return Err(HttpFault::MalformedRequest),
        }
    };
    Ok(Classified {
        requirement: RouteRequirement::new(
            ProfileRequirement::TranscriptionHttp {
                model,
                task,
                response_format: format,
                stream_mode: if stream {
                    StreamMode::Streaming
                } else {
                    StreamMode::NonStreaming
                },
            },
            trust.clone(),
        ),
        credits: 1,
    })
}

fn model_selection(
    model: Option<String>,
    route_assertion: Option<&str>,
) -> Result<ModelSelection, HttpFault> {
    let model = model.filter(|value| !value.is_empty());
    match (model, route_assertion) {
        (Some(model), Some(asserted)) if model != asserted => Err(HttpFault::MalformedRequest),
        (Some(model), _) => Ok(ModelSelection::Explicit(model)),
        (None, Some(asserted)) => Ok(ModelSelection::WorkerDefault {
            expected_model_id: asserted.to_owned(),
        }),
        (None, None) => Ok(ModelSelection::UnresolvedDefault),
    }
}

fn merge_stream(body: Option<bool>, route: Option<bool>) -> Result<bool, HttpFault> {
    let effective = body.unwrap_or(false);
    if route.is_some_and(|asserted| asserted != effective) {
        Err(HttpFault::MalformedRequest)
    } else {
        Ok(effective)
    }
}

fn insert_once<T: Eq>(values: &mut Vec<T>, value: T) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn parse_speech(bytes: &[u8]) -> Result<SpeechFields, HttpFault> {
    parse(bytes, RootMode::Speech).map(|parsed| parsed.0)
}

fn parse_batch(bytes: &[u8]) -> Result<(SpeechFields, Vec<BatchSpeechFields>), HttpFault> {
    parse(bytes, RootMode::Batch)
}

fn parse(
    bytes: &[u8],
    mode: RootMode,
) -> Result<(SpeechFields, Vec<BatchSpeechFields>), HttpFault> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let parsed = RootSeed(mode)
        .deserialize(&mut deserializer)
        .map_err(|_| HttpFault::MalformedRequest)?;
    deserializer
        .end()
        .map_err(|_| HttpFault::MalformedRequest)?;
    Ok(parsed)
}

#[derive(Clone, Copy)]
enum RootMode {
    Speech,
    Batch,
}

struct RootSeed(RootMode);

impl<'de> DeserializeSeed<'de> for RootSeed {
    type Value = (SpeechFields, Vec<BatchSpeechFields>);

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(RootVisitor(self.0))
    }
}

struct RootVisitor(RootMode);

impl<'de> Visitor<'de> for RootVisitor {
    type Value = (SpeechFields, Vec<BatchSpeechFields>);

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a speech request object")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut fields = SpeechFields::default();
        let mut items = None;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "stream" => read_speech_stream(&mut map, &mut fields)?,
                "items" if matches!(self.0, RootMode::Batch) => {
                    items = Some(map.next_value_seed(ItemsSeed)?)
                }
                _ => {
                    if !read_speech_field(&key, &mut map, &mut fields)? {
                        let _ignored = map.next_value::<IgnoredAny>()?;
                    }
                }
            }
        }
        match self.0 {
            RootMode::Speech => Ok((fields, Vec::new())),
            RootMode::Batch => Ok((
                fields,
                items.ok_or_else(|| de::Error::missing_field("items"))?,
            )),
        }
    }
}

struct ItemsSeed;

impl<'de> DeserializeSeed<'de> for ItemsSeed {
    type Value = Vec<BatchSpeechFields>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ItemsVisitor;
        impl<'de> Visitor<'de> for ItemsVisitor {
            type Value = Vec<BatchSpeechFields>;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a speech batch item array")
            }
            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut result = Vec::new();
                while let Some(item) = sequence.next_element_seed(ItemSeed)? {
                    result.push(item);
                    if result.len() > usize::from(u16::MAX) {
                        return Err(de::Error::custom("too many batch items"));
                    }
                }
                Ok(result)
            }
        }
        deserializer.deserialize_seq(ItemsVisitor)
    }
}

struct ItemSeed;

impl<'de> DeserializeSeed<'de> for ItemSeed {
    type Value = BatchSpeechFields;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ItemVisitor;
        impl<'de> Visitor<'de> for ItemVisitor {
            type Value = BatchSpeechFields;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a speech batch item object")
            }
            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut fields = BatchSpeechFields::default();
                while let Some(key) = map.next_key::<String>()? {
                    if !read_batch_field(&key, &mut map, &mut fields)? {
                        let _ignored = map.next_value::<IgnoredAny>()?;
                    }
                }
                Ok(fields)
            }
        }
        deserializer.deserialize_map(ItemVisitor)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::config::Config;
    use crate::worker_pool::{
        ModelSelection, ProfileRequirement, ReferenceForm, SpeechResponseFormat, SpeechTask,
        StreamMode, TrustDomain, WorkerPool,
    };

    use super::{
        HttpFault, batch, batch_with_hints, classify_task, merge_stream, speech, speech_with_hints,
    };

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

    fn pool() -> WorkerPool {
        let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "sgl-omni-media-classify-{}-{sequence}.toml",
            std::process::id()
        ));
        let config = r#"
schema_version = 1
[server]
listen = "127.0.0.1:30000"
[shutdown]
drain_timeout_ms = 1000
[logging]
format = "json"
filter = "error"
[router]
strategy = "round_robin"
[admission]
global = 16
generation_http = 1
speech_http = 4
transcription_http = 4
speech_batch = 16
[health]
interval_ms = 100
timeout_ms = 50
success_threshold = 1
failure_threshold = 1
[http]
buffered_request_total_bytes = 4096
connect_timeout_ms = 100
pool_idle_timeout_ms = 1000
pool_max_idle_per_host = 1
[http_generation]
trust_domain = "local"
buffered_request_max_bytes = 1024
streamed_request_max_bytes = 8192
request_timeout_ms = 1000
[[workers]]
worker_id = "worker"
base_url = "http://127.0.0.1:9"
trust_domain = "local"
default_model_id = "tts"
[[workers.service_profiles]]
service = "generation_http"
model_ids = ["tts"]
message_content_forms = ["string"]
media_placements = []
input_modalities = ["text"]
output_modalities = ["text"]
chat_audio_formats = []
stream_modes = ["non_streaming"]
[[workers.service_profiles]]
service = "speech_http"
model_ids = ["tts", "other"]
response_formats = ["mp3", "opus", "aac", "flac", "wav"]
stream_modes = ["non_streaming"]
tasks = ["text_to_speech", "voice_clone", "voice_design"]
reference_forms = ["none", "direct", "list", "vq_codes"]
voice_name_policy = "preset"
[[workers.service_profiles]]
service = "speech_http"
model_ids = ["tts", "other"]
response_formats = ["pcm"]
stream_modes = ["non_streaming", "streaming"]
tasks = ["text_to_speech", "voice_clone", "voice_design"]
reference_forms = ["none", "direct", "list", "vq_codes"]
voice_name_policy = "preset"
[[workers.service_profiles]]
service = "speech_batch"
model_ids = ["tts", "other"]
response_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
tasks = ["text_to_speech", "voice_clone", "voice_design"]
reference_forms = ["none", "direct", "list", "vq_codes"]
voice_name_policy = "preset"
max_batch_size = 16
[[workers.service_profiles]]
service = "transcription_http"
model_ids = ["tts", "other"]
task = "transcribe"
response_formats = ["json", "text", "verbose_json", "srt", "vtt", "sse"]
stream_modes = ["non_streaming", "streaming"]
"#;
        fs::write(&path, config).expect("write classifier config");
        let parsed = Config::load(&path).expect("load classifier config");
        let _removed = fs::remove_file(path);
        WorkerPool::build(&parsed).expect("build classifier pool")
    }

    #[test]
    fn speech_classifies_every_format_mode_task_and_mixed_reference_set() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        for (name, expected) in [
            ("mp3", SpeechResponseFormat::Mp3),
            ("opus", SpeechResponseFormat::Opus),
            ("aac", SpeechResponseFormat::Aac),
            ("flac", SpeechResponseFormat::Flac),
            ("wav", SpeechResponseFormat::Wav),
            ("pcm", SpeechResponseFormat::Pcm),
        ] {
            let body =
                format!("{{\"model\":\"tts\",\"input\":\"x\",\"response_format\":\"{name}\"}}");
            let classified = speech(body.as_bytes(), &pool, &trust).expect("classify format");
            let ProfileRequirement::SpeechHttp {
                response_format, ..
            } = classified.requirement.profile()
            else {
                panic!("speech requirement")
            };
            assert_eq!(*response_format, expected);
        }
        let mixed = br#"{"model":"tts","input":"x","response_format":"pcm","stream":true,"task_type":"Base","ref_audio":"direct","references":[{"audio_path":"list"},{"vq_codes":[1]}]}"#;
        let classified = speech(mixed, &pool, &trust).expect("classify mixed speech");
        let ProfileRequirement::SpeechHttp {
            stream_mode,
            task,
            reference_forms,
            named_voice,
            ..
        } = classified.requirement.profile()
        else {
            panic!("speech requirement")
        };
        assert_eq!(*stream_mode, StreamMode::Streaming);
        assert_eq!(*task, Some(SpeechTask::VoiceClone));
        assert_eq!(
            reference_forms,
            &[
                ReferenceForm::Direct,
                ReferenceForm::List,
                ReferenceForm::VqCodes,
            ]
        );
        assert!(!named_voice);
        let duplicate = speech(br#"{"model":"a","model":"tts","input":"x"}"#, &pool, &trust)
            .expect("duplicate JSON fields use last-wins parsing");
        let ProfileRequirement::SpeechHttp { model, .. } = duplicate.requirement.profile() else {
            panic!("speech requirement")
        };
        assert_eq!(model.expected_model_id(), Some("tts"));
    }

    #[test]
    fn qwen_task_types_follow_worker_capabilities() {
        for (value, expected) in [
            ("Base", SpeechTask::VoiceClone),
            (" base ", SpeechTask::VoiceClone),
            ("CustomVoice", SpeechTask::TextToSpeech),
            ("custom_voice", SpeechTask::TextToSpeech),
            ("VoiceDesign", SpeechTask::VoiceDesign),
            ("voice-design", SpeechTask::VoiceDesign),
        ] {
            assert_eq!(classify_task(value), Some(expected));
        }
        assert_eq!(classify_task("future"), None);

        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        for (task_type, expected) in [
            ("Base", SpeechTask::VoiceClone),
            ("CustomVoice", SpeechTask::TextToSpeech),
        ] {
            let body = format!(r#"{{"model":"tts","input":"x","task_type":"{task_type}"}}"#);
            let classified = speech(body.as_bytes(), &pool, &trust).expect("classify Qwen task");
            let ProfileRequirement::SpeechHttp { task, .. } = classified.requirement.profile()
            else {
                panic!("speech requirement")
            };
            assert_eq!(*task, Some(expected));
        }

        let classified = batch(
            br#"{
                "model":"tts",
                "task_type":"CustomVoice",
                "items":[
                    {"input":"preset"},
                    {"input":"clone","task_type":"Base","ref_audio":"reference"}
                ]
            }"#,
            &pool,
            &trust,
        )
        .expect("classify inherited and overridden Qwen tasks");
        let ProfileRequirement::SpeechBatch { tasks, .. } = classified.requirement.profile() else {
            panic!("speech batch requirement")
        };
        assert_eq!(tasks, &[SpeechTask::TextToSpeech, SpeechTask::VoiceClone]);
    }

    #[test]
    fn speech_preserves_worker_owned_alias_reference_and_task_semantics() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        for body in [
            br#"{"model":"tts","voice":"default","speaker":"named","references":[{}]}"#.as_slice(),
            br#"{"model":"tts","speaker":"named","voice":"default","references":[{}]}"#.as_slice(),
        ] {
            let classified = speech(body, &pool, &trust).expect("classify worker-owned fields");
            let ProfileRequirement::SpeechHttp {
                task,
                reference_forms,
                named_voice,
                ..
            } = classified.requirement.profile()
            else {
                panic!("speech requirement")
            };
            assert_eq!(*task, None);
            assert_eq!(reference_forms, &[ReferenceForm::List]);
            assert!(!named_voice, "voice takes precedence over speaker");
        }
    }

    #[test]
    fn top_level_routing_fields_use_the_final_json_occurrence() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        let classified = speech(
            br#"{
                "model":7,"model":"tts",
                "response_format":[],"response_format":"pcm",
                "stream":"bad","stream":true,
                "task_type":false,"task_type":"Base",
                "voice":{},"voice":"named",
                "ref_audio":{},"ref_audio":null,
                "references":"bad","references":null,
                "input":"x"
            }"#,
            &pool,
            &trust,
        )
        .expect("valid final speech routing fields");
        let ProfileRequirement::SpeechHttp {
            model,
            response_format,
            stream_mode,
            task,
            named_voice,
            ..
        } = classified.requirement.profile()
        else {
            panic!("speech requirement")
        };
        assert_eq!(model.expected_model_id(), Some("tts"));
        assert_eq!(*response_format, SpeechResponseFormat::Pcm);
        assert_eq!(*stream_mode, StreamMode::Streaming);
        assert_eq!(*task, Some(SpeechTask::VoiceClone));
        assert!(*named_voice);

        let final_invalid = speech(br#"{"model":"tts","model":7,"input":"x"}"#, &pool, &trust)
            .expect("invalid final worker-owned value becomes an absent routing fact");
        let ProfileRequirement::SpeechHttp { model, .. } = final_invalid.requirement.profile()
        else {
            panic!("speech requirement")
        };
        assert!(matches!(model, ModelSelection::UnresolvedDefault));

        let batch = batch(
            br#"{
                "model":7,"model":"tts",
                "response_format":[],"response_format":"wav",
                "stream":"bad","stream":false,
                "items":[{"input":"x"}]
            }"#,
            &pool,
            &trust,
        )
        .expect("valid final batch-default routing fields");
        let ProfileRequirement::SpeechBatch {
            models,
            response_formats,
            ..
        } = batch.requirement.profile()
        else {
            panic!("batch requirement")
        };
        assert_eq!(models[0].expected_model_id(), Some("tts"));
        assert_eq!(response_formats, &[SpeechResponseFormat::Wav]);
    }

    #[test]
    fn named_speech_uses_voice_as_its_reference_requirement() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        let classified = speech(
            br#"{"model":"tts","input":"x","voice":"named"}"#,
            &pool,
            &trust,
        )
        .expect("classify managed speech");
        let ProfileRequirement::SpeechHttp {
            reference_forms,
            named_voice,
            ..
        } = classified.requirement.profile()
        else {
            panic!("speech requirement")
        };
        assert!(reference_forms.is_empty());
        assert!(*named_voice);
    }

    #[test]
    fn batch_builds_unique_effective_unions() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        let body = br#"{
            "model":"tts","response_format":"wav","task_type":"Base","voice":"default",
            "items":[
                {"input":"first"},
                {"input":"second","model":"other","response_format":"mp3","task_type":"VoiceDesign","ref_audio":"direct","voice":"named"},
                {"input":"third","references":[{"audio":"list"},{"vq_codes":[1]}],"voice":"managed"}
            ]
        }"#;
        let classified = batch(body, &pool, &trust).expect("classify complete batch");
        let ProfileRequirement::SpeechBatch {
            models,
            response_formats,
            tasks,
            reference_forms,
            named_voice,
            batch_size,
        } = classified.requirement.profile()
        else {
            panic!("batch requirement")
        };
        assert_eq!(*batch_size, 3);
        assert_eq!(models.len(), 2);
        assert!(
            models
                .iter()
                .all(|model| matches!(model, ModelSelection::Explicit(_)))
        );
        assert!(
            models
                .iter()
                .any(|model| model.expected_model_id() == Some("tts"))
        );
        assert!(
            models
                .iter()
                .any(|model| model.expected_model_id() == Some("other"))
        );
        assert_eq!(
            response_formats,
            &[SpeechResponseFormat::Wav, SpeechResponseFormat::Mp3]
        );
        assert_eq!(tasks, &[SpeechTask::VoiceClone, SpeechTask::VoiceDesign]);
        assert_eq!(
            reference_forms,
            &[
                ReferenceForm::None,
                ReferenceForm::Direct,
                ReferenceForm::List,
                ReferenceForm::VqCodes,
            ]
        );
        assert!(
            !named_voice,
            "explicit references avoid named voice routing"
        );
    }

    #[test]
    fn batch_default_named_voice_remains_a_top_level_requirement() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        let body = br#"{
            "model":"tts",
            "voice":"named-default",
            "items":[
                {"input":"first","voice":"default"},
                {"input":"second","voice":"default"}
            ]
        }"#;
        let classified = batch(body, &pool, &trust).expect("classify overridden default voice");
        let ProfileRequirement::SpeechBatch {
            reference_forms,
            named_voice,
            ..
        } = classified.requirement.profile()
        else {
            panic!("batch requirement")
        };
        assert_eq!(reference_forms, &[ReferenceForm::None]);
        assert!(*named_voice);
    }

    #[test]
    fn batch_default_response_format_remains_a_top_level_requirement() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        let classified = batch(
            br#"{
                "model":"tts",
                "response_format":"mp3",
                "items":[
                    {"input":"first","response_format":"wav"},
                    {"input":"second","response_format":"wav"}
                ]
            }"#,
            &pool,
            &trust,
        )
        .expect("classify overridden default response format");
        let ProfileRequirement::SpeechBatch {
            response_formats, ..
        } = classified.requirement.profile()
        else {
            panic!("batch requirement")
        };
        assert_eq!(
            response_formats,
            &[SpeechResponseFormat::Mp3, SpeechResponseFormat::Wav]
        );
    }

    #[test]
    fn batch_combines_named_voice_and_external_reference_requirements() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        let classified = batch(
            br#"{
                "model":"tts",
                "voice":"named-default",
                "items":[
                    {"input":"managed"},
                    {"input":"direct","ref_audio":"reference"},
                    {"input":"plain","voice":"default"}
                ]
            }"#,
            &pool,
            &trust,
        )
        .expect("classify mixed managed batch");
        let ProfileRequirement::SpeechBatch {
            reference_forms,
            named_voice,
            ..
        } = classified.requirement.profile()
        else {
            panic!("batch requirement")
        };
        assert_eq!(
            reference_forms,
            &[ReferenceForm::Direct, ReferenceForm::None]
        );
        assert!(*named_voice);
    }

    #[test]
    fn batch_item_validation_remains_worker_owned() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        let classified = batch(
            br#"{
                "model":"tts",
                "items":[{"input":"bad","response_format":"unknown","task_type":"unknown"}]
            }"#,
            &pool,
            &trust,
        )
        .expect("invalid item fields become worker-owned per-item failures");
        let ProfileRequirement::SpeechBatch {
            response_formats,
            tasks,
            ..
        } = classified.requirement.profile()
        else {
            panic!("batch requirement")
        };
        assert_eq!(response_formats, &[SpeechResponseFormat::Wav]);
        assert!(tasks.is_empty());

        assert_eq!(
            batch(
                br#"{"model":"tts","response_format":"unknown","items":[{"input":"x"}]}"#,
                &pool,
                &trust,
            )
            .err(),
            Some(HttpFault::MalformedRequest),
            "invalid batch defaults reject the complete request at the worker"
        );
    }

    #[test]
    fn malformed_batch_item_facts_do_not_constrain_valid_siblings() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        let classified = batch(
            br#"{
                "model":"tts",
                "response_format":"wav",
                "items":[
                    {"input":"bad","model":7,"response_format":"mp3","ref_audio":{},"references":"bad"},
                    {"input":"good","model":7,"model":"other","response_format":[],"response_format":"pcm"}
                ]
            }"#,
            &pool,
            &trust,
        )
        .expect("malformed item remains a worker-owned failure");
        let ProfileRequirement::SpeechBatch {
            models,
            response_formats,
            reference_forms,
            batch_size,
            ..
        } = classified.requirement.profile()
        else {
            panic!("batch requirement")
        };
        assert_eq!(*batch_size, 2);
        assert_eq!(classified.credits, 2);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].expected_model_id(), Some("other"));
        assert_eq!(
            response_formats,
            &[SpeechResponseFormat::Wav, SpeechResponseFormat::Pcm]
        );
        assert_eq!(reference_forms, &[ReferenceForm::None]);
    }

    #[test]
    fn batch_reference_overrides_follow_exclude_none_semantics() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        let inherited = batch(
            br#"{
                "model":"tts",
                "ref_audio":"default-direct",
                "references":[{"audio":"default-list"}],
                "items":[{"input":"first","ref_audio":null,"references":null}]
            }"#,
            &pool,
            &trust,
        )
        .expect("null item references inherit batch defaults");
        let ProfileRequirement::SpeechBatch {
            reference_forms, ..
        } = inherited.requirement.profile()
        else {
            panic!("batch requirement")
        };
        assert_eq!(
            reference_forms,
            &[ReferenceForm::Direct, ReferenceForm::List]
        );

        let suppressed = batch(
            br#"{
                "model":"tts",
                "references":[{"audio":"default-list"}],
                "items":[{"input":"first","references":[]}]
            }"#,
            &pool,
            &trust,
        )
        .expect("empty item reference list overrides batch default");
        let ProfileRequirement::SpeechBatch {
            reference_forms, ..
        } = suppressed.requirement.profile()
        else {
            panic!("batch requirement")
        };
        assert_eq!(reference_forms, &[ReferenceForm::None]);
    }

    #[test]
    fn preserves_empty_whitespace_model_and_voice_semantics() {
        let pool = pool();
        let trust = TrustDomain::new(String::from("local"));
        for (body, defaulted, expected_model, named) in [
            (
                br#"{"model":"","input":"x","voice":""}"#.as_slice(),
                true,
                None,
                false,
            ),
            (
                br#"{"model":" ","input":"x","voice":" default "}"#.as_slice(),
                false,
                Some(" "),
                true,
            ),
            (
                br#"{"input":"x","voice":"DeFaUlT"}"#.as_slice(),
                true,
                None,
                false,
            ),
        ] {
            let classified = speech(body, &pool, &trust).expect("classify model and voice facts");
            let ProfileRequirement::SpeechHttp {
                model, named_voice, ..
            } = classified.requirement.profile()
            else {
                panic!("speech requirement")
            };
            assert_eq!(model.expected_model_id(), expected_model);
            assert_eq!(
                matches!(model, ModelSelection::UnresolvedDefault),
                defaulted
            );
            assert_eq!(*named_voice, named);
        }

        for (voice, expected) in [("", false), ("DEFAULT", false), (" default ", true)] {
            let body = format!(r#"{{"voice":"{voice}","items":[{{"input":"x"}}]}}"#);
            let classified =
                batch(body.as_bytes(), &pool, &trust).expect("classify batch named voice fact");
            let ProfileRequirement::SpeechBatch { named_voice, .. } =
                classified.requirement.profile()
            else {
                panic!("batch requirement")
            };
            assert_eq!(*named_voice, expected);
        }
    }

    #[test]
    fn route_header_assertions_cannot_override_worker_semantics() {
        let trust = TrustDomain::new(String::from("local"));
        let asserted = speech_with_hints(br#"{"input":"x"}"#, Some("tts"), Some(false), &trust)
            .expect("matching default assertions");
        let ProfileRequirement::SpeechHttp {
            model, stream_mode, ..
        } = asserted.requirement.profile()
        else {
            panic!("speech requirement")
        };
        assert!(matches!(model, ModelSelection::WorkerDefault { .. }));
        assert_eq!(*stream_mode, StreamMode::NonStreaming);

        let empty_asserted =
            speech_with_hints(br#"{"model":"","input":"x"}"#, Some("tts"), None, &trust)
                .expect("empty model retains default semantics");
        let ProfileRequirement::SpeechHttp { model, .. } = empty_asserted.requirement.profile()
        else {
            panic!("speech requirement")
        };
        assert!(matches!(model, ModelSelection::WorkerDefault { .. }));

        let explicit = speech_with_hints(
            br#"{"model":"tts","input":"x","stream":true,"response_format":"pcm"}"#,
            Some("tts"),
            Some(true),
            &trust,
        )
        .expect("matching explicit assertions");
        let ProfileRequirement::SpeechHttp {
            model, stream_mode, ..
        } = explicit.requirement.profile()
        else {
            panic!("speech requirement")
        };
        assert!(matches!(model, ModelSelection::Explicit(_)));
        assert_eq!(*stream_mode, StreamMode::Streaming);

        assert_eq!(
            speech_with_hints(
                br#"{"model":"tts","input":"x"}"#,
                Some("other"),
                None,
                &trust,
            )
            .err(),
            Some(HttpFault::MalformedRequest)
        );
        assert_eq!(
            merge_stream(None, Some(true)),
            Err(HttpFault::MalformedRequest)
        );
        assert_eq!(merge_stream(None, Some(false)), Ok(false));
        assert_eq!(
            merge_stream(Some(false), Some(true)),
            Err(HttpFault::MalformedRequest)
        );
        assert_eq!(
            speech_with_hints(
                br#"{"model":"tts","input":"x","stream":true,"response_format":"wav"}"#,
                None,
                None,
                &trust,
            )
            .err(),
            Some(HttpFault::MalformedRequest)
        );
        assert_eq!(
            batch_with_hints(
                br#"{"items":[{"input":"x","model":"tts"},{"input":"y","model":"other"}]}"#,
                Some("tts"),
                None,
                &trust,
            )
            .err(),
            Some(HttpFault::MalformedRequest)
        );
    }
}
