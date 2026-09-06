use std::fmt;

use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};

use crate::error::HttpFault;
use crate::speech_facts::ScalarFactSeed;
use crate::worker_pool::profile::{InputModality, OutputModality, StreamMode};
use crate::worker_pool::{
    ChatAudioFormat, MediaPlacement, MessageContentForm, ModelSelection, ProfileRequirement,
    RouteRequirement, TrustDomain,
};

/// Worker-eligibility facts derived without reconstructing the request body.
pub(crate) struct ClassifiedRequest {
    pub(crate) requirement: RouteRequirement,
}

#[derive(Default)]
struct Facts {
    model: Option<String>,
    messages: MessageFacts,
    top_audio: bool,
    top_image: bool,
    top_video: bool,
    modalities: Vec<OutputModality>,
    audio_format: Option<ChatAudioFormat>,
    stream: Option<bool>,
}

#[derive(Default)]
struct MessageFacts {
    string: bool,
    typed: bool,
    text: bool,
    audio: bool,
    image: bool,
    video: bool,
}

#[derive(Default)]
struct PartFacts {
    text: bool,
    audio: bool,
    image: bool,
    video: bool,
}

/// Extracts routing facts while leaving worker-owned request validation upstream.
pub(crate) fn classify(bytes: &[u8], trust: &TrustDomain) -> Result<ClassifiedRequest, HttpFault> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let facts = RootSeed
        .deserialize(&mut deserializer)
        .map_err(|_| HttpFault::MalformedRequest)?;
    deserializer
        .end()
        .map_err(|_| HttpFault::MalformedRequest)?;

    let model = facts
        .model
        .filter(|model| !model.is_empty())
        .map_or(ModelSelection::UnresolvedDefault, ModelSelection::Explicit);

    let mut message_content_forms = Vec::with_capacity(2);
    if facts.messages.string {
        message_content_forms.push(MessageContentForm::String);
    }
    if facts.messages.typed {
        message_content_forms.push(MessageContentForm::TypedParts);
    }
    let top_media = facts.top_audio || facts.top_image || facts.top_video;
    let mut media_placements = Vec::with_capacity(2);
    if top_media {
        media_placements.push(MediaPlacement::TopLevel);
    }
    if facts.messages.audio || facts.messages.image || facts.messages.video {
        media_placements.push(MediaPlacement::TypedParts);
    }
    let mut input_modalities = Vec::with_capacity(4);
    if facts.messages.text {
        input_modalities.push(InputModality::Text);
    }
    if facts.top_audio || facts.messages.audio {
        input_modalities.push(InputModality::Audio);
    }
    if facts.top_image || facts.messages.image {
        input_modalities.push(InputModality::Image);
    }
    if facts.top_video || facts.messages.video {
        input_modalities.push(InputModality::Video);
    }
    let output_modalities = if facts.modalities.is_empty() {
        vec![OutputModality::Text]
    } else {
        facts.modalities
    };
    let audio_requested = output_modalities.contains(&OutputModality::Audio);
    let audio_format = if audio_requested {
        Some(facts.audio_format.unwrap_or(ChatAudioFormat::Wav))
    } else {
        None
    };
    let stream_mode = if facts.stream.unwrap_or(false) {
        StreamMode::Streaming
    } else {
        StreamMode::NonStreaming
    };
    Ok(ClassifiedRequest {
        requirement: RouteRequirement::new(
            ProfileRequirement::GenerationHttp {
                model,
                message_content_forms,
                media_placements,
                input_modalities,
                output_modalities,
                audio_format,
                stream_mode,
            },
            trust.clone(),
        ),
    })
}

struct RootSeed;

impl<'de> DeserializeSeed<'de> for RootSeed {
    type Value = Facts;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_map(RootVisitor)
    }
}

struct RootVisitor;

impl<'de> Visitor<'de> for RootVisitor {
    type Value = Facts;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a chat request object")
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut facts = Facts::default();
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "model" => facts.model = map.next_value_seed(ScalarFactSeed)?.into_string(),
                "messages" => facts.messages = map.next_value_seed(MessagesSeed)?,
                "audios" => facts.top_audio = map.next_value_seed(NonEmptyArraySeed)?,
                "images" => facts.top_image = map.next_value_seed(NonEmptyArraySeed)?,
                "videos" => facts.top_video = map.next_value_seed(NonEmptyArraySeed)?,
                "modalities" => facts.modalities = map.next_value_seed(NullableModalitiesSeed)?,
                "audio" => facts.audio_format = map.next_value_seed(NullableAudioSeed)?,
                "stream" => facts.stream = map.next_value_seed(ScalarFactSeed)?.into_bool(),
                _ => {
                    let _ignored = map.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok(facts)
    }
}

fn ignore_sequence<'de, A>(mut sequence: A) -> Result<(), A::Error>
where
    A: SeqAccess<'de>,
{
    while sequence.next_element::<IgnoredAny>()?.is_some() {}
    Ok(())
}

fn ignore_map<'de, A>(mut map: A) -> Result<(), A::Error>
where
    A: MapAccess<'de>,
{
    while map.next_key::<String>()?.is_some() {
        let _ignored = map.next_value::<IgnoredAny>()?;
    }
    Ok(())
}

struct NonEmptyArraySeed;

impl<'de> DeserializeSeed<'de> for NonEmptyArraySeed {
    type Value = bool;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ArrayVisitor;
        impl<'de> Visitor<'de> for ArrayVisitor {
            type Value = bool;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an array, null, or worker-owned value")
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<bool, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut nonempty = false;
                while seq.next_element::<IgnoredAny>()?.is_some() {
                    nonempty = true;
                }
                Ok(nonempty)
            }

            fn visit_unit<E>(self) -> Result<bool, E> {
                Ok(false)
            }

            fn visit_bool<E>(self, _value: bool) -> Result<bool, E> {
                Ok(false)
            }

            fn visit_i64<E>(self, _value: i64) -> Result<bool, E> {
                Ok(false)
            }

            fn visit_u64<E>(self, _value: u64) -> Result<bool, E> {
                Ok(false)
            }

            fn visit_f64<E>(self, _value: f64) -> Result<bool, E> {
                Ok(false)
            }

            fn visit_str<E>(self, _value: &str) -> Result<bool, E> {
                Ok(false)
            }

            fn visit_map<A>(self, map: A) -> Result<bool, A::Error>
            where
                A: MapAccess<'de>,
            {
                ignore_map(map)?;
                Ok(false)
            }
        }
        deserializer.deserialize_any(ArrayVisitor)
    }
}

struct MessagesSeed;

impl<'de> DeserializeSeed<'de> for MessagesSeed {
    type Value = MessageFacts;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct MessagesVisitor;
        impl<'de> Visitor<'de> for MessagesVisitor {
            type Value = MessageFacts;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a message array or worker-owned value")
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<MessageFacts, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut all = MessageFacts::default();
                while let Some(message) = seq.next_element_seed(MessageSeed)? {
                    all.string |= message.string;
                    all.typed |= message.typed;
                    all.text |= message.text;
                    all.audio |= message.audio;
                    all.image |= message.image;
                    all.video |= message.video;
                }
                Ok(all)
            }

            fn visit_unit<E>(self) -> Result<MessageFacts, E> {
                Ok(MessageFacts::default())
            }

            fn visit_bool<E>(self, _value: bool) -> Result<MessageFacts, E> {
                Ok(MessageFacts::default())
            }

            fn visit_i64<E>(self, _value: i64) -> Result<MessageFacts, E> {
                Ok(MessageFacts::default())
            }

            fn visit_u64<E>(self, _value: u64) -> Result<MessageFacts, E> {
                Ok(MessageFacts::default())
            }

            fn visit_f64<E>(self, _value: f64) -> Result<MessageFacts, E> {
                Ok(MessageFacts::default())
            }

            fn visit_str<E>(self, _value: &str) -> Result<MessageFacts, E> {
                Ok(MessageFacts::default())
            }

            fn visit_map<A>(self, map: A) -> Result<MessageFacts, A::Error>
            where
                A: MapAccess<'de>,
            {
                ignore_map(map)?;
                Ok(MessageFacts::default())
            }
        }
        deserializer.deserialize_any(MessagesVisitor)
    }
}

struct MessageSeed;

impl<'de> DeserializeSeed<'de> for MessageSeed {
    type Value = MessageFacts;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct MessageVisitor;
        impl<'de> Visitor<'de> for MessageVisitor {
            type Value = MessageFacts;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a message object")
            }
            fn visit_map<A>(self, mut map: A) -> Result<MessageFacts, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut content = MessageFacts::default();
                while let Some(key) = map.next_key::<String>()? {
                    if key == "content" {
                        content = map.next_value_seed(ContentSeed)?;
                    } else {
                        let _ignored = map.next_value::<IgnoredAny>()?;
                    }
                }
                Ok(content)
            }

            fn visit_unit<E>(self) -> Result<MessageFacts, E> {
                Ok(MessageFacts::default())
            }

            fn visit_bool<E>(self, _value: bool) -> Result<MessageFacts, E> {
                Ok(MessageFacts::default())
            }

            fn visit_i64<E>(self, _value: i64) -> Result<MessageFacts, E> {
                Ok(MessageFacts::default())
            }

            fn visit_u64<E>(self, _value: u64) -> Result<MessageFacts, E> {
                Ok(MessageFacts::default())
            }

            fn visit_f64<E>(self, _value: f64) -> Result<MessageFacts, E> {
                Ok(MessageFacts::default())
            }

            fn visit_str<E>(self, _value: &str) -> Result<MessageFacts, E> {
                Ok(MessageFacts::default())
            }

            fn visit_seq<A>(self, sequence: A) -> Result<MessageFacts, A::Error>
            where
                A: SeqAccess<'de>,
            {
                ignore_sequence(sequence)?;
                Ok(MessageFacts::default())
            }
        }
        deserializer.deserialize_any(MessageVisitor)
    }
}

struct ContentSeed;

fn text_message_facts() -> MessageFacts {
    MessageFacts {
        string: true,
        text: true,
        ..MessageFacts::default()
    }
}

impl<'de> DeserializeSeed<'de> for ContentSeed {
    type Value = MessageFacts;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ContentVisitor;
        impl<'de> Visitor<'de> for ContentVisitor {
            type Value = MessageFacts;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a content string or typed-part array")
            }
            fn visit_str<E>(self, _value: &str) -> Result<MessageFacts, E> {
                Ok(text_message_facts())
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<MessageFacts, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut facts = MessageFacts {
                    typed: true,
                    ..MessageFacts::default()
                };
                while let Some(part) = seq.next_element_seed(PartSeed)? {
                    facts.text |= part.text;
                    facts.audio |= part.audio;
                    facts.image |= part.image;
                    facts.video |= part.video;
                }
                Ok(facts)
            }

            fn visit_unit<E>(self) -> Result<MessageFacts, E> {
                Ok(MessageFacts::default())
            }

            fn visit_bool<E>(self, _value: bool) -> Result<MessageFacts, E> {
                Ok(text_message_facts())
            }

            fn visit_i64<E>(self, _value: i64) -> Result<MessageFacts, E> {
                Ok(text_message_facts())
            }

            fn visit_u64<E>(self, _value: u64) -> Result<MessageFacts, E> {
                Ok(text_message_facts())
            }

            fn visit_f64<E>(self, _value: f64) -> Result<MessageFacts, E> {
                Ok(text_message_facts())
            }

            fn visit_map<A>(self, map: A) -> Result<MessageFacts, A::Error>
            where
                A: MapAccess<'de>,
            {
                ignore_map(map)?;
                Ok(text_message_facts())
            }
        }
        deserializer.deserialize_any(ContentVisitor)
    }
}

struct PartSeed;

fn text_part_facts() -> PartFacts {
    PartFacts {
        text: true,
        ..PartFacts::default()
    }
}

impl<'de> DeserializeSeed<'de> for PartSeed {
    type Value = PartFacts;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct PartVisitor;
        impl<'de> Visitor<'de> for PartVisitor {
            type Value = PartFacts;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a typed content-part object")
            }
            fn visit_map<A>(self, mut map: A) -> Result<PartFacts, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut part_type = None;
                let mut type_present = false;
                while let Some(key) = map.next_key::<String>()? {
                    if key == "type" {
                        type_present = true;
                        part_type = map.next_value_seed(ScalarFactSeed)?.into_string();
                    } else {
                        let _ignored = map.next_value::<IgnoredAny>()?;
                    }
                }
                let mut facts = PartFacts::default();
                match part_type {
                    None if !type_present => facts.text = true,
                    Some(value) if value == "text" => facts.text = true,
                    Some(value) if matches!(value.as_str(), "audio_url" | "input_audio") => {
                        facts.audio = true;
                    }
                    Some(value) if matches!(value.as_str(), "image_url" | "image") => {
                        facts.image = true;
                    }
                    Some(value) if matches!(value.as_str(), "video_url" | "video") => {
                        facts.video = true;
                    }
                    _ => {}
                }
                Ok(facts)
            }

            fn visit_str<E>(self, _value: &str) -> Result<PartFacts, E> {
                Ok(text_part_facts())
            }

            fn visit_unit<E>(self) -> Result<PartFacts, E> {
                Ok(PartFacts::default())
            }

            fn visit_bool<E>(self, _value: bool) -> Result<PartFacts, E> {
                Ok(PartFacts::default())
            }

            fn visit_i64<E>(self, _value: i64) -> Result<PartFacts, E> {
                Ok(PartFacts::default())
            }

            fn visit_u64<E>(self, _value: u64) -> Result<PartFacts, E> {
                Ok(PartFacts::default())
            }

            fn visit_f64<E>(self, _value: f64) -> Result<PartFacts, E> {
                Ok(PartFacts::default())
            }

            fn visit_seq<A>(self, sequence: A) -> Result<PartFacts, A::Error>
            where
                A: SeqAccess<'de>,
            {
                ignore_sequence(sequence)?;
                Ok(PartFacts::default())
            }
        }
        deserializer.deserialize_any(PartVisitor)
    }
}

struct NullableModalitiesSeed;

struct ModalitySeed;

impl<'de> DeserializeSeed<'de> for ModalitySeed {
    type Value = Option<OutputModality>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(
            match ScalarFactSeed
                .deserialize(deserializer)?
                .into_string()
                .as_deref()
            {
                Some("text") => Some(OutputModality::Text),
                Some("audio") => Some(OutputModality::Audio),
                _ => None,
            },
        )
    }
}

impl<'de> DeserializeSeed<'de> for NullableModalitiesSeed {
    type Value = Vec<OutputModality>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ModalitiesVisitor;
        impl<'de> Visitor<'de> for ModalitiesVisitor {
            type Value = Vec<OutputModality>;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an output-modality array or worker-owned value")
            }
            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(Vec::new())
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut result = Vec::with_capacity(2);
                while let Some(modality) = seq.next_element_seed(ModalitySeed)? {
                    if let Some(modality) = modality
                        && !result.contains(&modality)
                    {
                        result.push(modality);
                    }
                }
                Ok(result)
            }

            fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
                Ok(Vec::new())
            }

            fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
                Ok(Vec::new())
            }

            fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
                Ok(Vec::new())
            }

            fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
                Ok(Vec::new())
            }

            fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
                Ok(Vec::new())
            }

            fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                ignore_map(map)?;
                Ok(Vec::new())
            }
        }
        deserializer.deserialize_any(ModalitiesVisitor)
    }
}

struct NullableAudioSeed;

impl<'de> DeserializeSeed<'de> for NullableAudioSeed {
    type Value = Option<ChatAudioFormat>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct AudioVisitor;
        impl<'de> Visitor<'de> for AudioVisitor {
            type Value = Option<ChatAudioFormat>;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an audio configuration object or worker-owned value")
            }
            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(None)
            }
            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut format = None;
                while let Some(key) = map.next_key::<String>()? {
                    if key == "format" {
                        format = map.next_value_seed(ScalarFactSeed)?.into_string();
                    } else {
                        let _ignored = map.next_value::<IgnoredAny>()?;
                    }
                }
                let Some(format) = format else {
                    return Ok(None);
                };
                Ok(parse_audio_format(&format))
            }

            fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
                Ok(None)
            }

            fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
                Ok(None)
            }

            fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
                Ok(None)
            }

            fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
                Ok(None)
            }

            fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
                Ok(None)
            }

            fn visit_seq<A>(self, sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                ignore_sequence(sequence)?;
                Ok(None)
            }
        }
        deserializer.deserialize_any(AudioVisitor)
    }
}

fn parse_audio_format(value: &str) -> Option<ChatAudioFormat> {
    let value = value.trim();
    if value.eq_ignore_ascii_case("wav") {
        Some(ChatAudioFormat::Wav)
    } else if value.eq_ignore_ascii_case("mp3") {
        Some(ChatAudioFormat::Mp3)
    } else if value.eq_ignore_ascii_case("flac") {
        Some(ChatAudioFormat::Flac)
    } else if value.eq_ignore_ascii_case("pcm") {
        Some(ChatAudioFormat::Pcm)
    } else if value.eq_ignore_ascii_case("aac") {
        Some(ChatAudioFormat::Aac)
    } else if value.eq_ignore_ascii_case("opus") {
        Some(ChatAudioFormat::Opus)
    } else {
        None
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use std::fs;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicU64, Ordering};

    use http_body::Body as _;

    use crate::config::Config;
    use crate::worker_pool::{ModelSelection, ProfileRequirement, WorkerPool};

    use super::{ClassifiedRequest, HttpFault, TrustDomain, classify};

    static NEXT_CONFIG: AtomicU64 = AtomicU64::new(0);

    fn pool(extra_worker: &str) -> WorkerPool {
        let sequence = NEXT_CONFIG.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "sgl-omni-chat-classify-{}-{sequence}.toml",
            std::process::id()
        ));
        let text = format!(
            r#"schema_version = 1
[server]
listen = "127.0.0.1:30000"
[shutdown]
drain_timeout_ms = 30000
[logging]
format = "json"
filter = "info"
[router]
[admission]
global = 8
generation_http = 4
[health]
interval_ms = 100
timeout_ms = 50
success_threshold = 1
failure_threshold = 1
[http_generation]
trust_domain = "local"
[[workers]]
worker_id = "worker-a"
base_url = "http://127.0.0.1:9"
trust_domain = "local"
default_model_id = "omni"
[[workers.service_profiles]]
service = "generation_http"
model_ids = ["omni", "other"]
message_content_forms = ["string", "typed_parts"]
media_placements = ["top_level", "typed_parts"]
input_modalities = ["text", "audio", "image", "video"]
output_modalities = ["text", "audio"]
chat_audio_formats = ["wav", "mp3", "flac", "pcm", "aac", "opus"]
stream_modes = ["non_streaming", "streaming"]
{extra_worker}
"#
        );
        fs::write(&path, text).expect("write classifier config");
        let config = Config::load(&path).expect("load classifier config");
        let _removed = fs::remove_file(path);
        WorkerPool::build(&config).expect("build classifier pool")
    }

    fn classify_with(body: &str, pool: &WorkerPool) -> Result<(), HttpFault> {
        classify_full(body.as_bytes(), pool).map(drop)
    }

    fn classify_full(bytes: &[u8], _pool: &WorkerPool) -> Result<ClassifiedRequest, HttpFault> {
        classify(bytes, &TrustDomain::new(String::from("local")))
    }

    fn permutations(fields: &mut [&str], index: usize, output: &mut Vec<String>) {
        if index == fields.len() {
            output.push(format!("{{{}}}", fields.join(",")));
            return;
        }
        for swap in index..fields.len() {
            fields.swap(index, swap);
            permutations(fields, index + 1, output);
            fields.swap(index, swap);
        }
    }

    fn next_xorshift(state: &mut u64) -> u64 {
        let mut value = *state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        *state = value;
        value
    }

    #[test]
    fn model_and_output_defaults_match_direct_omni() {
        let pool = pool("");
        for model in ["", r#","model":null"#, r#","model":"omni""#] {
            let body = format!(r#"{{"messages":[{{"content":"hello"}}]{model}}}"#);
            assert_eq!(classify_with(&body, &pool), Ok(()));
        }
        for modalities in ["", r#","modalities":null"#, r#","modalities":[]"#] {
            let body = format!(r#"{{"messages":[{{"content":"hello"}}]{modalities}}}"#);
            assert!(classify_with(&body, &pool).is_ok());
        }
        assert_eq!(
            classify_with(
                r#"{"messages":[{"content":"hello"}],"modalities":["audio"],"audio":null,"stream":true}"#,
                &pool,
            ),
            Ok(())
        );
    }

    #[test]
    fn mixed_message_and_media_forms_are_classified_without_field_order_dependence() {
        let pool = pool("");
        let requests = [
            r#"{"videos":[1],"messages":[{"content":"text"},{"content":[{"image_url":{},"type":"image_url"},{"type":"audio_url"},{"type":"video"},{"type":"unknown"}]}],"images":[1],"audios":[1],"model":"omni"}"#,
            r#"{"model":"omni","audios":[1],"images":[1],"messages":[{"content":"text"},{"content":[{"type":"image"},{"type":"input_audio"},{"type":"video_url"}]}],"videos":[1]}"#,
        ];
        for request in requests {
            assert!(classify_with(request, &pool).is_ok());
        }
        for format in ["wav", "mp3", "flac", "pcm", "aac", "opus"] {
            let body = format!(
                r#"{{"messages":[{{"content":"x"}}],"modalities":["text","audio"],"audio":{{"format":"{format}"}}}}"#
            );
            assert!(classify_with(&body, &pool).is_ok());
        }
    }

    #[test]
    fn audio_formats_match_worker_normalization() {
        let pool = pool("");
        for (canonical, variant) in [
            ("wav", " WAV "),
            ("mp3", " Mp3 "),
            ("flac", " FLAC "),
            ("pcm", " pCm "),
            ("aac", " AAC "),
            ("opus", " OPUS "),
        ] {
            let canonical = format!(
                r#"{{"messages":[{{"content":"x"}}],"modalities":["audio"],"audio":{{"format":"{canonical}"}}}}"#
            );
            let variant = format!(
                r#"{{"messages":[{{"content":"x"}}],"modalities":["audio"],"audio":{{"format":"{variant}"}}}}"#
            );
            assert_eq!(
                classify_full(variant.as_bytes(), &pool)
                    .expect("worker-normalized format must classify")
                    .requirement,
                classify_full(canonical.as_bytes(), &pool)
                    .expect("canonical format must classify")
                    .requirement,
            );
        }

        let unknown = classify_full(
            br#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{"format":" vendor "}}"#,
            &pool,
        )
        .expect("unknown worker format falls back to wav");
        let wav = classify_full(
            br#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{"format":"wav"}}"#,
            &pool,
        )
        .expect("wav request must classify");
        assert_eq!(unknown.requirement, wav.requirement);
    }

    #[test]
    fn worker_owned_values_do_not_change_request_acceptance() {
        let pool = pool("");
        let accepted = [
            r#"{"model":"omni","model":"omni","messages":[{"content":"x"}]}"#,
            r#"{"messages":[{"content":"x","content":"y"}]}"#,
            r#"{"messages":[{"content":[{"type":"text","type":"text"}]}]}"#,
            r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{"format":"wav","format":"wav"}}"#,
            r#"{"messages":[]}"#,
            r#"{"model":"","messages":[{"content":"x"}]}"#,
            r#"{"messages":[{}]}"#,
            r#"{"messages":[{"content":null}]}"#,
            r#"{"messages":[{"content":3}]}"#,
            r#"{"messages":[{"content":{}}]}"#,
            r#"{"messages":[{"content":["raw"]}]}"#,
            r#"{"messages":[{"content":[{}]}]}"#,
            r#"{"messages":[{"content":"x"}],"modalities":["text","text"]}"#,
            r#"{"messages":[{"content":"x"}],"modalities":["video"]}"#,
            r#"{"messages":[{"content":"x"}],"stream":"true"}"#,
            r#"{"messages":[{"content":"x"}],"stream":1}"#,
            r#"{"messages":[{"content":"x"}],"audios":null}"#,
            r#"{"messages":[{"content":"x"}],"audios":{}}"#,
        ];
        for request in accepted {
            assert!(
                classify_with(request, &pool).is_ok(),
                "worker-owned request was rejected: {request}"
            );
        }
        assert_eq!(
            classify_with(r#"{"messages":[{"content":"x"}]} trailing"#, &pool),
            Err(HttpFault::MalformedRequest)
        );
        assert!(
            classify_with(
                r#"{"messages":[{"content":"x"}],"vendor":{"temperature":1,"temperature":2}}"#,
                &pool,
            )
            .is_ok()
        );
        assert!(
            classify_with(
                r#"{"messages":[{"content":[{"type":"vendor_future_part"}]}]}"#,
                &pool,
            )
            .is_ok()
        );
        let long_model = format!(
            r#"{{"model":"{}","messages":[{{"content":"x"}}]}}"#,
            "m".repeat(257)
        );
        assert!(
            classify_with(&long_model, &pool).is_ok(),
            "nonempty model IDs remain dispatch-owned even when no row will match"
        );
    }

    #[test]
    fn ignored_nesting_is_iterative_and_default_resolution_is_deferred() {
        let base_pool = pool("");
        let nested = format!(
            "{{\"messages\":[{{\"content\":\"x\"}}],\"ignored\":{}0{}}}",
            "[".repeat(129),
            "]".repeat(129)
        );
        assert!(classify_with(&nested, &base_pool).is_ok());
        let second = r#"
[[workers]]
worker_id = "worker-b"
base_url = "http://127.0.0.1:10"
trust_domain = "local"
default_model_id = "other"
[[workers.service_profiles]]
service = "generation_http"
model_ids = ["omni", "other"]
message_content_forms = ["string", "typed_parts"]
media_placements = ["top_level", "typed_parts"]
input_modalities = ["text", "audio", "image", "video"]
output_modalities = ["text", "audio"]
chat_audio_formats = ["wav", "mp3", "flac", "pcm", "aac", "opus"]
stream_modes = ["non_streaming", "streaming"]
"#;
        let ambiguous = pool(second);
        let classified = classify_full(br#"{"messages":[{"content":"x"}]}"#, &ambiguous)
            .expect("classification defers default resolution to dispatch");
        let ProfileRequirement::GenerationHttp { model, .. } = classified.requirement.profile()
        else {
            panic!("generation requirement")
        };
        assert!(matches!(model, ModelSelection::UnresolvedDefault));
        assert!(
            classify_with(
                r#"{"model":"omni","messages":[{"content":"x"}]}"#,
                &ambiguous,
            )
            .is_ok()
        );
    }

    #[test]
    fn all_root_field_permutations_and_buffer_chunk_replays_are_identical() {
        let pool = pool("");
        let mut fields = [
            r#""model":"omni""#,
            r#""messages":[{"content":"text"},{"content":[{"type":"input_audio"},{"type":"image_url"},{"type":"video"}]}]"#,
            r#""audios":[1]"#,
            r#""modalities":["text","audio"]"#,
            r#""audio":{"format":"opus"}"#,
            r#""stream":true"#,
        ];
        let mut requests = Vec::new();
        permutations(&mut fields, 0, &mut requests);
        assert_eq!(requests.len(), 720);

        let baseline = classify_full(requests[0].as_bytes(), &pool)
            .expect("baseline permutation must classify");
        for request in &requests[1..] {
            let classified = classify_full(request.as_bytes(), &pool)
                .expect("every root permutation must classify");
            assert_eq!(classified.requirement, baseline.requirement);
        }

        for width in [1, 2, 3, 7, 31, requests[0].len()] {
            let replay: Vec<u8> = requests[0]
                .as_bytes()
                .chunks(width)
                .flat_map(|chunk| chunk.iter().copied())
                .collect();
            let classified = classify_full(&replay, &pool).expect("chunk replay must classify");
            assert_eq!(classified.requirement, baseline.requirement);
        }
    }

    #[test]
    fn routing_duplicates_use_the_last_value_and_replay_deterministically() {
        let pool = pool("");
        let cases = [
            (
                r#"{"model":"other","model":"omni","messages":[{"content":"x"}]}"#,
                r#"{"model":"omni","messages":[{"content":"x"}]}"#,
            ),
            (
                r#"{"messages":[{"content":[{"type":"image"}]}],"messages":[{"content":"y"}]}"#,
                r#"{"messages":[{"content":"y"}]}"#,
            ),
            (
                r#"{"messages":[{"content":"x"}],"audios":[],"audios":[1]}"#,
                r#"{"messages":[{"content":"x"}],"audios":[1]}"#,
            ),
            (
                r#"{"messages":[{"content":"x"}],"images":[],"images":[1]}"#,
                r#"{"messages":[{"content":"x"}],"images":[1]}"#,
            ),
            (
                r#"{"messages":[{"content":"x"}],"videos":[],"videos":[1]}"#,
                r#"{"messages":[{"content":"x"}],"videos":[1]}"#,
            ),
            (
                r#"{"messages":[{"content":"x"}],"modalities":["audio"],"modalities":["text"]}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["text"]}"#,
            ),
            (
                r#"{"messages":[{"content":"x"}],"audio":{"format":"opus"},"audio":{"format":"wav"}}"#,
                r#"{"messages":[{"content":"x"}],"audio":{"format":"wav"}}"#,
            ),
            (
                r#"{"messages":[{"content":"x"}],"stream":false,"stream":true}"#,
                r#"{"messages":[{"content":"x"}],"stream":true}"#,
            ),
            (
                r#"{"messages":[{"content":[{"type":"image","type":"text"}]}]}"#,
                r#"{"messages":[{"content":[{"type":"text"}]}]}"#,
            ),
            (
                r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{"format":"wav","format":"opus"}}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{"format":"opus"}}"#,
            ),
            (
                r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{"format":"opus","format":" vendor "}}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{"format":"wav"}}"#,
            ),
            (
                r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{"format":"vendor","format":" OPUS "}}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{"format":"opus"}}"#,
            ),
        ];
        for (request, canonical) in cases {
            let expected = classify_full(canonical.as_bytes(), &pool)
                .expect("canonical last-value request must classify");
            for width in [1, 2, 5, request.len()] {
                let replay: Vec<u8> = request
                    .as_bytes()
                    .chunks(width)
                    .flat_map(|chunk| chunk.iter().copied())
                    .collect();
                assert_eq!(
                    classify_full(&replay, &pool)
                        .expect("last-value request must classify")
                        .requirement,
                    expected.requirement,
                    "duplicate replay width {width} changed: {request}"
                );
            }
        }

        for nested_arrays in [128, 129, 512] {
            let request = format!(
                "{{\"messages\":[{{\"content\":\"x\"}}],\"ignored\":{}0{}}}",
                "[".repeat(nested_arrays),
                "]".repeat(nested_arrays)
            );
            assert!(
                classify_full(request.as_bytes(), &pool).is_ok(),
                "ignored nesting failed at {nested_arrays} arrays"
            );
        }
    }

    #[test]
    fn typed_modality_and_default_equivalence_corpus_is_exact() {
        let pool = pool("");
        let equivalent_groups: &[&[&str]] = &[
            &[
                r#"{"messages":[{"content":"x"}]}"#,
                r#"{"messages":[{"content":"x"}],"model":null}"#,
                r#"{"messages":[{"content":"x"}],"model":""}"#,
            ],
            &[
                r#"{"messages":[{"content":"x"}]}"#,
                r#"{"messages":[{"content":"x"}],"modalities":null}"#,
                r#"{"messages":[{"content":"x"}],"modalities":[]}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["text"]}"#,
            ],
            &[
                r#"{"messages":[{"content":"x"}],"modalities":["audio"]}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":null}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{}}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{"format":null}}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{"format":"wav"}}"#,
            ],
            &[
                r#"{"messages":[{"content":[{"type":"audio_url"}]}]}"#,
                r#"{"messages":[{"content":[{"type":"input_audio"}]}]}"#,
            ],
            &[
                r#"{"messages":[{"content":[{"type":"image_url"}]}]}"#,
                r#"{"messages":[{"content":[{"type":"image"}]}]}"#,
            ],
            &[
                r#"{"messages":[{"content":[{"type":"video_url"}]}]}"#,
                r#"{"messages":[{"content":[{"type":"video"}]}]}"#,
            ],
            &[
                r#"{"messages":[{"content":[{"type":"text"}]}]}"#,
                r#"{"messages":[{"content":[{}]}]}"#,
                r#"{"messages":[{"content":["raw"]}]}"#,
            ],
            &[
                r#"{"messages":[{"content":[]}]}"#,
                r#"{"messages":[{"content":[{"type":"vendor_future_part"}]}]}"#,
                r#"{"messages":[{"content":[{"type":"unknown_a"},{"type":"unknown_b"}]}]}"#,
            ],
            &[
                r#"{"messages":[{"content":"x"}]}"#,
                r#"{"messages":[{"content":"x"}],"stream":false}"#,
            ],
            &[
                r#"{"messages":[{"content":"x"}]}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["video"]}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["text","text"]}"#,
            ],
            &[
                r#"{"messages":[{"content":"x"}],"modalities":["audio"]}"#,
                r#"{"messages":[{"content":"x"}],"modalities":["audio"],"audio":{"format":"vendor"}}"#,
            ],
        ];

        for group in equivalent_groups {
            let baseline = classify_full(group[0].as_bytes(), &pool)
                .expect("equivalence baseline must classify");
            for request in &group[1..] {
                let classified = classify_full(request.as_bytes(), &pool)
                    .expect("equivalent request must classify");
                assert_eq!(classified.requirement, baseline.requirement, "{request}");
            }
        }

        let defaulted = classify_full(br#"{"messages":[{"content":"x"}]}"#, &pool)
            .expect("defaulted model request");
        let explicit = classify_full(br#"{"model":"omni","messages":[{"content":"x"}]}"#, &pool)
            .expect("explicit model request");
        assert_ne!(
            explicit.requirement, defaulted.requirement,
            "equal effective IDs retain explicit/default routing provenance"
        );

        let streaming = classify_full(br#"{"messages":[{"content":"x"}],"stream":true}"#, &pool)
            .expect("streaming request");
        for request in [
            r#"{"messages":[{"content":"x"}],"stream":"true"}"#,
            r#"{"messages":[{"content":"x"}],"stream":1}"#,
            r#"{"messages":[{"content":"x"}],"stream":"on"}"#,
        ] {
            assert_eq!(
                classify_full(request.as_bytes(), &pool)
                    .expect("worker-coercible stream value")
                    .requirement,
                streaming.requirement,
                "{request}"
            );
        }

        for request in [
            r#"{"model":1,"messages":[{"content":"x"}]}"#,
            r#"{"messages":"x"}"#,
            r#"{"messages":[{"content":"x"}],"modalities":"text"}"#,
            r#"{"messages":[{"content":"x"}],"audio":"wav"}"#,
            r#"{"messages":[{"content":"x"}],"stream":1}"#,
            r#"{"messages":[{"content":[{"type":null}]}]}"#,
        ] {
            assert!(
                classify_with(request, &pool).is_ok(),
                "worker-owned routing value was rejected: {request}"
            );
        }
    }

    #[test]
    fn fixed_seed_adversarial_mutation_replay_is_panic_free_and_deterministic() {
        const SEED: u64 = 0x4f4d_4e49_524f_5554;
        const MAX_MUTATED_BYTES: usize = 1024;
        const ADVERSARIAL_BYTES: &[u8] = &[
            b'{', b'}', b'[', b']', b'"', b'\\', b':', b',', 0, 0x1f, 0x7f, 0xff, b'n', b't', b'-',
        ];
        let corpus: &[&[u8]] = &[
            br#"{"messages":[{"content":"x"}]}"#,
            br#"{"model":"omni","model":"other","messages":[{"content":"x"}]}"#,
            br#"{"messages":[{"content":[{"type":"input_audio"},{"type":"image_url"},{"type":"video"}]}],"modalities":["text","audio"],"audio":{"format":"opus"},"stream":true}"#,
            br#"{"messages":[{"content":"x","content":"y"}],"ignored":{"vendor":[1,2,3]}}"#,
            br#"{"messages":[{"content":"x"}],"modalities":null,"audio":null,"stream":false}"#,
            br#"{"messages":[]}"#,
            br#"{"messages":[{"content":"x"}]} trailing"#,
            b"{\"messages\":[{\"content\":\"unterminated",
            b"\xff\x00[{\"messages\":true}]",
        ];
        let pool = pool("");
        let mut state = SEED;

        for case in 0..2048 {
            let corpus_index = next_xorshift(&mut state) as usize % corpus.len();
            let mut bytes = corpus[corpus_index].to_vec();
            let mutation_count = 1 + next_xorshift(&mut state) as usize % 12;
            for _ in 0..mutation_count {
                let operation = next_xorshift(&mut state) % 6;
                let random_byte =
                    ADVERSARIAL_BYTES[next_xorshift(&mut state) as usize % ADVERSARIAL_BYTES.len()];
                let index = if bytes.is_empty() {
                    0
                } else {
                    next_xorshift(&mut state) as usize % bytes.len()
                };
                match operation {
                    0 if !bytes.is_empty() => {
                        bytes[index] ^= 1 << (next_xorshift(&mut state) % 8);
                    }
                    1 if bytes.len() < MAX_MUTATED_BYTES => {
                        let insertion = if bytes.is_empty() {
                            0
                        } else {
                            next_xorshift(&mut state) as usize % (bytes.len() + 1)
                        };
                        bytes.insert(insertion, random_byte);
                    }
                    2 if !bytes.is_empty() => {
                        bytes.remove(index);
                    }
                    3 if !bytes.is_empty() => bytes[index] = random_byte,
                    4 if !bytes.is_empty() && bytes.len() < MAX_MUTATED_BYTES => {
                        let remaining = MAX_MUTATED_BYTES - bytes.len();
                        let width = (1 + next_xorshift(&mut state) as usize % 16)
                            .min(bytes.len() - index)
                            .min(remaining);
                        let duplicate = bytes[index..index + width].to_vec();
                        bytes.extend_from_slice(&duplicate);
                    }
                    _ if bytes.len() < MAX_MUTATED_BYTES => bytes.push(random_byte),
                    _ => {
                        bytes.truncate(index);
                    }
                }
            }

            let first = catch_unwind(AssertUnwindSafe(|| {
                classify_full(&bytes, &pool).map(|classified| classified.requirement)
            }));
            assert!(
                first.is_ok(),
                "classifier panicked for seed=0x{SEED:016x} case={case} bytes={bytes:?}"
            );
            let first = first.expect("panic result checked above");
            let second = catch_unwind(AssertUnwindSafe(|| {
                classify_full(&bytes, &pool).map(|classified| classified.requirement)
            }));
            assert!(
                second.is_ok(),
                "classifier replay panicked for seed=0x{SEED:016x} case={case} bytes={bytes:?}"
            );
            assert_eq!(
                first,
                second.expect("panic result checked above"),
                "classifier was nondeterministic for seed=0x{SEED:016x} case={case} bytes={bytes:?}"
            );
        }
    }

    #[test]
    fn fixed_error_envelopes_are_bounded_and_canonical() {
        let faults = [
            HttpFault::MalformedRequest,
            HttpFault::AmbiguousModel,
            HttpFault::RequestTimeout,
            HttpFault::RequestBodyTooLarge,
            HttpFault::UnsupportedMediaType,
            HttpFault::UnsupportedContentEncoding,
            HttpFault::ExpectationFailed,
            HttpFault::NoCompatibleWorker,
            HttpFault::RouterOverloaded,
            HttpFault::InternalError,
            HttpFault::UpstreamProtocolError,
            HttpFault::RouterUnavailable,
            HttpFault::UpstreamTimeout,
            HttpFault::HttpVersionNotSupported,
        ];
        for fault in faults {
            let response = fault.into_response();
            let exact = response
                .body()
                .size_hint()
                .exact()
                .expect("fixed router envelope has an exact body length");
            assert_eq!(
                response.headers().get("cache-control"),
                Some(&axum::http::HeaderValue::from_static("no-store"))
            );
            assert!(exact <= 256);
            assert_eq!(
                response
                    .headers()
                    .get("content-length")
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.parse::<u64>().ok()),
                Some(exact)
            );
        }
    }
}
