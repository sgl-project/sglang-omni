use std::borrow::Cow;
use std::fmt;

use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};

use crate::worker_pool::{ReferenceForm, SpeechResponseFormat, SpeechTask};

/// Raw routing fields shared by stateless and WebSocket speech requests.
#[derive(Clone, Default)]
pub(crate) struct SpeechFields {
    pub(crate) model: Option<Option<String>>,
    pub(crate) response_format: Option<Option<String>>,
    pub(crate) stream: Option<Option<bool>>,
    pub(crate) task: Option<Option<String>>,
    pub(crate) voice: Option<Option<String>>,
    voice_present: bool,
    pub(crate) ref_audio: Option<bool>,
    pub(crate) references: Option<Option<ReferenceFlags>>,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct ReferenceFlags {
    pub(crate) list: bool,
    pub(crate) vq_codes: bool,
}

#[derive(Default)]
pub(crate) struct BatchSpeechFields {
    pub(crate) fields: SpeechFields,
    invalid_fields: u8,
}

impl BatchSpeechFields {
    pub(crate) const fn routing_fields_valid(&self) -> bool {
        self.invalid_fields == 0
    }

    fn record(&mut self, field: u8, valid: bool) {
        if valid {
            self.invalid_fields &= !field;
        } else {
            self.invalid_fields |= field;
        }
    }
}

const MODEL_FIELD: u8 = 1 << 0;
const FORMAT_FIELD: u8 = 1 << 1;
const TASK_FIELD: u8 = 1 << 2;
const VOICE_FIELD: u8 = 1 << 3;
const REF_AUDIO_FIELD: u8 = 1 << 4;
const REFERENCES_FIELD: u8 = 1 << 5;

/// Reads one routing field, returning false without consuming unknown values.
pub(crate) fn read_field<'de, A>(
    key: &str,
    map: &mut A,
    fields: &mut SpeechFields,
) -> Result<bool, A::Error>
where
    A: MapAccess<'de>,
{
    match key {
        "model" => fields.model = Some(map.next_value_seed(ScalarFactSeed)?.into_string()),
        "response_format" => {
            fields.response_format = Some(map.next_value_seed(ScalarFactSeed)?.into_string())
        }
        "task_type" => fields.task = Some(map.next_value_seed(ScalarFactSeed)?.into_string()),
        "voice" => {
            fields.voice = Some(map.next_value_seed(ScalarFactSeed)?.into_string());
            fields.voice_present = true;
        }
        "speaker" => {
            let value = map.next_value_seed(ScalarFactSeed)?.into_string();
            if !fields.voice_present {
                fields.voice = Some(value);
            }
        }
        "ref_audio" => fields.ref_audio = Some(map.next_value_seed(ScalarFactSeed)?.is_string()),
        "references" => {
            fields.references = Some(map.next_value_seed(ReferencesFactSeed)?.into_value())
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// Reads a Pydantic-compatible boolean routing fact without rejecting other worker-owned values.
pub(crate) fn read_stream<'de, A>(map: &mut A, fields: &mut SpeechFields) -> Result<(), A::Error>
where
    A: MapAccess<'de>,
{
    fields.stream = Some(map.next_value_seed(ScalarFactSeed)?.into_bool());
    Ok(())
}

/// Reads one permissive batch-item field while retaining whether its final shape is routable.
pub(crate) fn read_batch_field<'de, A>(
    key: &str,
    map: &mut A,
    item: &mut BatchSpeechFields,
) -> Result<bool, A::Error>
where
    A: MapAccess<'de>,
{
    match key {
        "model" => {
            let (value, valid) = map.next_value_seed(ScalarFactSeed)?.into_nullable_string();
            item.fields.model = Some(value);
            item.record(MODEL_FIELD, valid);
        }
        "response_format" => {
            let (value, valid) = map.next_value_seed(ScalarFactSeed)?.into_nullable_string();
            item.fields.response_format = Some(value);
            item.record(FORMAT_FIELD, valid);
        }
        "task_type" => {
            let (value, valid) = map.next_value_seed(ScalarFactSeed)?.into_nullable_string();
            item.fields.task = Some(value);
            item.record(TASK_FIELD, valid);
        }
        "voice" => {
            let (value, valid) = map.next_value_seed(ScalarFactSeed)?.into_nullable_string();
            item.fields.voice = Some(value);
            item.fields.voice_present = true;
            item.record(VOICE_FIELD, valid);
        }
        "speaker" => {
            let (value, valid) = map.next_value_seed(ScalarFactSeed)?.into_nullable_string();
            if !item.fields.voice_present {
                item.fields.voice = Some(value);
                item.record(VOICE_FIELD, valid);
            }
        }
        "ref_audio" => {
            let (present, valid) = map.next_value_seed(ScalarFactSeed)?.into_string_presence();
            item.fields.ref_audio = Some(present);
            item.record(REF_AUDIO_FIELD, valid);
        }
        "references" => {
            let (value, valid) = map.next_value_seed(ReferencesFactSeed)?.into_parts();
            item.fields.references = Some(value);
            item.record(REFERENCES_FIELD, valid);
        }
        _ => return Ok(false),
    }
    Ok(true)
}

pub(crate) fn response_format(value: &str) -> Option<SpeechResponseFormat> {
    match value.trim().to_ascii_lowercase().as_str() {
        "mp3" => Some(SpeechResponseFormat::Mp3),
        "opus" => Some(SpeechResponseFormat::Opus),
        "aac" => Some(SpeechResponseFormat::Aac),
        "flac" => Some(SpeechResponseFormat::Flac),
        "wav" => Some(SpeechResponseFormat::Wav),
        "pcm" => Some(SpeechResponseFormat::Pcm),
        _ => None,
    }
}

pub(crate) fn task(value: &str) -> Option<SpeechTask> {
    let normalized = value.trim().replace(['_', '-'], "").to_ascii_lowercase();
    match normalized.as_str() {
        "base" => Some(SpeechTask::VoiceClone),
        "customvoice" => Some(SpeechTask::TextToSpeech),
        "voicedesign" => Some(SpeechTask::VoiceDesign),
        _ => None,
    }
}

pub(crate) fn reference_forms(fields: &SpeechFields) -> Vec<ReferenceForm> {
    collect_reference_forms(fields.ref_audio == Some(true), fields.references.flatten())
}

pub(crate) fn effective_reference_forms(
    defaults: &SpeechFields,
    item: &SpeechFields,
) -> Vec<ReferenceForm> {
    let has_ref_audio = item
        .ref_audio
        .filter(|present| *present)
        .or_else(|| defaults.ref_audio.filter(|present| *present))
        .is_some();
    let references = item
        .references
        .flatten()
        .or_else(|| defaults.references.flatten());
    collect_reference_forms(has_ref_audio, references)
}

fn collect_reference_forms(
    has_ref_audio: bool,
    references: Option<ReferenceFlags>,
) -> Vec<ReferenceForm> {
    let mut result = Vec::with_capacity(3);
    if has_ref_audio {
        result.push(ReferenceForm::Direct);
    }
    if let Some(flags) = references {
        if flags.list {
            result.push(ReferenceForm::List);
        }
        if flags.vq_codes {
            result.push(ReferenceForm::VqCodes);
        }
    }
    if result.is_empty() {
        result.push(ReferenceForm::None);
    }
    result
}

pub(crate) fn named_voice(fields: &SpeechFields, references: &[ReferenceForm]) -> bool {
    references == [ReferenceForm::None]
        && fields
            .voice
            .as_ref()
            .and_then(Option::as_deref)
            .is_some_and(|voice| !voice.is_empty() && !voice.eq_ignore_ascii_case("default"))
}

pub(crate) enum ScalarFact<'a> {
    String(Cow<'a, str>),
    Bool(bool),
    Signed(i64),
    Unsigned(u64),
    Float(f64),
    Null,
    Other,
}

impl ScalarFact<'_> {
    pub(crate) fn into_string(self) -> Option<String> {
        match self {
            Self::String(value) => Some(value.into_owned()),
            _ => None,
        }
    }

    fn into_nullable_string(self) -> (Option<String>, bool) {
        match self {
            Self::String(value) => (Some(value.into_owned()), true),
            Self::Null => (None, true),
            Self::Bool(_) | Self::Signed(_) | Self::Unsigned(_) | Self::Float(_) | Self::Other => {
                (None, false)
            }
        }
    }

    fn into_string_presence(self) -> (bool, bool) {
        match self {
            Self::String(_) => (true, true),
            Self::Null => (false, true),
            Self::Bool(_) | Self::Signed(_) | Self::Unsigned(_) | Self::Float(_) | Self::Other => {
                (false, false)
            }
        }
    }

    pub(crate) fn into_bool(self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(value),
            Self::Signed(value) => bool_from_integer(value),
            Self::Unsigned(value) => bool_from_integer(value),
            Self::Float(0.0) => Some(false),
            Self::Float(1.0) => Some(true),
            Self::String(value) => parse_bool_fact(&value),
            Self::Float(_) | Self::Null | Self::Other => None,
        }
    }

    fn is_string(&self) -> bool {
        matches!(self, Self::String(_))
    }
}

pub(crate) struct ScalarFactSeed;

impl<'de> DeserializeSeed<'de> for ScalarFactSeed {
    type Value = ScalarFact<'de>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ScalarVisitor;
        impl<'de> Visitor<'de> for ScalarVisitor {
            type Value = ScalarFact<'de>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a scalar or worker-owned value")
            }

            fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E> {
                Ok(ScalarFact::String(Cow::Borrowed(value)))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
                Ok(ScalarFact::String(Cow::Owned(value.to_owned())))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
                Ok(ScalarFact::String(Cow::Owned(value)))
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(ScalarFact::Null)
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
                Ok(ScalarFact::Bool(value))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
                Ok(ScalarFact::Signed(value))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
                Ok(ScalarFact::Unsigned(value))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E> {
                Ok(ScalarFact::Float(value))
            }

            fn visit_seq<A>(self, sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                ignore_sequence(sequence)?;
                Ok(ScalarFact::Other)
            }

            fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                ignore_map(map)?;
                Ok(ScalarFact::Other)
            }
        }
        deserializer.deserialize_any(ScalarVisitor)
    }
}

fn bool_from_integer<T>(value: T) -> Option<bool>
where
    T: Eq + From<u8>,
{
    if value == T::from(0) {
        Some(false)
    } else if value == T::from(1) {
        Some(true)
    } else {
        None
    }
}

fn parse_bool_fact(value: &str) -> Option<bool> {
    if ["1", "on", "t", "true", "y", "yes"]
        .iter()
        .any(|candidate| value.eq_ignore_ascii_case(candidate))
    {
        Some(true)
    } else if ["0", "off", "f", "false", "n", "no"]
        .iter()
        .any(|candidate| value.eq_ignore_ascii_case(candidate))
    {
        Some(false)
    } else {
        None
    }
}

enum ReferencesFact {
    Value(Option<ReferenceFlags>),
    Invalid,
}

impl ReferencesFact {
    const fn into_value(self) -> Option<ReferenceFlags> {
        match self {
            Self::Value(value) => value,
            Self::Invalid => None,
        }
    }

    const fn into_parts(self) -> (Option<ReferenceFlags>, bool) {
        match self {
            Self::Value(value) => (value, true),
            Self::Invalid => (None, false),
        }
    }
}

struct ReferencesFactSeed;

impl<'de> DeserializeSeed<'de> for ReferencesFactSeed {
    type Value = ReferencesFact;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ReferencesVisitor;
        impl<'de> Visitor<'de> for ReferencesVisitor {
            type Value = ReferencesFact;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a reference array or worker-owned value")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut flags = ReferenceFlags::default();
                let mut valid = true;
                while let Some(entry) = sequence.next_element_seed(ReferenceFactSeed)? {
                    if let Some(entry) = entry {
                        flags.list = true;
                        flags.vq_codes |= entry.vq_codes;
                    } else {
                        valid = false;
                    }
                }
                Ok(if valid {
                    ReferencesFact::Value(Some(flags))
                } else {
                    ReferencesFact::Invalid
                })
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(ReferencesFact::Value(None))
            }

            fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
                Ok(ReferencesFact::Invalid)
            }

            fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
                Ok(ReferencesFact::Invalid)
            }

            fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
                Ok(ReferencesFact::Invalid)
            }

            fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
                Ok(ReferencesFact::Invalid)
            }

            fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
                Ok(ReferencesFact::Invalid)
            }

            fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                ignore_map(map)?;
                Ok(ReferencesFact::Invalid)
            }
        }
        deserializer.deserialize_any(ReferencesVisitor)
    }
}

struct ReferenceFactSeed;

impl<'de> DeserializeSeed<'de> for ReferenceFactSeed {
    type Value = Option<ReferenceFlags>;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct ReferenceVisitor;
        impl<'de> Visitor<'de> for ReferenceVisitor {
            type Value = Option<ReferenceFlags>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a speech reference object or worker-owned value")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut vq_codes = false;
                while let Some(key) = map.next_key::<String>()? {
                    if key == "vq_codes" {
                        vq_codes = map.next_value::<Option<IgnoredAny>>()?.is_some();
                    } else {
                        let _ignored = map.next_value::<IgnoredAny>()?;
                    }
                }
                Ok(Some(ReferenceFlags {
                    list: true,
                    vq_codes,
                }))
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(None)
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
        deserializer.deserialize_any(ReferenceVisitor)
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
    while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
    Ok(())
}
