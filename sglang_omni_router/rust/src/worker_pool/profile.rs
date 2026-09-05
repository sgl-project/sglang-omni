use std::collections::HashSet;

use serde::Deserialize;

use crate::error::ConfigError;

pub(super) const MAX_WORKERS: usize = 256;
const MAX_PROFILES_PER_WORKER: usize = 64;
const MAX_SET_ITEMS: usize = 64;
const MAX_ID_BYTES: usize = 128;
const MAX_MODEL_ID_BYTES: usize = 256;
const MAX_BASE_URL_BYTES: usize = 2_048;
const MAX_HEALTH_PATH_BYTES: usize = 128;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ServiceClass {
    GenerationHttp,
    SpeechHttp,
    SpeechBatch,
    TranscriptionHttp,
    SpeechWebsocket,
    RealtimeWebsocket,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CapacityClass {
    GenerationHttp,
    SpeechHttp,
    SpeechBatch,
    TranscriptionHttp,
    SpeechWebsocket,
    RealtimeWebsocket,
}

pub(super) const CAPACITY_CLASS_COUNT: usize = 6;

impl CapacityClass {
    pub(super) const fn index(self) -> usize {
        match self {
            Self::GenerationHttp => 0,
            Self::SpeechHttp => 1,
            Self::SpeechBatch => 2,
            Self::TranscriptionHttp => 3,
            Self::SpeechWebsocket => 4,
            Self::RealtimeWebsocket => 5,
        }
    }
}

impl ServiceClass {
    pub(crate) const fn capacity(self) -> CapacityClass {
        match self {
            Self::GenerationHttp => CapacityClass::GenerationHttp,
            Self::SpeechHttp => CapacityClass::SpeechHttp,
            Self::SpeechBatch => CapacityClass::SpeechBatch,
            Self::TranscriptionHttp => CapacityClass::TranscriptionHttp,
            Self::SpeechWebsocket => CapacityClass::SpeechWebsocket,
            Self::RealtimeWebsocket => CapacityClass::RealtimeWebsocket,
        }
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct WorkerId(String);

impl WorkerId {
    pub(super) fn new(value: String) -> Self {
        Self(value)
    }

    pub(super) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct RegistrationId(usize);

impl RegistrationId {
    pub(super) const fn from_startup_ordinal(ordinal: usize) -> Self {
        Self(ordinal)
    }

    pub(super) const fn startup_ordinal(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TrustDomain(String);

impl TrustDomain {
    pub(crate) fn new(value: String) -> Self {
        Self(value)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerConfig {
    pub(crate) worker_id: String,
    pub(crate) base_url: String,
    pub(crate) trust_domain: String,
    pub(crate) default_model_id: Option<String>,
    #[serde(default = "default_health_path")]
    pub(crate) health_path: String,
    #[serde(default)]
    pub(crate) capacity: WorkerCapacityConfig,
    pub(crate) service_profiles: Vec<ServiceProfile>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(crate) struct WorkerCapacityConfig {
    pub(crate) speech_websocket: Option<u32>,
    pub(crate) realtime_websocket: Option<u32>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MessageContentForm {
    String,
    TypedParts,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MediaPlacement {
    TopLevel,
    TypedParts,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ChatAudioFormat {
    Wav,
    Mp3,
    Flac,
    Pcm,
    Aac,
    Opus,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum InputModality {
    Text,
    Image,
    Audio,
    Video,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum OutputModality {
    Text,
    Audio,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StreamMode {
    NonStreaming,
    Streaming,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SpeechResponseFormat {
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SpeechTask {
    TextToSpeech,
    VoiceClone,
    VoiceDesign,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReferenceForm {
    None,
    Direct,
    List,
    VqCodes,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum VoiceNamePolicy {
    Preset,
    Uploaded,
}

impl VoiceNamePolicy {
    pub(super) const fn bit(self) -> u8 {
        match self {
            Self::Preset => 1,
            Self::Uploaded => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SpeechToTextTask {
    Transcribe,
    Translate,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TranscriptionResponseFormat {
    Json,
    Text,
    VerboseJson,
    Srt,
    Vtt,
    Sse,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(tag = "service", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ServiceProfile {
    GenerationHttp {
        model_ids: Vec<String>,
        message_content_forms: Vec<MessageContentForm>,
        media_placements: Vec<MediaPlacement>,
        input_modalities: Vec<InputModality>,
        output_modalities: Vec<OutputModality>,
        chat_audio_formats: Vec<ChatAudioFormat>,
        stream_modes: Vec<StreamMode>,
    },
    SpeechHttp {
        model_ids: Vec<String>,
        response_formats: Vec<SpeechResponseFormat>,
        stream_modes: Vec<StreamMode>,
        tasks: Vec<SpeechTask>,
        reference_forms: Vec<ReferenceForm>,
        voice_name_policy: VoiceNamePolicy,
    },
    SpeechBatch {
        model_ids: Vec<String>,
        response_formats: Vec<SpeechResponseFormat>,
        tasks: Vec<SpeechTask>,
        reference_forms: Vec<ReferenceForm>,
        voice_name_policy: VoiceNamePolicy,
        max_batch_size: u16,
    },
    TranscriptionHttp {
        model_ids: Vec<String>,
        task: SpeechToTextTask,
        response_formats: Vec<TranscriptionResponseFormat>,
        stream_modes: Vec<StreamMode>,
    },
    SpeechWebsocket {
        model_ids: Vec<String>,
        response_formats: Vec<SpeechResponseFormat>,
        stream_modes: Vec<StreamMode>,
        tasks: Vec<SpeechTask>,
        reference_forms: Vec<ReferenceForm>,
        voice_name_policy: VoiceNamePolicy,
    },
    RealtimeWebsocket,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RouteRequirement {
    pub(super) profile: ProfileRequirement,
    trust_domain: TrustDomain,
}

/// One correlated worker requirement; matching must never combine profile rows.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ProfileRequirement {
    GenerationHttp {
        model: ModelSelection,
        message_content_forms: Vec<MessageContentForm>,
        media_placements: Vec<MediaPlacement>,
        input_modalities: Vec<InputModality>,
        output_modalities: Vec<OutputModality>,
        audio_format: Option<ChatAudioFormat>,
        stream_mode: StreamMode,
    },
    SpeechHttp {
        model: ModelSelection,
        response_format: SpeechResponseFormat,
        stream_mode: StreamMode,
        task: Option<SpeechTask>,
        reference_forms: Vec<ReferenceForm>,
        named_voice: bool,
    },
    SpeechBatch {
        models: Vec<ModelSelection>,
        response_formats: Vec<SpeechResponseFormat>,
        tasks: Vec<SpeechTask>,
        reference_forms: Vec<ReferenceForm>,
        named_voice: bool,
        batch_size: u16,
    },
    TranscriptionHttp {
        model: ModelSelection,
        task: SpeechToTextTask,
        response_format: TranscriptionResponseFormat,
        stream_mode: StreamMode,
    },
    SpeechWebsocket {
        model: ModelSelection,
        response_format: Option<SpeechResponseFormat>,
        stream_mode: StreamMode,
        task: Option<SpeechTask>,
        reference_forms: Vec<ReferenceForm>,
        named_voice: bool,
    },
    RealtimeWebsocket {
        model: Option<String>,
    },
}

/// Preserves whether the caller selected a model or relied on a worker default.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum ModelSelection {
    Explicit(String),
    WorkerDefault { expected_model_id: String },
    UnresolvedDefault,
}

impl ModelSelection {
    #[cfg(test)]
    pub(crate) fn expected_model_id(&self) -> Option<&str> {
        match self {
            Self::Explicit(model_id) => Some(model_id),
            Self::WorkerDefault { expected_model_id } => Some(expected_model_id),
            Self::UnresolvedDefault => None,
        }
    }

    fn matches_profile_models(&self, model_ids: &[String], worker_default: Option<&str>) -> bool {
        match self {
            Self::Explicit(model_id) => model_ids.iter().any(|candidate| candidate == model_id),
            Self::WorkerDefault { expected_model_id } => {
                worker_default == Some(expected_model_id.as_str())
                    && model_ids
                        .iter()
                        .any(|candidate| candidate == expected_model_id)
            }
            Self::UnresolvedDefault => worker_default
                .is_none_or(|model_id| model_ids.iter().any(|candidate| candidate == model_id)),
        }
    }

    fn matches_translation(&self, model_ids: &[String], worker_default: Option<&str>) -> bool {
        match self {
            Self::Explicit(model_id)
            | Self::WorkerDefault {
                expected_model_id: model_id,
            } => {
                worker_default == Some(model_id.as_str())
                    && model_ids.iter().any(|candidate| candidate == model_id)
            }
            Self::UnresolvedDefault => worker_default
                .is_some_and(|model_id| model_ids.iter().any(|candidate| candidate == model_id)),
        }
    }

    const fn requires_resolution(&self) -> bool {
        matches!(self, Self::UnresolvedDefault)
    }
}

impl RouteRequirement {
    pub(crate) fn new(profile: ProfileRequirement, trust_domain: TrustDomain) -> Self {
        Self {
            profile,
            trust_domain,
        }
    }

    pub(super) fn trust_domain(&self) -> &TrustDomain {
        &self.trust_domain
    }

    #[cfg(test)]
    pub(crate) fn profile(&self) -> &ProfileRequirement {
        &self.profile
    }

    pub(super) fn capacity_class(&self) -> CapacityClass {
        self.profile.service_class().capacity()
    }
}

impl ProfileRequirement {
    pub(super) const fn service_class(&self) -> ServiceClass {
        match self {
            Self::GenerationHttp { .. } => ServiceClass::GenerationHttp,
            Self::SpeechHttp { .. } => ServiceClass::SpeechHttp,
            Self::SpeechBatch { .. } => ServiceClass::SpeechBatch,
            Self::TranscriptionHttp { .. } => ServiceClass::TranscriptionHttp,
            Self::SpeechWebsocket { .. } => ServiceClass::SpeechWebsocket,
            Self::RealtimeWebsocket { .. } => ServiceClass::RealtimeWebsocket,
        }
    }

    pub(super) fn requires_default_resolution(&self) -> bool {
        match self {
            Self::GenerationHttp { model, .. }
            | Self::SpeechHttp { model, .. }
            | Self::TranscriptionHttp { model, .. }
            | Self::SpeechWebsocket { model, .. } => model.requires_resolution(),
            Self::SpeechBatch { models, .. } => {
                models.iter().any(ModelSelection::requires_resolution)
            }
            Self::RealtimeWebsocket { .. } => false,
        }
    }

    pub(super) const fn has_named_voice(&self) -> bool {
        match self {
            Self::SpeechHttp { named_voice, .. }
            | Self::SpeechBatch { named_voice, .. }
            | Self::SpeechWebsocket { named_voice, .. } => *named_voice,
            Self::GenerationHttp { .. }
            | Self::TranscriptionHttp { .. }
            | Self::RealtimeWebsocket { .. } => false,
        }
    }
}

impl ServiceProfile {
    pub(super) fn validate(&self) -> Result<(), ConfigError> {
        match self {
            Self::GenerationHttp {
                model_ids,
                message_content_forms,
                media_placements,
                input_modalities,
                output_modalities,
                chat_audio_formats,
                stream_modes,
            } => {
                validate_models(model_ids)?;
                validate_set(
                    message_content_forms,
                    "workers.service_profiles.message_content_forms",
                    false,
                )?;
                validate_set(
                    media_placements,
                    "workers.service_profiles.media_placements",
                    true,
                )?;
                validate_set(
                    input_modalities,
                    "workers.service_profiles.input_modalities",
                    false,
                )?;
                validate_set(
                    output_modalities,
                    "workers.service_profiles.output_modalities",
                    false,
                )?;
                validate_set(
                    chat_audio_formats,
                    "workers.service_profiles.chat_audio_formats",
                    true,
                )?;
                validate_set(stream_modes, "workers.service_profiles.stream_modes", false)?;
                let supports_audio = output_modalities.contains(&OutputModality::Audio);
                let has_audio_formats = !chat_audio_formats.is_empty();
                if supports_audio != has_audio_formats {
                    return Err(ConfigError::invalid(
                        "workers.service_profiles.chat_audio_formats",
                        "must be nonempty exactly when audio output is supported",
                    ));
                }
                Ok(())
            }
            Self::SpeechHttp {
                model_ids,
                response_formats,
                stream_modes,
                tasks,
                reference_forms,
                ..
            } => {
                validate_models(model_ids)?;
                validate_set(
                    response_formats,
                    "workers.service_profiles.response_formats",
                    false,
                )?;
                validate_set(stream_modes, "workers.service_profiles.stream_modes", false)?;
                validate_set(tasks, "workers.service_profiles.tasks", false)?;
                validate_set(
                    reference_forms,
                    "workers.service_profiles.reference_forms",
                    false,
                )?;
                if stream_modes.contains(&StreamMode::Streaming)
                    && (response_formats.len() != 1
                        || response_formats[0] != SpeechResponseFormat::Pcm)
                {
                    return Err(ConfigError::invalid(
                        "workers.service_profiles.response_formats",
                        "a streaming speech row may contain only pcm",
                    ));
                }
                Ok(())
            }
            Self::SpeechBatch {
                model_ids,
                response_formats,
                tasks,
                reference_forms,
                max_batch_size,
                ..
            } => {
                validate_models(model_ids)?;
                validate_set(
                    response_formats,
                    "workers.service_profiles.response_formats",
                    false,
                )?;
                validate_set(tasks, "workers.service_profiles.tasks", false)?;
                validate_set(
                    reference_forms,
                    "workers.service_profiles.reference_forms",
                    false,
                )?;
                if *max_batch_size == 0 {
                    return Err(ConfigError::invalid(
                        "workers.service_profiles.max_batch_size",
                        "must be positive",
                    ));
                }
                Ok(())
            }
            Self::TranscriptionHttp {
                model_ids,
                response_formats,
                stream_modes,
                ..
            } => {
                validate_models(model_ids)?;
                validate_set(
                    response_formats,
                    "workers.service_profiles.response_formats",
                    false,
                )?;
                validate_set(stream_modes, "workers.service_profiles.stream_modes", false)?;
                if response_formats.contains(&TranscriptionResponseFormat::Sse)
                    != stream_modes.contains(&StreamMode::Streaming)
                {
                    return Err(ConfigError::invalid(
                        "workers.service_profiles.response_formats",
                        "sse support must match streaming support",
                    ));
                }
                Ok(())
            }
            Self::SpeechWebsocket {
                model_ids,
                response_formats,
                stream_modes,
                tasks,
                reference_forms,
                ..
            } => {
                validate_models(model_ids)?;
                validate_set(
                    response_formats,
                    "workers.service_profiles.response_formats",
                    false,
                )?;
                validate_set(stream_modes, "workers.service_profiles.stream_modes", false)?;
                validate_set(tasks, "workers.service_profiles.tasks", false)?;
                validate_set(
                    reference_forms,
                    "workers.service_profiles.reference_forms",
                    false,
                )?;
                if stream_modes.contains(&StreamMode::Streaming)
                    && (response_formats.len() != 1
                        || response_formats[0] != SpeechResponseFormat::Pcm)
                {
                    return Err(ConfigError::invalid(
                        "workers.service_profiles.response_formats",
                        "a streaming speech row may contain only pcm",
                    ));
                }
                Ok(())
            }
            Self::RealtimeWebsocket => Ok(()),
        }
    }

    pub(super) fn semantically_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::GenerationHttp {
                    model_ids: a_models,
                    message_content_forms: a_forms,
                    media_placements: a_placements,
                    input_modalities: a_inputs,
                    output_modalities: a_outputs,
                    chat_audio_formats: a_audio,
                    stream_modes: a_streams,
                },
                Self::GenerationHttp {
                    model_ids: b_models,
                    message_content_forms: b_forms,
                    media_placements: b_placements,
                    input_modalities: b_inputs,
                    output_modalities: b_outputs,
                    chat_audio_formats: b_audio,
                    stream_modes: b_streams,
                },
            ) => {
                set_eq(a_models, b_models)
                    && set_eq(a_forms, b_forms)
                    && set_eq(a_placements, b_placements)
                    && set_eq(a_inputs, b_inputs)
                    && set_eq(a_outputs, b_outputs)
                    && set_eq(a_audio, b_audio)
                    && set_eq(a_streams, b_streams)
            }
            (
                Self::SpeechHttp {
                    model_ids: am,
                    response_formats: af,
                    stream_modes: asm,
                    tasks: at,
                    reference_forms: ar,
                    voice_name_policy: av,
                },
                Self::SpeechHttp {
                    model_ids: bm,
                    response_formats: bf,
                    stream_modes: bsm,
                    tasks: bt,
                    reference_forms: br,
                    voice_name_policy: bv,
                },
            ) => {
                av == bv
                    && set_eq(am, bm)
                    && set_eq(af, bf)
                    && set_eq(asm, bsm)
                    && set_eq(at, bt)
                    && set_eq(ar, br)
            }
            (
                Self::SpeechBatch {
                    model_ids: am,
                    response_formats: af,
                    tasks: at,
                    reference_forms: ar,
                    voice_name_policy: av,
                    max_batch_size: ab,
                },
                Self::SpeechBatch {
                    model_ids: bm,
                    response_formats: bf,
                    tasks: bt,
                    reference_forms: br,
                    voice_name_policy: bv,
                    max_batch_size: bb,
                },
            ) => {
                av == bv
                    && ab == bb
                    && set_eq(am, bm)
                    && set_eq(af, bf)
                    && set_eq(at, bt)
                    && set_eq(ar, br)
            }
            (
                Self::TranscriptionHttp {
                    model_ids: am,
                    task: at,
                    response_formats: af,
                    stream_modes: asm,
                },
                Self::TranscriptionHttp {
                    model_ids: bm,
                    task: bt,
                    response_formats: bf,
                    stream_modes: bsm,
                },
            ) => at == bt && set_eq(am, bm) && set_eq(af, bf) && set_eq(asm, bsm),
            (
                Self::SpeechWebsocket {
                    model_ids: am,
                    response_formats: af,
                    stream_modes: asm,
                    tasks: at,
                    reference_forms: ar,
                    voice_name_policy: av,
                },
                Self::SpeechWebsocket {
                    model_ids: bm,
                    response_formats: bf,
                    stream_modes: bsm,
                    tasks: bt,
                    reference_forms: br,
                    voice_name_policy: bv,
                },
            ) => {
                av == bv
                    && set_eq(am, bm)
                    && set_eq(af, bf)
                    && set_eq(asm, bsm)
                    && set_eq(at, bt)
                    && set_eq(ar, br)
            }
            (Self::RealtimeWebsocket, Self::RealtimeWebsocket) => true,
            _ => false,
        }
    }

    pub(super) fn matches(
        &self,
        requirement: &ProfileRequirement,
        worker_default: Option<&str>,
    ) -> bool {
        match (self, requirement) {
            (
                Self::GenerationHttp {
                    model_ids,
                    message_content_forms,
                    media_placements,
                    input_modalities,
                    output_modalities,
                    chat_audio_formats,
                    stream_modes,
                },
                ProfileRequirement::GenerationHttp {
                    model,
                    message_content_forms: required_forms,
                    media_placements: required_placements,
                    input_modalities: required_inputs,
                    output_modalities: required_outputs,
                    audio_format,
                    stream_mode,
                },
            ) => {
                model.matches_profile_models(model_ids, worker_default)
                    && contains_all(message_content_forms, required_forms)
                    && contains_all(media_placements, required_placements)
                    && contains_all(input_modalities, required_inputs)
                    && contains_all(output_modalities, required_outputs)
                    && audio_format.is_none_or(|format| chat_audio_formats.contains(&format))
                    && stream_modes.contains(stream_mode)
            }
            (
                Self::SpeechHttp {
                    model_ids,
                    response_formats,
                    stream_modes,
                    tasks,
                    reference_forms,
                    voice_name_policy,
                },
                ProfileRequirement::SpeechHttp {
                    model,
                    response_format,
                    stream_mode,
                    task,
                    reference_forms: required_references,
                    named_voice,
                },
            ) => {
                model.matches_profile_models(model_ids, worker_default)
                    && response_formats.contains(response_format)
                    && stream_modes.contains(stream_mode)
                    && task.is_none_or(|task| tasks.contains(&task))
                    && matches_speech_references(
                        reference_forms,
                        required_references,
                        *named_voice,
                        *voice_name_policy,
                    )
            }
            (
                Self::SpeechBatch {
                    model_ids,
                    response_formats,
                    tasks,
                    reference_forms,
                    voice_name_policy,
                    max_batch_size,
                },
                ProfileRequirement::SpeechBatch {
                    models,
                    response_formats: required_formats,
                    tasks: required_tasks,
                    reference_forms: required_references,
                    named_voice,
                    batch_size,
                },
            ) => {
                models
                    .iter()
                    .all(|model| model.matches_profile_models(model_ids, worker_default))
                    && *batch_size <= *max_batch_size
                    && contains_all(response_formats, required_formats)
                    && contains_all(tasks, required_tasks)
                    && matches_speech_references(
                        reference_forms,
                        required_references,
                        *named_voice,
                        *voice_name_policy,
                    )
            }
            (
                Self::TranscriptionHttp {
                    model_ids,
                    task,
                    response_formats,
                    stream_modes,
                },
                ProfileRequirement::TranscriptionHttp {
                    model,
                    task: required_task,
                    response_format,
                    stream_mode,
                },
            ) => {
                (if *required_task == SpeechToTextTask::Translate {
                    model.matches_translation(model_ids, worker_default)
                } else {
                    model.matches_profile_models(model_ids, worker_default)
                }) && task == required_task
                    && response_formats.contains(response_format)
                    && stream_modes.contains(stream_mode)
            }
            (
                Self::SpeechWebsocket {
                    model_ids,
                    response_formats,
                    stream_modes,
                    tasks,
                    reference_forms,
                    voice_name_policy,
                },
                ProfileRequirement::SpeechWebsocket {
                    model,
                    response_format,
                    stream_mode,
                    task,
                    reference_forms: required_references,
                    named_voice,
                },
            ) => {
                model.matches_profile_models(model_ids, worker_default)
                    && response_format.is_none_or(|format| response_formats.contains(&format))
                    && stream_modes.contains(stream_mode)
                    && task.is_none_or(|task| tasks.contains(&task))
                    && matches_speech_references(
                        reference_forms,
                        required_references,
                        *named_voice,
                        *voice_name_policy,
                    )
            }
            (Self::RealtimeWebsocket, ProfileRequirement::RealtimeWebsocket { model }) => model
                .as_deref()
                .is_none_or(|model| worker_default == Some(model)),
            _ => false,
        }
    }

    fn contains_model(&self, model: &str) -> bool {
        match self {
            Self::GenerationHttp { model_ids, .. }
            | Self::SpeechHttp { model_ids, .. }
            | Self::SpeechBatch { model_ids, .. }
            | Self::TranscriptionHttp { model_ids, .. }
            | Self::SpeechWebsocket { model_ids, .. } => model_ids.iter().any(|item| item == model),
            Self::RealtimeWebsocket => true,
        }
    }

    pub(crate) const fn service_class(&self) -> ServiceClass {
        match self {
            Self::GenerationHttp { .. } => ServiceClass::GenerationHttp,
            Self::SpeechHttp { .. } => ServiceClass::SpeechHttp,
            Self::SpeechBatch { .. } => ServiceClass::SpeechBatch,
            Self::TranscriptionHttp { .. } => ServiceClass::TranscriptionHttp,
            Self::SpeechWebsocket { .. } => ServiceClass::SpeechWebsocket,
            Self::RealtimeWebsocket => ServiceClass::RealtimeWebsocket,
        }
    }

    pub(super) const fn voice_name_policy(&self) -> Option<VoiceNamePolicy> {
        match self {
            Self::SpeechHttp {
                voice_name_policy, ..
            }
            | Self::SpeechBatch {
                voice_name_policy, ..
            }
            | Self::SpeechWebsocket {
                voice_name_policy, ..
            } => Some(*voice_name_policy),
            Self::GenerationHttp { .. }
            | Self::TranscriptionHttp { .. }
            | Self::RealtimeWebsocket => None,
        }
    }
}

pub(crate) fn validate_workers(workers: &[WorkerConfig]) -> Result<(), ConfigError> {
    if workers.is_empty() || workers.len() > MAX_WORKERS {
        return Err(ConfigError::invalid(
            "workers",
            "must contain between 1 and 256 workers",
        ));
    }
    let mut ids = HashSet::with_capacity(workers.len());
    let mut targets = HashSet::with_capacity(workers.len());
    for worker in workers {
        validate_identifier(&worker.worker_id, "workers.worker_id")?;
        if !ids.insert(worker.worker_id.as_str()) {
            return Err(ConfigError::invalid("workers.worker_id", "must be unique"));
        }
        validate_identifier(&worker.trust_domain, "workers.trust_domain")?;
        if worker
            .default_model_id
            .as_deref()
            .is_some_and(|model| !valid_model_id(model))
        {
            return Err(ConfigError::invalid(
                "workers.default_model_id",
                "must be 1 to 256 bytes",
            ));
        }
        if worker.base_url.is_empty() || worker.base_url.len() > MAX_BASE_URL_BYTES {
            return Err(ConfigError::invalid(
                "workers.base_url",
                "must contain between 1 and 2048 bytes",
            ));
        }
        if worker.health_path.is_empty()
            || worker.health_path.len() > MAX_HEALTH_PATH_BYTES
            || !worker.health_path.starts_with('/')
            || worker.health_path.contains('?')
            || worker.health_path.contains('#')
        {
            return Err(ConfigError::invalid(
                "workers.health_path",
                "must be an absolute path without query or fragment",
            ));
        }
        let target = super::resolver::ResolvedTarget::from_worker(worker).ok_or_else(|| {
            ConfigError::invalid(
                "workers.base_url",
                "must be a canonical HTTP or HTTPS origin",
            )
        })?;
        if !targets.insert(target.base_url().as_str().to_owned()) {
            return Err(ConfigError::invalid(
                "workers.base_url",
                "worker origins must be unique",
            ));
        }
        for (field, value) in [
            (
                "workers.capacity.speech_websocket",
                worker.capacity.speech_websocket,
            ),
            (
                "workers.capacity.realtime_websocket",
                worker.capacity.realtime_websocket,
            ),
        ] {
            if value.is_some_and(|limit| limit == 0 || limit > 65_535) {
                return Err(ConfigError::invalid(field, "must be between 1 and 65535"));
            }
        }
        if worker.service_profiles.is_empty()
            || worker.service_profiles.len() > MAX_PROFILES_PER_WORKER
        {
            return Err(ConfigError::invalid(
                "workers.service_profiles",
                "must contain between 1 and 64 rows",
            ));
        }
        for (index, profile) in worker.service_profiles.iter().enumerate() {
            profile.validate()?;
            if worker
                .service_profiles
                .iter()
                .take(index)
                .any(|earlier| profile.semantically_eq(earlier))
            {
                return Err(ConfigError::invalid(
                    "workers.service_profiles",
                    "contains a duplicate correlated row",
                ));
            }
        }
        for profile in &worker.service_profiles {
            let configured = match profile.service_class() {
                ServiceClass::SpeechWebsocket => worker.capacity.speech_websocket,
                ServiceClass::RealtimeWebsocket => worker.capacity.realtime_websocket,
                ServiceClass::GenerationHttp
                | ServiceClass::SpeechHttp
                | ServiceClass::SpeechBatch
                | ServiceClass::TranscriptionHttp => continue,
            };
            if configured.is_none() {
                return Err(ConfigError::invalid(
                    "workers.capacity",
                    "must configure capacity for every WebSocket service profile",
                ));
            }
        }
        if let Some(default) = worker.default_model_id.as_deref() {
            for advertised in &worker.service_profiles {
                let service = advertised.service_class();
                if !worker.service_profiles.iter().any(|profile| {
                    profile.service_class() == service
                        && match (advertised, profile) {
                            (
                                ServiceProfile::TranscriptionHttp {
                                    task: advertised_task,
                                    ..
                                },
                                ServiceProfile::TranscriptionHttp {
                                    task: candidate_task,
                                    ..
                                },
                            ) => advertised_task == candidate_task,
                            (ServiceProfile::TranscriptionHttp { .. }, _)
                            | (_, ServiceProfile::TranscriptionHttp { .. }) => false,
                            _ => true,
                        }
                        && profile.contains_model(default)
                }) {
                    return Err(ConfigError::invalid(
                        "workers.default_model_id",
                        "must belong to every advertised model-executing service and task",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_models(values: &[String]) -> Result<(), ConfigError> {
    if values.is_empty()
        || values.len() > MAX_SET_ITEMS
        || values.iter().any(|value| !valid_model_id(value))
    {
        return Err(ConfigError::invalid(
            "workers.service_profiles.model_ids",
            "must contain 1 to 64 unique model IDs",
        ));
    }
    let unique: HashSet<_> = values.iter().collect();
    if unique.len() != values.len() {
        return Err(ConfigError::invalid(
            "workers.service_profiles.model_ids",
            "must not contain duplicates",
        ));
    }
    Ok(())
}

fn validate_set<T: Eq + std::hash::Hash>(
    values: &[T],
    field: &'static str,
    allow_empty: bool,
) -> Result<(), ConfigError> {
    if values.len() > MAX_SET_ITEMS || (!allow_empty && values.is_empty()) {
        return Err(ConfigError::invalid(field, "has an invalid item count"));
    }
    let unique: HashSet<_> = values.iter().collect();
    if unique.len() != values.len() {
        return Err(ConfigError::invalid(field, "must not contain duplicates"));
    }
    Ok(())
}

fn set_eq<T: Eq>(left: &[T], right: &[T]) -> bool {
    left.len() == right.len() && left.iter().all(|item| right.contains(item))
}

fn contains_all<T: Eq>(available: &[T], required: &[T]) -> bool {
    required.iter().all(|item| available.contains(item))
}

fn matches_speech_references(
    available: &[ReferenceForm],
    required: &[ReferenceForm],
    named_voice: bool,
    voice_name_policy: VoiceNamePolicy,
) -> bool {
    contains_all(available, required)
        && (!named_voice
            || voice_name_policy == VoiceNamePolicy::Uploaded
            || available.contains(&ReferenceForm::None))
}

pub(crate) fn validate_identifier(value: &str, field: &'static str) -> Result<(), ConfigError> {
    if value.is_empty()
        || value.len() > MAX_ID_BYTES
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(ConfigError::invalid(
            field,
            "must be 1 to 128 ASCII identifier bytes",
        ));
    }
    Ok(())
}

pub(crate) fn valid_model_id(value: &str) -> bool {
    !value.is_empty() && value.len() <= MAX_MODEL_ID_BYTES && !value.chars().any(char::is_control)
}

fn default_health_path() -> String {
    String::from("/health")
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn profile(model: &str) -> ServiceProfile {
        ServiceProfile::GenerationHttp {
            model_ids: vec![model.to_owned()],
            message_content_forms: vec![MessageContentForm::String],
            media_placements: Vec::new(),
            input_modalities: vec![InputModality::Text],
            output_modalities: vec![OutputModality::Text],
            chat_audio_formats: Vec::new(),
            stream_modes: vec![StreamMode::NonStreaming],
        }
    }

    fn worker() -> WorkerConfig {
        WorkerConfig {
            worker_id: String::from("worker-a"),
            base_url: String::from("http://127.0.0.1:8000/"),
            trust_domain: String::from("local"),
            default_model_id: Some(String::from("omni")),
            health_path: String::from("/health"),
            capacity: WorkerCapacityConfig::default(),
            service_profiles: vec![profile("omni")],
        }
    }

    #[test]
    fn stable_worker_shape_and_correlated_generation_row_validate() {
        assert!(validate_workers(&[worker()]).is_ok());
        let mut maximum = Vec::new();
        for index in 0..MAX_WORKERS {
            let mut item = worker();
            item.worker_id = format!("worker-{index}");
            item.base_url = format!("http://127.0.0.1:{}/", 10_000 + index);
            maximum.push(item);
        }
        assert!(validate_workers(&maximum).is_ok());
    }

    #[test]
    fn strict_profile_and_default_correlation_fail_closed() {
        let mut missing_default = worker();
        missing_default.default_model_id = Some(String::from("other"));
        assert!(validate_workers(&[missing_default]).is_err());

        let mut duplicate = worker();
        duplicate.service_profiles.push(profile("omni"));
        assert!(validate_workers(&[duplicate]).is_err());

        let mut invalid_audio = worker();
        invalid_audio.service_profiles = vec![ServiceProfile::GenerationHttp {
            model_ids: vec![String::from("omni")],
            message_content_forms: vec![MessageContentForm::TypedParts],
            media_placements: vec![MediaPlacement::TypedParts],
            input_modalities: vec![InputModality::Audio],
            output_modalities: vec![OutputModality::Audio],
            chat_audio_formats: Vec::new(),
            stream_modes: vec![StreamMode::Streaming],
        }];
        assert!(validate_workers(&[invalid_audio]).is_err());
    }

    #[test]
    fn streaming_speech_rows_are_pcm_only() {
        let speech = |response_formats| ServiceProfile::SpeechHttp {
            model_ids: vec![String::from("tts")],
            response_formats,
            stream_modes: vec![StreamMode::NonStreaming, StreamMode::Streaming],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::None],
            voice_name_policy: VoiceNamePolicy::Preset,
        };
        assert!(speech(vec![SpeechResponseFormat::Pcm]).validate().is_ok());
        assert!(
            speech(vec![SpeechResponseFormat::Pcm, SpeechResponseFormat::Wav])
                .validate()
                .is_err()
        );
        let websocket = |response_formats| ServiceProfile::SpeechWebsocket {
            model_ids: vec![String::from("tts")],
            response_formats,
            stream_modes: vec![StreamMode::NonStreaming, StreamMode::Streaming],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::None],
            voice_name_policy: VoiceNamePolicy::Preset,
        };
        assert!(
            websocket(vec![SpeechResponseFormat::Pcm])
                .validate()
                .is_ok()
        );
        assert!(
            websocket(vec![SpeechResponseFormat::Pcm, SpeechResponseFormat::Wav])
                .validate()
                .is_err()
        );
    }

    #[test]
    fn worker_default_is_valid_for_every_advertised_service() {
        let speech = |models| ServiceProfile::SpeechHttp {
            model_ids: models,
            response_formats: vec![SpeechResponseFormat::Wav],
            stream_modes: vec![StreamMode::NonStreaming],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::None],
            voice_name_policy: VoiceNamePolicy::Preset,
        };
        let mut missing_service_default = worker();
        missing_service_default
            .service_profiles
            .push(speech(vec![String::from("other")]));
        assert!(validate_workers(&[missing_service_default]).is_err());

        let mut correlated_default = worker();
        correlated_default
            .service_profiles
            .push(speech(vec![String::from("omni"), String::from("other")]));
        assert!(validate_workers(&[correlated_default]).is_ok());

        let transcription = |task, models| ServiceProfile::TranscriptionHttp {
            model_ids: models,
            task,
            response_formats: vec![TranscriptionResponseFormat::Json],
            stream_modes: vec![StreamMode::NonStreaming],
        };
        let mut mismatched_task_default = worker();
        mismatched_task_default.service_profiles.extend([
            transcription(SpeechToTextTask::Transcribe, vec![String::from("omni")]),
            transcription(SpeechToTextTask::Translate, vec![String::from("other")]),
        ]);
        assert!(validate_workers(&[mismatched_task_default]).is_err());
    }

    #[test]
    fn matching_never_combines_correlated_rows() {
        let text = profile("omni");
        let audio = ServiceProfile::GenerationHttp {
            model_ids: vec![String::from("audio")],
            message_content_forms: vec![MessageContentForm::TypedParts],
            media_placements: vec![MediaPlacement::TypedParts],
            input_modalities: vec![InputModality::Audio],
            output_modalities: vec![OutputModality::Audio],
            chat_audio_formats: vec![ChatAudioFormat::Wav],
            stream_modes: vec![StreamMode::Streaming],
        };
        let cross_row = ProfileRequirement::GenerationHttp {
            model: ModelSelection::Explicit(String::from("omni")),
            message_content_forms: vec![MessageContentForm::TypedParts],
            media_placements: vec![MediaPlacement::TypedParts],
            input_modalities: vec![InputModality::Audio],
            output_modalities: vec![OutputModality::Audio],
            audio_format: Some(ChatAudioFormat::Wav),
            stream_mode: StreamMode::Streaming,
        };
        assert!(!text.matches(&cross_row, Some("omni")));
        assert!(!audio.matches(&cross_row, Some("audio")));
    }

    #[test]
    fn speech_to_text_task_is_singular_and_media_only_workers_need_no_generation_shape() {
        let profile = |task| ServiceProfile::TranscriptionHttp {
            model_ids: vec![String::from("asr")],
            task,
            response_formats: vec![TranscriptionResponseFormat::Json],
            stream_modes: vec![StreamMode::NonStreaming],
        };
        let requirement = ProfileRequirement::TranscriptionHttp {
            model: ModelSelection::WorkerDefault {
                expected_model_id: String::from("asr"),
            },
            task: SpeechToTextTask::Translate,
            response_format: TranscriptionResponseFormat::Json,
            stream_mode: StreamMode::NonStreaming,
        };
        assert!(!profile(SpeechToTextTask::Transcribe).matches(&requirement, Some("asr")));
        assert!(profile(SpeechToTextTask::Translate).matches(&requirement, Some("asr")));

        let unresolved = ProfileRequirement::TranscriptionHttp {
            model: ModelSelection::UnresolvedDefault,
            task: SpeechToTextTask::Translate,
            response_format: TranscriptionResponseFormat::Json,
            stream_mode: StreamMode::NonStreaming,
        };
        assert!(profile(SpeechToTextTask::Translate).matches(&unresolved, Some("asr")));
        assert!(!profile(SpeechToTextTask::Translate).matches(&unresolved, None));

        let alias = ProfileRequirement::TranscriptionHttp {
            model: ModelSelection::Explicit(String::from("alias")),
            task: SpeechToTextTask::Translate,
            response_format: TranscriptionResponseFormat::Json,
            stream_mode: StreamMode::NonStreaming,
        };
        let translated_alias = ServiceProfile::TranscriptionHttp {
            model_ids: vec![String::from("asr"), String::from("alias")],
            task: SpeechToTextTask::Translate,
            response_formats: vec![TranscriptionResponseFormat::Json],
            stream_modes: vec![StreamMode::NonStreaming],
        };
        assert!(
            !translated_alias.matches(&alias, Some("asr")),
            "translation accepts only the worker default model"
        );

        let worker = WorkerConfig {
            worker_id: String::from("media-only"),
            base_url: String::from("http://127.0.0.1:8001/"),
            trust_domain: String::from("local"),
            default_model_id: None,
            health_path: String::from("/health"),
            capacity: WorkerCapacityConfig::default(),
            service_profiles: vec![profile(SpeechToTextTask::Transcribe)],
        };
        assert!(validate_workers(&[worker]).is_ok());
    }

    #[test]
    fn named_voice_policy_controls_the_implicit_reference() {
        let mut row = ServiceProfile::SpeechHttp {
            model_ids: vec![String::from("tts")],
            response_formats: vec![SpeechResponseFormat::Wav],
            stream_modes: vec![StreamMode::NonStreaming],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::Direct],
            voice_name_policy: VoiceNamePolicy::Uploaded,
        };
        let requirement = |reference_forms, named_voice| ProfileRequirement::SpeechHttp {
            model: ModelSelection::Explicit(String::from("tts")),
            response_format: SpeechResponseFormat::Wav,
            stream_mode: StreamMode::NonStreaming,
            task: None,
            reference_forms,
            named_voice,
        };
        assert_eq!(row.voice_name_policy(), Some(VoiceNamePolicy::Uploaded));
        assert!(row.matches(&requirement(Vec::new(), true), Some("tts")));
        assert!(row.matches(
            &requirement(vec![ReferenceForm::Direct], false),
            Some("tts")
        ));
        assert!(!row.matches(&requirement(vec![ReferenceForm::None], false), Some("tts")));

        if let ServiceProfile::SpeechHttp {
            voice_name_policy, ..
        } = &mut row
        {
            *voice_name_policy = VoiceNamePolicy::Preset;
        }
        assert!(!row.matches(&requirement(Vec::new(), true), Some("tts")));
        if let ServiceProfile::SpeechHttp {
            reference_forms, ..
        } = &mut row
        {
            reference_forms.push(ReferenceForm::None);
        }
        assert!(row.matches(&requirement(Vec::new(), true), Some("tts")));
    }

    #[test]
    fn speech_batch_combines_named_voice_and_reference_requirements() {
        let mut row = ServiceProfile::SpeechBatch {
            model_ids: vec![String::from("tts")],
            response_formats: vec![SpeechResponseFormat::Wav],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::None],
            voice_name_policy: VoiceNamePolicy::Uploaded,
            max_batch_size: 2,
        };
        let requirement = ProfileRequirement::SpeechBatch {
            models: vec![ModelSelection::Explicit(String::from("tts"))],
            response_formats: vec![SpeechResponseFormat::Wav],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::Direct],
            named_voice: true,
            batch_size: 2,
        };
        assert!(!row.matches(&requirement, Some("tts")));
        let ServiceProfile::SpeechBatch {
            reference_forms, ..
        } = &mut row
        else {
            unreachable!()
        };
        reference_forms.push(ReferenceForm::Direct);
        assert!(row.matches(&requirement, Some("tts")));

        if let ServiceProfile::SpeechBatch {
            voice_name_policy, ..
        } = &mut row
        {
            *voice_name_policy = VoiceNamePolicy::Preset;
        }
        assert!(row.matches(&requirement, Some("tts")));
        if let ServiceProfile::SpeechBatch {
            reference_forms, ..
        } = &mut row
        {
            reference_forms.retain(|form| *form != ReferenceForm::None);
        }
        assert!(!row.matches(&requirement, Some("tts")));
    }

    #[test]
    fn named_voice_websocket_uses_the_profile_voice_policy() {
        let mut row = ServiceProfile::SpeechWebsocket {
            model_ids: vec![String::from("tts")],
            response_formats: vec![SpeechResponseFormat::Pcm],
            stream_modes: vec![StreamMode::Streaming],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::Direct],
            voice_name_policy: VoiceNamePolicy::Uploaded,
        };
        let requirement = |reference_forms, named_voice| ProfileRequirement::SpeechWebsocket {
            model: ModelSelection::Explicit(String::from("tts")),
            response_format: Some(SpeechResponseFormat::Pcm),
            stream_mode: StreamMode::Streaming,
            task: None,
            reference_forms,
            named_voice,
        };
        assert_eq!(row.voice_name_policy(), Some(VoiceNamePolicy::Uploaded));
        assert!(row.matches(&requirement(Vec::new(), true), Some("tts")));
        assert!(row.matches(
            &requirement(vec![ReferenceForm::Direct], false),
            Some("tts")
        ));
        assert!(!row.matches(&requirement(vec![ReferenceForm::None], false), Some("tts")));

        if let ServiceProfile::SpeechWebsocket {
            voice_name_policy, ..
        } = &mut row
        {
            *voice_name_policy = VoiceNamePolicy::Preset;
        }
        assert!(!row.matches(&requirement(Vec::new(), true), Some("tts")));
        if let ServiceProfile::SpeechWebsocket {
            reference_forms, ..
        } = &mut row
        {
            reference_forms.push(ReferenceForm::None);
        }
        assert!(row.matches(&requirement(Vec::new(), true), Some("tts")));
        assert!(row.matches(
            &requirement(vec![ReferenceForm::Direct], false),
            Some("tts")
        ));
    }
}
