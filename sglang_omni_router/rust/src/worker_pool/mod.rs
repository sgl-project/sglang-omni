mod admission;
mod health;
pub(crate) mod profile;
mod resolver;
mod selection;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::Notify;

use crate::config::{Config, RoutingStrategy};

pub(crate) use admission::{
    AdmissionError, AdmissionLease, DispatchError, EnvelopeLease, RequestLease,
};
pub(crate) use health::{HealthSupervisor, WorkerHealth};
pub(crate) use profile::{
    CapacityClass, ChatAudioFormat, MediaPlacement, MessageContentForm, ModelSelection,
    ProfileRequirement, ReferenceForm, RouteRequirement, ServiceClass, SpeechResponseFormat,
    SpeechTask, SpeechToTextTask, StreamMode, TranscriptionResponseFormat, TrustDomain,
};
pub(crate) use resolver::ResolvedTarget;

use admission::AdmissionController;
use health::AtomicHealth;
use profile::{MAX_WORKERS, RegistrationId, ServiceProfile, WorkerId};
use resolver::{build_health_client, build_http_client};
use selection::{Selector, SelectorGuard};

/// One static startup registration with independently updated health and load.
pub(super) struct WorkerRecord {
    worker_id: WorkerId,
    default_model_id: Option<String>,
    registration_id: RegistrationId,
    target: ResolvedTarget,
    trust_domain: TrustDomain,
    profiles: Vec<ServiceProfile>,
    active_requests: AtomicUsize,
    health: AtomicHealth,
    immediate_probe: Notify,
}

impl WorkerRecord {
    fn profile_match(&self, requirement: &RouteRequirement) -> (bool, u8) {
        let mut matched = false;
        let mut voice_policies = 0;
        for profile in &self.profiles {
            if !profile.matches(&requirement.profile, self.default_model_id.as_deref()) {
                continue;
            }
            matched = true;
            if requirement.profile.has_named_voice()
                && let Some(policy) = profile.voice_name_policy()
            {
                voice_policies |= policy.bit();
            }
        }
        (matched, voice_policies)
    }

    fn is_routable(&self) -> bool {
        self.health.load() == WorkerHealth::Healthy
    }

    fn load(&self) -> usize {
        self.active_requests.load(Ordering::Relaxed)
    }

    fn increment_load(&self, weight: usize) {
        self.active_requests.fetch_add(weight, Ordering::Relaxed);
    }

    fn decrement_load(&self, weight: usize) {
        let previous =
            self.active_requests
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    current.checked_sub(weight)
                });
        debug_assert!(previous.is_ok(), "worker load cannot underflow");
    }
}

/// Static-membership worker pool with bounded admission, deterministic policy
/// state, and independently owned health and weighted load.
pub(crate) struct WorkerPool {
    records: Vec<Arc<WorkerRecord>>,
    admission: AdmissionController,
    selector: Selector,
    homogeneous_generation_http: Vec<HomogeneousGenerationCohort>,
    homogeneous_media_http: Vec<HomogeneousMediaCohort>,
    health_client: reqwest::Client,
    http_client: reqwest::Client,
}

struct HomogeneousGenerationCohort {
    trust_domain: TrustDomain,
}

struct HomogeneousMediaCohort {
    trust_domain: TrustDomain,
    service: ServiceClass,
    task: Option<SpeechToTextTask>,
}

/// Startup proof that chat body inspection cannot change the route cohort.
pub(crate) struct ContentBlindGenerationHttp<'a> {
    pool: &'a WorkerPool,
    trust: &'a TrustDomain,
}

pub(crate) struct ContentBlindMediaHttp<'a> {
    pool: &'a WorkerPool,
    cohort: &'a HomogeneousMediaCohort,
}

impl WorkerPool {
    pub(crate) fn build(config: &Config) -> Result<Self, crate::error::RouterError> {
        let targets: Vec<_> = config
            .workers
            .iter()
            .map(ResolvedTarget::from_worker)
            .collect::<Option<_>>()
            .ok_or(crate::error::RouterError::WorkerPoolInvariant)?;
        let health_client = build_health_client(config.health.timeout(), config.health.interval())
            .map_err(crate::error::RouterError::HealthClient)?;
        let http_client = build_http_client(
            config.http.connect_timeout(),
            config.http.pool_idle_timeout(),
            config.http.pool_max_idle_per_host,
        )
        .map_err(crate::error::RouterError::HttpClient)?;
        let admission_limit = |limit: Option<u32>| {
            limit
                .map(usize::try_from)
                .transpose()
                .map_err(|_| crate::error::RouterError::WorkerPoolInvariant)
        };
        let admission = AdmissionController::new(
            usize::try_from(config.admission.global)
                .map_err(|_| crate::error::RouterError::WorkerPoolInvariant)?,
            [
                admission_limit(config.admission.generation_http)?,
                admission_limit(config.admission.speech_http)?,
                admission_limit(config.admission.speech_batch)?,
                admission_limit(config.admission.transcription_http)?,
            ],
        );
        let mut records = Vec::with_capacity(config.workers.len());
        for (ordinal, (worker, target)) in config.workers.iter().zip(targets).enumerate() {
            records.push(Arc::new(WorkerRecord {
                worker_id: WorkerId::new(worker.worker_id.clone()),
                default_model_id: worker.default_model_id.clone(),
                registration_id: RegistrationId::from_startup_ordinal(ordinal),
                target,
                trust_domain: TrustDomain::new(worker.trust_domain.clone()),
                profiles: worker.service_profiles.clone(),
                active_requests: AtomicUsize::new(0),
                health: AtomicHealth::unknown(),
                immediate_probe: Notify::new(),
            }));
        }
        let homogeneous_generation_http = build_content_blind_generation_cohorts(&records);
        let homogeneous_media_http = build_content_blind_media_cohorts(&records);
        let selector = Selector::new(config.router.strategy, records.len());
        Ok(Self {
            records,
            admission,
            selector,
            homogeneous_generation_http,
            homogeneous_media_http,
            health_client,
            http_client,
        })
    }

    pub(crate) fn start_health(&self, config: &Config) -> HealthSupervisor {
        HealthSupervisor::start(
            &self.records,
            self.health_client.clone(),
            config.health.interval(),
            config.health.success_threshold(),
            config.health.failure_threshold(),
        )
    }

    pub(crate) fn http_client(&self) -> reqwest::Client {
        self.http_client.clone()
    }

    pub(crate) fn try_admit(
        &self,
        class: CapacityClass,
        credits: u32,
    ) -> Result<AdmissionLease, AdmissionError> {
        self.admission.try_admit(class, credits)
    }

    pub(crate) fn try_admit_envelope(&self) -> Result<EnvelopeLease, AdmissionError> {
        self.admission.try_admit_envelope()
    }

    pub(crate) fn try_admit_class(
        &self,
        envelope: EnvelopeLease,
        class: CapacityClass,
        credits: u32,
    ) -> Result<AdmissionLease, AdmissionError> {
        self.admission.try_admit_class(envelope, class, credits)
    }

    pub(crate) fn dispatch(
        &self,
        admission: AdmissionLease,
        requirement: &RouteRequirement,
    ) -> Result<RequestLease, DispatchError> {
        if admission.class() != requirement.capacity_class() {
            return Err(DispatchError::Internal);
        }
        let mut matching = [false; MAX_WORKERS];
        let mut voice_policies = 0;
        for (index, record) in self.records.iter().enumerate() {
            if &record.trust_domain != requirement.trust_domain() {
                continue;
            }
            let (matched, policies) = record.profile_match(requirement);
            matching[index] = matched;
            voice_policies |= policies;
        }
        let profile_found = matching[..self.records.len()].contains(&true);
        if requirement.profile.has_named_voice() && voice_policies.count_ones() > 1 {
            return Err(DispatchError::AmbiguousModel);
        }
        if profile_found && requirement.profile.requires_default_resolution() {
            let mut resolved = None;
            for (index, record) in self.records.iter().enumerate() {
                if !matching[index] {
                    continue;
                }
                let Some(model_id) = record.default_model_id.as_deref() else {
                    return Err(DispatchError::AmbiguousModel);
                };
                match resolved {
                    None => resolved = Some(model_id),
                    Some(expected) if expected == model_id => {}
                    Some(_) => return Err(DispatchError::AmbiguousModel),
                }
            }
        }
        self.dispatch_matching(admission, profile_found, |record| {
            matching[record.registration_id.startup_ordinal()]
        })
    }

    fn dispatch_matching(
        &self,
        admission: AdmissionLease,
        profile_found: bool,
        matches: impl Fn(&WorkerRecord) -> bool,
    ) -> Result<RequestLease, DispatchError> {
        if !profile_found {
            return Err(DispatchError::NoEligibleProfile);
        }
        let mut eligible = [0; MAX_WORKERS];
        let mut eligible_count = 0;
        for record in &self.records {
            if matches(record) && record.is_routable() {
                eligible[eligible_count] = record.registration_id.startup_ordinal();
                eligible_count += 1;
            }
        }
        if eligible_count == 0 {
            return Err(DispatchError::Unavailable);
        }
        let eligible = &eligible[..eligible_count];
        let mut selector = self.selector.lock();
        let selected = match self.selector.strategy() {
            RoutingStrategy::RoundRobin => {
                let candidates = candidate_set(eligible);
                selector
                    .select(&candidates)
                    .map(|index| Arc::clone(&self.records[index]))
            }
            RoutingStrategy::LeastRequests => self.select_least_requests(eligible, &mut selector),
        }
        .ok_or(DispatchError::Internal)?;
        let lease = RequestLease::new(admission, selected);
        drop(selector);
        Ok(lease)
    }

    fn select_least_requests(
        &self,
        eligible: &[usize],
        selector: &mut SelectorGuard<'_>,
    ) -> Option<Arc<WorkerRecord>> {
        let mut minimum = usize::MAX;
        let mut ties = [0; MAX_WORKERS];
        let mut tie_count = 0;
        for &index in eligible {
            let record = &self.records[index];
            let load = record.load();
            if load < minimum {
                minimum = load;
                ties[0] = index;
                tie_count = 1;
            } else if load == minimum {
                ties[tie_count] = index;
                tie_count += 1;
            }
        }
        let candidates = candidate_set(&ties[..tie_count]);
        selector
            .select(&candidates)
            .map(|index| Arc::clone(&self.records[index]))
    }

    pub(crate) fn content_blind_generation_http(
        &self,
        trust: &TrustDomain,
    ) -> Option<ContentBlindGenerationHttp<'_>> {
        self.homogeneous_generation_http
            .iter()
            .find(|cohort| &cohort.trust_domain == trust)
            .map(|cohort| ContentBlindGenerationHttp {
                pool: self,
                trust: &cohort.trust_domain,
            })
    }

    pub(crate) fn content_blind_media_http(
        &self,
        trust: &TrustDomain,
        route: crate::config::HttpMediaRoute,
    ) -> Option<ContentBlindMediaHttp<'_>> {
        self.homogeneous_media_http
            .iter()
            .find(|cohort| {
                &cohort.trust_domain == trust
                    && cohort.service == route.service_class()
                    && cohort.task == route.speech_to_text_task()
            })
            .map(|cohort| ContentBlindMediaHttp { pool: self, cohort })
    }

    pub(crate) fn generation_http_ready(&self, trust: &TrustDomain) -> bool {
        self.records.iter().any(|record| {
            &record.trust_domain == trust
                && record.is_routable()
                && record
                    .profiles
                    .iter()
                    .any(|profile| profile.service_class() == ServiceClass::GenerationHttp)
        })
    }

    pub(crate) fn media_http_ready(
        &self,
        trust: &TrustDomain,
        routes: &[crate::config::HttpMediaRoute],
    ) -> bool {
        routes.iter().all(|route| {
            self.records.iter().any(|record| {
                &record.trust_domain == trust
                    && record.is_routable()
                    && record
                        .profiles
                        .iter()
                        .any(|profile| route.matches_profile(profile))
            })
        })
    }

    pub(crate) fn supports_speech_batch_size(&self, trust: &TrustDomain, size: u32) -> bool {
        self.records.iter().any(|record| {
            &record.trust_domain == trust
                && record.profiles.iter().any(|profile| {
                    matches!(
                        profile,
                        ServiceProfile::SpeechBatch { max_batch_size, .. }
                            if u32::from(*max_batch_size) >= size
                    )
                })
        })
    }

    pub(crate) fn drain(&self) {
        self.admission.close();
    }
}

fn candidate_set(indices: &[usize]) -> [bool; MAX_WORKERS] {
    let mut candidates = [false; MAX_WORKERS];
    for &index in indices {
        candidates[index] = true;
    }
    candidates
}

impl ContentBlindGenerationHttp<'_> {
    pub(crate) fn dispatch(self, admission: AdmissionLease) -> Result<RequestLease, DispatchError> {
        self.pool.dispatch_matching(admission, true, |record| {
            &record.trust_domain == self.trust
                && record
                    .profiles
                    .iter()
                    .any(|profile| profile.service_class() == ServiceClass::GenerationHttp)
        })
    }
}

impl ContentBlindMediaHttp<'_> {
    pub(crate) fn dispatch(self, admission: AdmissionLease) -> Result<RequestLease, DispatchError> {
        self.pool.dispatch_matching(admission, true, |record| {
            record.trust_domain == self.cohort.trust_domain
                && record.profiles.iter().any(|profile| {
                    profile.service_class() == self.cohort.service
                        && match (profile, self.cohort.task) {
                            (ServiceProfile::TranscriptionHttp { task, .. }, Some(required)) => {
                                *task == required
                            }
                            (ServiceProfile::TranscriptionHttp { .. }, None) => false,
                            (_, None) => true,
                            (_, Some(_)) => false,
                        }
                })
        })
    }
}

fn build_content_blind_generation_cohorts(
    records: &[Arc<WorkerRecord>],
) -> Vec<HomogeneousGenerationCohort> {
    let mut result = Vec::new();
    for record in records {
        if result
            .iter()
            .any(|cohort: &HomogeneousGenerationCohort| cohort.trust_domain == record.trust_domain)
        {
            continue;
        }
        let mut members = records.iter().filter(|candidate| {
            candidate.trust_domain == record.trust_domain
                && candidate
                    .profiles
                    .iter()
                    .any(|profile| profile.service_class() == ServiceClass::GenerationHttp)
        });
        let Some(first) = members.next() else {
            continue;
        };
        if first.default_model_id.is_some()
            && members.all(|candidate| {
                candidate.default_model_id == first.default_model_id
                    && generation_rows_equal(&candidate.profiles, &first.profiles)
            })
        {
            result.push(HomogeneousGenerationCohort {
                trust_domain: record.trust_domain.clone(),
            });
        }
    }
    result
}

fn build_content_blind_media_cohorts(records: &[Arc<WorkerRecord>]) -> Vec<HomogeneousMediaCohort> {
    let mut result = Vec::new();
    for record in records {
        for profile in &record.profiles {
            let service = profile.service_class();
            if service == ServiceClass::GenerationHttp || service == ServiceClass::SpeechBatch {
                continue;
            }
            let task = match profile {
                ServiceProfile::TranscriptionHttp { task, .. } => Some(*task),
                _ => None,
            };
            if result.iter().any(|cohort: &HomogeneousMediaCohort| {
                cohort.trust_domain == record.trust_domain
                    && cohort.service == service
                    && cohort.task == task
            }) {
                continue;
            }
            let members: Vec<_> = records
                .iter()
                .filter(|candidate| {
                    candidate.trust_domain == record.trust_domain
                        && candidate.profiles.iter().any(|row| {
                            row.service_class() == service
                                && match (row, task) {
                                    (
                                        ServiceProfile::TranscriptionHttp {
                                            task: row_task, ..
                                        },
                                        Some(required),
                                    ) => *row_task == required,
                                    (ServiceProfile::TranscriptionHttp { .. }, None) => false,
                                    (_, None) => true,
                                    (_, Some(_)) => false,
                                }
                        })
                })
                .collect();
            let Some(first) = members.first() else {
                continue;
            };
            if first.default_model_id.is_some()
                && members.iter().all(|candidate| {
                    candidate.default_model_id == first.default_model_id
                        && service_rows_equal(&candidate.profiles, &first.profiles, service, task)
                })
            {
                result.push(HomogeneousMediaCohort {
                    trust_domain: record.trust_domain.clone(),
                    service,
                    task,
                });
            }
        }
    }
    result
}

fn service_rows_equal(
    left: &[ServiceProfile],
    right: &[ServiceProfile],
    service: ServiceClass,
    task: Option<SpeechToTextTask>,
) -> bool {
    let relevant = |profile: &&ServiceProfile| {
        profile.service_class() == service
            && match (*profile, task) {
                (ServiceProfile::TranscriptionHttp { task: row_task, .. }, Some(required)) => {
                    *row_task == required
                }
                (ServiceProfile::TranscriptionHttp { .. }, None) => false,
                (_, None) => true,
                (_, Some(_)) => false,
            }
    };
    left.iter().filter(relevant).count() == right.iter().filter(relevant).count()
        && left.iter().filter(relevant).all(|profile| {
            right
                .iter()
                .filter(relevant)
                .any(|other| profile.semantically_eq(other))
        })
}

fn generation_rows_equal(left: &[ServiceProfile], right: &[ServiceProfile]) -> bool {
    service_rows_equal(left, right, ServiceClass::GenerationHttp, None)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::thread;

    use super::profile::{
        InputModality, MessageContentForm, ModelSelection, OutputModality, ProfileRequirement,
        ServiceProfile, StreamMode, VoiceNamePolicy,
    };
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

    fn speech_http_profile(
        model: &str,
        response_format: SpeechResponseFormat,
        stream_mode: StreamMode,
    ) -> ServiceProfile {
        ServiceProfile::SpeechHttp {
            model_ids: vec![model.to_owned()],
            response_formats: vec![response_format],
            stream_modes: vec![stream_mode],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::None],
            voice_name_policy: VoiceNamePolicy::Preset,
        }
    }

    fn requirement(model: &str, trust: &str) -> RouteRequirement {
        RouteRequirement::new(
            ProfileRequirement::GenerationHttp {
                model: ModelSelection::Explicit(model.to_owned()),
                message_content_forms: vec![MessageContentForm::String],
                media_placements: Vec::new(),
                input_modalities: vec![InputModality::Text],
                output_modalities: vec![OutputModality::Text],
                audio_format: None,
                stream_mode: StreamMode::NonStreaming,
            },
            TrustDomain::new(trust.to_owned()),
        )
    }

    fn record_with_profile(
        ordinal: usize,
        trust: &str,
        model: &str,
        service_profile: ServiceProfile,
    ) -> Arc<WorkerRecord> {
        let health = AtomicHealth::unknown();
        health.store(WorkerHealth::Healthy);
        Arc::new(WorkerRecord {
            worker_id: WorkerId::new(format!("worker-{ordinal}")),
            default_model_id: Some(model.to_owned()),
            registration_id: RegistrationId::from_startup_ordinal(ordinal),
            target: ResolvedTarget::from_parts(
                &format!("http://127.0.0.1:{}/", 10_000 + ordinal),
                "/health",
            )
            .expect("test target"),
            trust_domain: TrustDomain::new(trust.to_owned()),
            profiles: vec![service_profile],
            active_requests: AtomicUsize::new(0),
            health,
            immediate_probe: Notify::new(),
        })
    }

    fn record(ordinal: usize, trust: &str, model: &str) -> Arc<WorkerRecord> {
        record_with_profile(ordinal, trust, model, profile(model))
    }

    fn pool(
        strategy: RoutingStrategy,
        records: Vec<Arc<WorkerRecord>>,
        admission: usize,
    ) -> WorkerPool {
        let client = build_health_client(
            std::time::Duration::from_secs(1),
            std::time::Duration::from_secs(1),
        )
        .expect("test client");
        let selector = Selector::new(strategy, records.len());
        WorkerPool {
            homogeneous_generation_http: build_content_blind_generation_cohorts(&records),
            homogeneous_media_http: build_content_blind_media_cohorts(&records),
            records,
            admission: AdmissionController::new(admission, [Some(admission), None, None, None]),
            selector,
            health_client: client.clone(),
            http_client: client,
        }
    }

    fn dispatch_subset(pool: &WorkerPool, ordinals: &[usize]) -> usize {
        let lease = pool
            .dispatch_matching(
                pool.try_admit(CapacityClass::GenerationHttp, 1)
                    .expect("admit subset"),
                true,
                |record| ordinals.contains(&record.registration_id.startup_ordinal()),
            )
            .expect("dispatch subset");
        let selected = lease.registration_ordinal();
        drop(lease);
        selected
    }

    #[test]
    fn direct_proof_requires_equal_defaults_profiles_and_trust_scopes() {
        let local = TrustDomain::new(String::from("local"));
        let sole = pool(
            RoutingStrategy::RoundRobin,
            vec![record(0, "local", "omni")],
            4,
        );
        assert!(sole.content_blind_generation_http(&local).is_some());

        let replicas = pool(
            RoutingStrategy::RoundRobin,
            vec![record(0, "local", "omni"), record(1, "local", "omni")],
            4,
        );
        assert!(replicas.content_blind_generation_http(&local).is_some());

        let mut missing_default = record(0, "local", "omni");
        Arc::get_mut(&mut missing_default)
            .expect("new test record is uniquely owned")
            .default_model_id = None;
        let no_default = pool(RoutingStrategy::RoundRobin, vec![missing_default], 4);
        assert!(no_default.content_blind_generation_http(&local).is_none());

        let defaults_differ = pool(
            RoutingStrategy::RoundRobin,
            vec![record(0, "local", "omni"), record(1, "local", "other")],
            4,
        );
        assert!(
            defaults_differ
                .content_blind_generation_http(&local)
                .is_none()
        );

        let mutations: [fn(&mut ServiceProfile); 6] = [
            |profile| {
                if let ServiceProfile::GenerationHttp { model_ids, .. } = profile {
                    model_ids.push(String::from("other"));
                }
            },
            |profile| {
                if let ServiceProfile::GenerationHttp {
                    message_content_forms,
                    ..
                } = profile
                {
                    message_content_forms.push(MessageContentForm::TypedParts);
                }
            },
            |profile| {
                if let ServiceProfile::GenerationHttp {
                    media_placements, ..
                } = profile
                {
                    media_placements.push(MediaPlacement::TypedParts);
                }
            },
            |profile| {
                if let ServiceProfile::GenerationHttp {
                    input_modalities, ..
                } = profile
                {
                    input_modalities.push(InputModality::Image);
                }
            },
            |profile| {
                if let ServiceProfile::GenerationHttp {
                    output_modalities,
                    chat_audio_formats,
                    ..
                } = profile
                {
                    output_modalities.push(OutputModality::Audio);
                    chat_audio_formats.push(ChatAudioFormat::Wav);
                }
            },
            |profile| {
                if let ServiceProfile::GenerationHttp { stream_modes, .. } = profile {
                    stream_modes.push(StreamMode::Streaming);
                }
            },
        ];
        for mutate in mutations {
            let mut different = profile("omni");
            mutate(&mut different);
            let heterogeneous = pool(
                RoutingStrategy::RoundRobin,
                vec![
                    record(0, "local", "omni"),
                    record_with_profile(1, "local", "omni", different),
                ],
                4,
            );
            assert!(
                heterogeneous
                    .content_blind_generation_http(&local)
                    .is_none()
            );
        }

        let mut extra_row = record(1, "local", "omni");
        Arc::get_mut(&mut extra_row)
            .expect("new test record is uniquely owned")
            .profiles
            .push(profile("other"));
        let row_count_differs = pool(
            RoutingStrategy::RoundRobin,
            vec![record(0, "local", "omni"), extra_row],
            4,
        );
        assert!(
            row_count_differs
                .content_blind_generation_http(&local)
                .is_none()
        );

        let separate = pool(
            RoutingStrategy::RoundRobin,
            vec![record(0, "local", "omni"), record(1, "remote", "other")],
            4,
        );
        assert!(separate.content_blind_generation_http(&local).is_some());
    }

    #[test]
    fn round_robin_balances_and_skips_unhealthy_workers() {
        let records = vec![record(0, "local", "omni"), record(1, "local", "omni")];
        let pool = pool(RoutingStrategy::RoundRobin, records.clone(), 8);
        let first = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp, 1)
                    .expect("admit first"),
                &requirement("omni", "local"),
            )
            .expect("first dispatch");
        let second = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp, 1)
                    .expect("admit second"),
                &requirement("omni", "local"),
            )
            .expect("second dispatch");
        assert_ne!(first.registration_ordinal(), second.registration_ordinal());
        drop(first);
        drop(second);
        records[0].health.store(WorkerHealth::Unhealthy);
        records[1].health.store(WorkerHealth::Unhealthy);
        let unavailable = pool.dispatch(
            pool.try_admit(CapacityClass::GenerationHttp, 1)
                .expect("admit unavailable"),
            &requirement("omni", "local"),
        );
        assert!(matches!(unavailable, Err(DispatchError::Unavailable)));
    }

    #[test]
    fn round_robin_rotates_over_sparse_eligible_workers_without_bias() {
        let records = vec![
            record(0, "local", "omni"),
            record(1, "remote", "other"),
            record(2, "local", "omni"),
        ];
        let pool = pool(RoutingStrategy::RoundRobin, records, 8);
        let mut selected = Vec::new();
        for _ in 0..6 {
            let lease = pool
                .dispatch(
                    pool.try_admit(CapacityClass::GenerationHttp, 1)
                        .expect("admit sparse round robin"),
                    &requirement("omni", "local"),
                )
                .expect("dispatch sparse round robin");
            selected.push(lease.registration_ordinal());
            drop(lease);
        }
        assert_eq!(selected, [0, 2, 0, 2, 0, 2]);
    }

    #[test]
    fn round_robin_rotates_alternating_disjoint_sets() {
        let pool = pool(
            RoutingStrategy::RoundRobin,
            vec![
                record(0, "local", "omni"),
                record(1, "local", "omni"),
                record(2, "local", "omni"),
                record(3, "local", "omni"),
            ],
            8,
        );
        let mut selected = Vec::new();
        for _ in 0..4 {
            selected.push(dispatch_subset(&pool, &[0, 1]));
            selected.push(dispatch_subset(&pool, &[2, 3]));
        }

        assert_eq!(selected, [0, 2, 1, 3, 0, 2, 1, 3]);
    }

    #[test]
    fn round_robin_rotates_overlapping_sets_without_starvation() {
        let pool = pool(
            RoutingStrategy::RoundRobin,
            vec![
                record(0, "local", "omni"),
                record(1, "local", "omni"),
                record(2, "local", "omni"),
            ],
            8,
        );
        let mut selected = Vec::new();
        for _ in 0..3 {
            selected.push(dispatch_subset(&pool, &[0, 1]));
            selected.push(dispatch_subset(&pool, &[1, 2]));
        }

        assert_eq!(selected, [0, 1, 0, 2, 1, 2]);
    }

    #[test]
    fn least_requests_rotates_equal_load_over_sparse_eligible_workers_without_bias() {
        let records = vec![
            record(0, "remote", "other"),
            record(1, "local", "omni"),
            record(2, "remote", "other"),
            record(3, "local", "omni"),
        ];
        let pool = pool(RoutingStrategy::LeastRequests, records, 8);
        let mut selected = Vec::new();
        for _ in 0..6 {
            let lease = pool
                .dispatch(
                    pool.try_admit(CapacityClass::GenerationHttp, 1)
                        .expect("admit sparse least requests"),
                    &requirement("omni", "local"),
                )
                .expect("dispatch sparse least requests");
            selected.push(lease.registration_ordinal());
            drop(lease);
        }
        assert_eq!(selected, [1, 3, 1, 3, 1, 3]);
    }

    #[test]
    fn least_requests_rotates_only_over_the_minimum_occupancy_tie() {
        let records = vec![
            record(0, "local", "omni"),
            record(1, "local", "omni"),
            record(2, "local", "omni"),
        ];
        let middle = Arc::clone(&records[1]);
        middle.increment_load(1);
        let pool = pool(RoutingStrategy::LeastRequests, records, 8);
        let trust = TrustDomain::new(String::from("local"));
        let mut selected = Vec::new();
        for _ in 0..6 {
            let lease = pool
                .content_blind_generation_http(&trust)
                .expect("homogeneous cohort")
                .dispatch(
                    pool.try_admit(CapacityClass::GenerationHttp, 1)
                        .expect("admit tied least requests"),
                )
                .expect("dispatch tied least requests");
            selected.push(lease.registration_ordinal());
            drop(lease);
        }
        assert_eq!(selected, [0, 2, 0, 2, 0, 2]);
        middle.decrement_load(1);
    }

    #[test]
    fn least_requests_choose_and_reserve_is_linearized() {
        const REQUESTS: usize = 32;
        let records = vec![record(0, "local", "omni"), record(1, "local", "omni")];
        let pool = Arc::new(pool(RoutingStrategy::LeastRequests, records, REQUESTS));
        let start = Arc::new(Barrier::new(REQUESTS + 1));
        let mut threads = Vec::new();
        for _ in 0..REQUESTS {
            let pool = Arc::clone(&pool);
            let start = Arc::clone(&start);
            threads.push(thread::spawn(move || {
                let admission = pool
                    .try_admit(CapacityClass::GenerationHttp, 1)
                    .expect("concurrent admission");
                start.wait();
                pool.dispatch(admission, &requirement("omni", "local"))
                    .expect("concurrent dispatch")
            }));
        }
        start.wait();
        let leases: Vec<_> = threads
            .into_iter()
            .map(|thread| thread.join().expect("join dispatcher"))
            .collect();
        let first = leases
            .iter()
            .filter(|lease| lease.registration_ordinal() == 0)
            .count();
        assert_eq!(first, REQUESTS / 2);
    }

    #[test]
    fn unresolved_default_routes_to_the_only_compatible_worker() {
        let mut multimodal = profile("vision");
        if let ServiceProfile::GenerationHttp {
            message_content_forms,
            media_placements,
            input_modalities,
            ..
        } = &mut multimodal
        {
            message_content_forms.push(MessageContentForm::TypedParts);
            media_placements.push(MediaPlacement::TypedParts);
            input_modalities.push(InputModality::Image);
        }
        let pool = pool(
            RoutingStrategy::RoundRobin,
            vec![
                record(0, "local", "text"),
                record_with_profile(1, "local", "vision", multimodal),
            ],
            2,
        );
        let requirement = RouteRequirement::new(
            ProfileRequirement::GenerationHttp {
                model: ModelSelection::UnresolvedDefault,
                message_content_forms: vec![MessageContentForm::TypedParts],
                media_placements: vec![MediaPlacement::TypedParts],
                input_modalities: vec![InputModality::Text, InputModality::Image],
                output_modalities: vec![OutputModality::Text],
                audio_format: None,
                stream_mode: StreamMode::NonStreaming,
            },
            TrustDomain::new(String::from("local")),
        );
        let lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp, 1)
                    .expect("admit heterogeneous request"),
                &requirement,
            )
            .expect("dispatch heterogeneous default");
        assert_eq!(lease.registration_ordinal(), 1);
    }

    #[test]
    fn unresolved_default_agreement_is_stable_across_health_changes() {
        let records = vec![record(0, "local", "first"), record(1, "local", "second")];
        records[1].health.store(WorkerHealth::Unhealthy);
        let pool = pool(RoutingStrategy::RoundRobin, records, 2);
        let requirement = RouteRequirement::new(
            ProfileRequirement::GenerationHttp {
                model: ModelSelection::UnresolvedDefault,
                message_content_forms: vec![MessageContentForm::String],
                media_placements: Vec::new(),
                input_modalities: vec![InputModality::Text],
                output_modalities: vec![OutputModality::Text],
                audio_format: None,
                stream_mode: StreamMode::NonStreaming,
            },
            TrustDomain::new(String::from("local")),
        );
        let result = pool.dispatch(
            pool.try_admit(CapacityClass::GenerationHttp, 1)
                .expect("admit unresolved request"),
            &requirement,
        );
        assert!(matches!(result, Err(DispatchError::AmbiguousModel)));
    }

    #[test]
    fn media_default_resolution_uses_format_and_stream_constraints() {
        let records = vec![
            record_with_profile(
                0,
                "local",
                "wav-model",
                speech_http_profile(
                    "wav-model",
                    SpeechResponseFormat::Wav,
                    StreamMode::NonStreaming,
                ),
            ),
            record_with_profile(
                1,
                "local",
                "pcm-model",
                speech_http_profile(
                    "pcm-model",
                    SpeechResponseFormat::Pcm,
                    StreamMode::Streaming,
                ),
            ),
        ];
        let mut pool = pool(RoutingStrategy::RoundRobin, records, 2);
        pool.admission = AdmissionController::new(2, [None, Some(2), None, None]);
        let requirement = RouteRequirement::new(
            ProfileRequirement::SpeechHttp {
                model: ModelSelection::UnresolvedDefault,
                response_format: SpeechResponseFormat::Pcm,
                stream_mode: StreamMode::Streaming,
                task: Some(SpeechTask::TextToSpeech),
                reference_forms: vec![ReferenceForm::None],
                named_voice: false,
            },
            TrustDomain::new(String::from("local")),
        );
        let lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::SpeechHttp, 1)
                    .expect("admit speech request"),
                &requirement,
            )
            .expect("dispatch compatible default");
        assert_eq!(lease.registration_ordinal(), 1);
    }

    #[test]
    fn admission_and_worker_load_release_on_every_drop() {
        let pool = pool(
            RoutingStrategy::RoundRobin,
            vec![record(0, "local", "omni")],
            1,
        );
        let lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp, 1)
                    .expect("admit"),
                &requirement("omni", "local"),
            )
            .expect("dispatch");
        assert_eq!(pool.admission.available(), (0, [Some(0), None, None, None]));
        assert_eq!(pool.records[0].load(), 1);
        drop(lease);
        assert_eq!(pool.admission.available(), (1, [Some(1), None, None, None]));
        assert_eq!(pool.records[0].load(), 0);
    }

    #[test]
    fn readiness_tracks_worker_health() {
        let record = record(0, "local", "omni");
        record.health.store(WorkerHealth::Unknown);
        let pool = pool(RoutingStrategy::RoundRobin, vec![Arc::clone(&record)], 1);
        let trust = TrustDomain::new(String::from("local"));
        assert!(!pool.generation_http_ready(&trust));
        record.health.store(WorkerHealth::Healthy);
        assert!(pool.generation_http_ready(&trust));
    }

    #[test]
    fn drain_rejects_new_admission_and_preserves_admitted_work() {
        let pool = pool(
            RoutingStrategy::RoundRobin,
            vec![record(0, "local", "omni")],
            1,
        );
        let trust = TrustDomain::new(String::from("local"));
        let admission = pool
            .try_admit(CapacityClass::GenerationHttp, 1)
            .expect("admit before drain");

        pool.drain();

        assert!(matches!(
            pool.try_admit(CapacityClass::GenerationHttp, 1),
            Err(AdmissionError::Draining)
        ));
        let lease = pool
            .content_blind_generation_http(&trust)
            .expect("homogeneous cohort")
            .dispatch(admission)
            .expect("admitted request may dispatch during drain");
        drop(lease);
    }

    fn media_record(ordinal: usize, profile: ServiceProfile) -> Arc<WorkerRecord> {
        let health = AtomicHealth::unknown();
        health.store(WorkerHealth::Healthy);
        Arc::new(WorkerRecord {
            worker_id: WorkerId::new(format!("media-{ordinal}")),
            default_model_id: Some(String::from("tts")),
            registration_id: RegistrationId::from_startup_ordinal(ordinal),
            target: ResolvedTarget::from_parts(
                &format!("http://127.0.0.1:{}/", 12_000 + ordinal),
                "/health",
            )
            .expect("media target"),
            trust_domain: TrustDomain::new(String::from("local")),
            profiles: vec![profile],
            active_requests: AtomicUsize::new(0),
            health,
            immediate_probe: Notify::new(),
        })
    }

    fn media_pool(records: Vec<Arc<WorkerRecord>>) -> WorkerPool {
        let client = build_health_client(
            std::time::Duration::from_secs(1),
            std::time::Duration::from_secs(1),
        )
        .expect("media client");
        let selector = Selector::new(RoutingStrategy::RoundRobin, records.len());
        WorkerPool {
            homogeneous_generation_http: build_content_blind_generation_cohorts(&records),
            homogeneous_media_http: build_content_blind_media_cohorts(&records),
            records,
            admission: AdmissionController::new(8, [Some(8), Some(8), Some(8), Some(8)]),
            selector,
            health_client: client.clone(),
            http_client: client,
        }
    }

    fn batch_profile() -> ServiceProfile {
        ServiceProfile::SpeechBatch {
            model_ids: vec![String::from("tts")],
            response_formats: vec![SpeechResponseFormat::Wav],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::None],
            voice_name_policy: VoiceNamePolicy::Preset,
            max_batch_size: 8,
        }
    }

    fn speech_profile() -> ServiceProfile {
        ServiceProfile::SpeechHttp {
            model_ids: vec![String::from("tts")],
            response_formats: vec![SpeechResponseFormat::Wav],
            stream_modes: vec![StreamMode::NonStreaming],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::None],
            voice_name_policy: VoiceNamePolicy::Preset,
        }
    }

    fn named_speech_profile(model: &str, policy: VoiceNamePolicy) -> ServiceProfile {
        ServiceProfile::SpeechHttp {
            model_ids: vec![model.to_owned()],
            response_formats: vec![SpeechResponseFormat::Wav],
            stream_modes: vec![StreamMode::NonStreaming],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::None],
            voice_name_policy: policy,
        }
    }

    fn named_speech_requirement(model: ModelSelection) -> RouteRequirement {
        RouteRequirement::new(
            ProfileRequirement::SpeechHttp {
                model,
                response_format: SpeechResponseFormat::Wav,
                stream_mode: StreamMode::NonStreaming,
                task: Some(SpeechTask::TextToSpeech),
                reference_forms: Vec::new(),
                named_voice: true,
            },
            TrustDomain::new(String::from("local")),
        )
    }

    #[test]
    fn homogeneous_media_uses_existing_policy_and_skips_unhealthy_workers() {
        let first = media_record(0, speech_profile());
        let second = media_record(1, speech_profile());
        let pool = media_pool(vec![Arc::clone(&first), Arc::clone(&second)]);
        let trust = TrustDomain::new(String::from("local"));
        let route = crate::config::HttpMediaRoute::Speech;
        let first_lease = pool
            .content_blind_media_http(&trust, route)
            .expect("speech cohort")
            .dispatch(
                pool.try_admit(CapacityClass::SpeechHttp, 1)
                    .expect("first admission"),
            )
            .expect("first dispatch");
        assert_eq!(first_lease.registration_ordinal(), 0);
        let second_lease = pool
            .content_blind_media_http(&trust, route)
            .expect("speech cohort")
            .dispatch(
                pool.try_admit(CapacityClass::SpeechHttp, 1)
                    .expect("second admission"),
            )
            .expect("full-worker fallback");
        assert_eq!(second_lease.registration_ordinal(), 1);
        drop(first_lease);
        first.health.store(WorkerHealth::Unhealthy);
        drop(second_lease);
        let healthy = pool
            .content_blind_media_http(&trust, route)
            .expect("speech cohort")
            .dispatch(
                pool.try_admit(CapacityClass::SpeechHttp, 1)
                    .expect("healthy admission"),
            )
            .expect("unhealthy-worker fallback");
        assert_eq!(healthy.registration_ordinal(), 1);
    }

    #[test]
    fn unrelated_media_only_worker_does_not_change_chat_cohort_or_readiness() {
        let generation = record(0, "local", "omni");
        let media = media_record(1, speech_profile());
        let pool = media_pool(vec![generation, media]);
        let trust = TrustDomain::new(String::from("local"));
        assert!(pool.generation_http_ready(&trust));
        let lease = pool
            .content_blind_generation_http(&trust)
            .expect("generation cohort")
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp, 1)
                    .expect("generation admission"),
            )
            .expect("generation dispatch");
        assert_eq!(lease.registration_ordinal(), 0);
    }

    #[test]
    fn batch_counts_all_item_credits_and_releases_once() {
        let record = media_record(0, batch_profile());
        let pool = media_pool(vec![Arc::clone(&record)]);
        let requirement = RouteRequirement::new(
            ProfileRequirement::SpeechBatch {
                models: vec![ModelSelection::Explicit(String::from("tts"))],
                response_formats: vec![SpeechResponseFormat::Wav],
                tasks: vec![SpeechTask::TextToSpeech],
                reference_forms: vec![ReferenceForm::None],
                named_voice: false,
                batch_size: 3,
            },
            TrustDomain::new(String::from("local")),
        );
        let lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::SpeechBatch, 3)
                    .expect("batch admission"),
                &requirement,
            )
            .expect("batch dispatch");
        assert_eq!(record.load(), 3);
        drop(lease);
        assert_eq!(record.load(), 0);
        let oversized = RouteRequirement::new(
            ProfileRequirement::SpeechBatch {
                models: vec![ModelSelection::Explicit(String::from("tts"))],
                response_formats: vec![SpeechResponseFormat::Wav],
                tasks: vec![SpeechTask::TextToSpeech],
                reference_forms: vec![ReferenceForm::None],
                named_voice: false,
                batch_size: 5,
            },
            TrustDomain::new(String::from("local")),
        );
        let lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::SpeechBatch, 5)
                    .expect("five-item class admission"),
                &oversized,
            )
            .expect("five-item batch dispatch");
        assert_eq!(record.load(), 5);
        drop(lease);
        assert_eq!(record.load(), 0);
    }

    #[test]
    fn named_voice_policy_follows_the_selected_model() {
        let preset = record_with_profile(
            0,
            "local",
            "custom",
            named_speech_profile("custom", VoiceNamePolicy::Preset),
        );
        let uploaded = record_with_profile(
            1,
            "local",
            "base",
            named_speech_profile("base", VoiceNamePolicy::Uploaded),
        );
        let pool = media_pool(vec![preset, uploaded]);

        let preset_lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::SpeechHttp, 1)
                    .expect("preset admission"),
                &named_speech_requirement(ModelSelection::Explicit(String::from("custom"))),
            )
            .expect("preset dispatch");
        assert_eq!(preset_lease.registration_ordinal(), 0);
        drop(preset_lease);

        let uploaded_lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::SpeechHttp, 1)
                    .expect("uploaded admission"),
                &named_speech_requirement(ModelSelection::Explicit(String::from("base"))),
            )
            .expect("uploaded dispatch");
        assert_eq!(uploaded_lease.registration_ordinal(), 1);
        drop(uploaded_lease);

        assert!(matches!(
            pool.dispatch(
                pool.try_admit(CapacityClass::SpeechHttp, 1)
                    .expect("ambiguous admission"),
                &named_speech_requirement(ModelSelection::UnresolvedDefault),
            ),
            Err(DispatchError::AmbiguousModel)
        ));
    }
}
