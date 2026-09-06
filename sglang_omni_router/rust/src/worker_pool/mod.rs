mod admission;
mod health;
pub(crate) mod profile;
mod resolver;
mod selection;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::{Notify, Semaphore};

use crate::config::{Config, RoutingStrategy};
use crate::metrics::{Rejection, RouterMetrics};

pub(crate) use admission::{
    AdmissionError, AdmissionLease, DispatchError, EnvelopeLease, RequestLease,
};
pub(crate) use health::{HealthSupervisor, ProbeOutcome, ProbeSnapshot, WorkerHealth};
pub(crate) use profile::{
    CapacityClass, ChatAudioFormat, MediaPlacement, MessageContentForm, ModelSelection,
    ProfileRequirement, ReferenceForm, RouteRequirement, ServiceClass, SpeechResponseFormat,
    SpeechTask, SpeechToTextTask, StreamMode, TranscriptionResponseFormat, TrustDomain,
};
pub(crate) use resolver::{ConnectTarget, ResolvedTarget};

use admission::AdmissionController;
use health::{AtomicHealth, ProbeState};
use profile::{
    CAPACITY_CLASS_COUNT, MAX_WORKERS, RegistrationId, ServiceProfile, VoiceNamePolicy,
    WorkerCapacityConfig, WorkerId,
};
use resolver::{build_health_client, build_http_client};
use selection::{Selector, SelectorGuard};

struct SessionCapacity {
    limit: usize,
    semaphore: Arc<Semaphore>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SessionCapacitySnapshot {
    pub(crate) class: CapacityClass,
    pub(crate) limit: usize,
    pub(crate) in_flight: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionClass {
    Global,
    Service(CapacityClass),
}

impl AdmissionClass {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Service(class) => class.label(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AdmissionSnapshot {
    pub(crate) class: AdmissionClass,
    pub(crate) limit: usize,
    pub(crate) in_flight: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WorkerSnapshot {
    pub(crate) worker_id: String,
    pub(crate) registration_ordinal: usize,
    pub(crate) voice_owner: bool,
    pub(crate) health: WorkerHealth,
    pub(crate) probe: ProbeSnapshot,
    pub(crate) routable: bool,
    pub(crate) active_requests: usize,
    pub(crate) session_capacity: Vec<SessionCapacitySnapshot>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OperationsSnapshot {
    pub(crate) admission: [AdmissionSnapshot; CAPACITY_CLASS_COUNT + 1],
    pub(crate) workers: Vec<WorkerSnapshot>,
}

/// One static startup registration with independently updated health and load.
pub(super) struct WorkerRecord {
    worker_id: WorkerId,
    default_model_id: Option<String>,
    registration_id: RegistrationId,
    target: ResolvedTarget,
    trust_domain: TrustDomain,
    profiles: Vec<ServiceProfile>,
    active_requests: AtomicUsize,
    session_capacity: [Option<SessionCapacity>; 2],
    health: AtomicHealth,
    probe: ProbeState,
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

    fn session_capacity(&self, class: CapacityClass) -> Option<&SessionCapacity> {
        session_capacity_index(class).and_then(|index| self.session_capacity[index].as_ref())
    }
}

/// Static-membership worker pool with bounded admission, deterministic policy
/// state, and independently owned health and weighted load.
pub(crate) struct WorkerPool {
    records: Vec<Arc<WorkerRecord>>,
    voice_owner: Option<Arc<WorkerRecord>>,
    admission: AdmissionController,
    selector: Selector,
    homogeneous_generation_http: Vec<HomogeneousGenerationCohort>,
    homogeneous_media_http: Vec<HomogeneousMediaCohort>,
    health_client: reqwest::Client,
    http_client: reqwest::Client,
    metrics: Arc<RouterMetrics>,
}

struct HomogeneousGenerationCohort {
    trust_domain: TrustDomain,
}

struct HomogeneousMediaCohort {
    trust_domain: TrustDomain,
    service: ServiceClass,
    task: Option<SpeechToTextTask>,
}

struct ProfileMatches {
    workers: [bool; MAX_WORKERS],
    any: bool,
    voice_policy: Option<VoiceNamePolicy>,
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
    pub(crate) fn build(
        config: &Config,
        metrics: Arc<RouterMetrics>,
    ) -> Result<Self, crate::error::RouterError> {
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
                admission_limit(config.admission.speech_websocket)?,
                admission_limit(config.admission.realtime_websocket)?,
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
                session_capacity: build_session_capacity(&worker.capacity)?,
                health: AtomicHealth::unknown(),
                probe: ProbeState::pending(),
                immediate_probe: Notify::new(),
            }));
        }
        let voice_owner = config
            .router
            .voice_owner_worker_id
            .as_ref()
            .map(|owner_id| {
                records
                    .iter()
                    .find(|record| record.worker_id.as_str() == owner_id)
                    .cloned()
                    .ok_or(crate::error::RouterError::WorkerPoolInvariant)
            })
            .transpose()?;
        let homogeneous_generation_http = build_content_blind_generation_cohorts(&records);
        let homogeneous_media_http =
            build_content_blind_media_cohorts(&records, voice_owner.as_ref());
        let selector = Selector::new(config.router.strategy, records.len());
        Ok(Self {
            records,
            voice_owner,
            admission,
            selector,
            homogeneous_generation_http,
            homogeneous_media_http,
            health_client,
            http_client,
            metrics,
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
        let envelope = self.try_admit_envelope()?;
        self.try_admit_class(envelope, class, credits)
    }

    pub(crate) fn try_admit_envelope(&self) -> Result<EnvelopeLease, AdmissionError> {
        let admitted = self.admission.try_admit_envelope();
        if matches!(admitted, Err(AdmissionError::Overloaded)) {
            self.metrics.record_rejection(Rejection::GlobalAdmission);
        }
        admitted
    }

    pub(crate) fn try_admit_class(
        &self,
        envelope: EnvelopeLease,
        class: CapacityClass,
        credits: u32,
    ) -> Result<AdmissionLease, AdmissionError> {
        let admitted = self.admission.try_admit_class(envelope, class, credits);
        if matches!(admitted, Err(AdmissionError::Overloaded)) {
            self.metrics.record_rejection(Rejection::admission(class));
        }
        admitted
    }

    pub(crate) fn dispatch(
        &self,
        admission: AdmissionLease,
        requirement: &RouteRequirement,
    ) -> Result<RequestLease, DispatchError> {
        if admission.class() != requirement.capacity_class() {
            return Err(DispatchError::Internal);
        }
        let matching = self.matching_profiles(requirement)?;
        if matching.voice_policy == Some(VoiceNamePolicy::Uploaded) {
            let owner = self.uploaded_voice_owner(requirement, &matching)?;
            let policy = self.selector.lock();
            let lease = RequestLease::new(admission, Arc::clone(owner));
            drop(policy);
            return Ok(lease);
        }
        if matching.any && requirement.profile.requires_default_resolution() {
            let mut resolved = None;
            for (index, record) in self.records.iter().enumerate() {
                if !matching.workers[index] {
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
        self.dispatch_matching(admission, matching.any, |record| {
            matching.workers[record.registration_id.startup_ordinal()]
        })
    }

    pub(crate) fn dispatch_session(
        &self,
        admission: AdmissionLease,
        requirement: &RouteRequirement,
    ) -> Result<RequestLease, DispatchError> {
        let class = admission.class();
        if class != requirement.capacity_class() || session_capacity_index(class).is_none() {
            return Err(DispatchError::Internal);
        }

        let matching = self.matching_profiles(requirement)?;
        if matching.voice_policy == Some(VoiceNamePolicy::Uploaded) {
            let owner = self.uploaded_voice_owner(requirement, &matching)?;
            let selector = self.selector.lock();
            let capacity = owner
                .session_capacity(class)
                .ok_or(DispatchError::Internal)?;
            let permit = Arc::clone(&capacity.semaphore)
                .try_acquire_owned()
                .map_err(|_| self.session_overloaded(class))?;
            let lease = RequestLease::new_session(admission, permit, Arc::clone(owner));
            drop(selector);
            return Ok(lease);
        }

        let mut healthy_found = false;
        let mut eligible = [0; MAX_WORKERS];
        let mut eligible_count = 0;
        let requires_default_resolution = requirement.profile.requires_default_resolution();
        let mut resolved_default = None;
        let mut selector = self.selector.lock();
        for (index, record) in self.records.iter().enumerate() {
            if !matching.workers[index] {
                continue;
            }
            if requires_default_resolution {
                let Some(model_id) = record.default_model_id.as_deref() else {
                    return Err(DispatchError::AmbiguousModel);
                };
                match resolved_default {
                    None => resolved_default = Some(model_id),
                    Some(expected) if expected == model_id => {}
                    Some(_) => return Err(DispatchError::AmbiguousModel),
                }
            }
            if !record.is_routable() {
                continue;
            }
            healthy_found = true;
            if record
                .session_capacity(class)
                .is_some_and(|capacity| capacity.semaphore.available_permits() != 0)
            {
                let index = record.registration_id.startup_ordinal();
                eligible[eligible_count] = index;
                eligible_count += 1;
            }
        }
        if !matching.any {
            return Err(DispatchError::NoEligibleProfile);
        }
        if !healthy_found {
            return Err(DispatchError::Unavailable);
        }
        if eligible_count == 0 {
            return Err(self.session_overloaded(class));
        }

        let eligible = &eligible[..eligible_count];
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
        let capacity = selected
            .session_capacity(class)
            .ok_or(DispatchError::Internal)?;
        let permit = Arc::clone(&capacity.semaphore)
            .try_acquire_owned()
            .map_err(|_| DispatchError::Internal)?;
        let lease = RequestLease::new_session(admission, permit, selected);
        drop(selector);
        Ok(lease)
    }

    fn session_overloaded(&self, class: CapacityClass) -> DispatchError {
        if let Some(rejection) = Rejection::worker(class) {
            self.metrics.record_rejection(rejection);
        }
        DispatchError::Overloaded
    }

    fn matching_profiles(
        &self,
        requirement: &RouteRequirement,
    ) -> Result<ProfileMatches, DispatchError> {
        let mut workers = [false; MAX_WORKERS];
        let mut voice_policies = 0;
        for (index, record) in self.records.iter().enumerate() {
            if &record.trust_domain != requirement.trust_domain() {
                continue;
            }
            let (matched, policies) = record.profile_match(requirement);
            workers[index] = matched;
            voice_policies |= policies;
        }
        let voice_policy = match voice_policies {
            0 => None,
            policy if policy == VoiceNamePolicy::Preset.bit() => Some(VoiceNamePolicy::Preset),
            policy if policy == VoiceNamePolicy::Uploaded.bit() => Some(VoiceNamePolicy::Uploaded),
            _ => return Err(DispatchError::AmbiguousModel),
        };
        Ok(ProfileMatches {
            any: workers[..self.records.len()].contains(&true),
            workers,
            voice_policy,
        })
    }

    fn uploaded_voice_owner(
        &self,
        requirement: &RouteRequirement,
        matching: &ProfileMatches,
    ) -> Result<&Arc<WorkerRecord>, DispatchError> {
        let owner = self
            .voice_owner
            .as_ref()
            .ok_or(DispatchError::NoEligibleProfile)?;
        if !matching.workers[owner.registration_id.startup_ordinal()] {
            return Err(DispatchError::NoEligibleProfile);
        }
        if requirement.profile.requires_default_resolution() && owner.default_model_id.is_none() {
            return Err(DispatchError::AmbiguousModel);
        }
        if !owner.is_routable() {
            return Err(DispatchError::Unavailable);
        }
        Ok(owner)
    }

    pub(crate) fn voice_state_enabled(&self) -> bool {
        self.voice_owner.is_some()
    }

    pub(crate) fn voice_owner_ready(&self) -> bool {
        self.voice_owner
            .as_ref()
            .is_none_or(|owner| owner.is_routable())
    }

    pub(crate) fn dispatch_voice_owner(
        &self,
        envelope: EnvelopeLease,
    ) -> Result<RequestLease, DispatchError> {
        let owner = self.voice_owner.as_ref().ok_or(DispatchError::Internal)?;
        if !owner.is_routable() {
            return Err(DispatchError::Unavailable);
        }
        let policy = self.selector.lock();
        let lease = RequestLease::new_owner(envelope, Arc::clone(owner));
        drop(policy);
        Ok(lease)
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

    pub(crate) fn operations_snapshot(&self) -> OperationsSnapshot {
        let raw_admission = self.admission.snapshot();
        let admission = std::array::from_fn(|index| AdmissionSnapshot {
            class: if index == 0 {
                AdmissionClass::Global
            } else {
                AdmissionClass::Service(CapacityClass::ALL[index - 1])
            },
            limit: raw_admission[index].0,
            in_flight: raw_admission[index].1,
        });
        let workers = self
            .records
            .iter()
            .map(|record| {
                let health = record.health.load();
                WorkerSnapshot {
                    worker_id: record.worker_id.as_str().to_owned(),
                    registration_ordinal: record.registration_id.startup_ordinal(),
                    voice_owner: self
                        .voice_owner
                        .as_ref()
                        .is_some_and(|owner| Arc::ptr_eq(owner, record)),
                    health,
                    probe: record.probe.snapshot(),
                    routable: health == WorkerHealth::Healthy,
                    active_requests: record.load(),
                    session_capacity: SESSION_CAPACITY_CLASSES
                        .into_iter()
                        .filter_map(|class| {
                            record
                                .session_capacity(class)
                                .map(|capacity| SessionCapacitySnapshot {
                                    class,
                                    limit: capacity.limit,
                                    in_flight: capacity.limit
                                        - capacity.semaphore.available_permits(),
                                })
                        })
                        .collect(),
                }
            })
            .collect();
        OperationsSnapshot { admission, workers }
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

    pub(crate) fn service_ready(&self, trust: &TrustDomain, service: ServiceClass) -> bool {
        self.records.iter().any(|record| {
            &record.trust_domain == trust
                && record.is_routable()
                && record
                    .profiles
                    .iter()
                    .any(|profile| profile.service_class() == service)
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

fn build_content_blind_media_cohorts(
    records: &[Arc<WorkerRecord>],
    voice_owner: Option<&Arc<WorkerRecord>>,
) -> Vec<HomogeneousMediaCohort> {
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
            let requires_voice_owner = service == ServiceClass::SpeechHttp
                && first.profiles.iter().any(|row| {
                    row.service_class() == service
                        && row.voice_name_policy() == Some(VoiceNamePolicy::Uploaded)
                });
            if requires_voice_owner
                && !voice_owner
                    .is_some_and(|owner| members.iter().all(|member| Arc::ptr_eq(member, owner)))
            {
                continue;
            }
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

pub(crate) const SESSION_CAPACITY_CLASSES: [CapacityClass; 2] = [
    CapacityClass::SpeechWebsocket,
    CapacityClass::RealtimeWebsocket,
];

const fn session_capacity_index(class: CapacityClass) -> Option<usize> {
    match class {
        CapacityClass::SpeechWebsocket => Some(0),
        CapacityClass::RealtimeWebsocket => Some(1),
        CapacityClass::GenerationHttp
        | CapacityClass::SpeechHttp
        | CapacityClass::SpeechBatch
        | CapacityClass::TranscriptionHttp => None,
    }
}

fn build_session_capacity(
    config: &WorkerCapacityConfig,
) -> Result<[Option<SessionCapacity>; 2], crate::error::RouterError> {
    let build = |configured: Option<u32>| {
        configured
            .map(usize::try_from)
            .transpose()
            .map_err(|_| crate::error::RouterError::WorkerPoolInvariant)
            .map(|limit| {
                limit.map(|limit| SessionCapacity {
                    limit,
                    semaphore: Arc::new(Semaphore::new(limit)),
                })
            })
    };
    Ok([
        build(config.speech_websocket)?,
        build(config.realtime_websocket)?,
    ])
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::thread;

    use crate::metrics::Rejection;

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

    fn speech_websocket_profile(
        model: &str,
        response_format: SpeechResponseFormat,
        stream_mode: StreamMode,
    ) -> ServiceProfile {
        speech_websocket_profile_with_policy(
            model,
            response_format,
            stream_mode,
            VoiceNamePolicy::Preset,
        )
    }

    fn speech_websocket_profile_with_policy(
        model: &str,
        response_format: SpeechResponseFormat,
        stream_mode: StreamMode,
        voice_name_policy: VoiceNamePolicy,
    ) -> ServiceProfile {
        ServiceProfile::SpeechWebsocket {
            model_ids: vec![model.to_owned()],
            response_formats: vec![response_format],
            stream_modes: vec![stream_mode],
            tasks: vec![SpeechTask::TextToSpeech],
            reference_forms: vec![ReferenceForm::None],
            voice_name_policy,
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
            session_capacity: [None, None],
            health,
            probe: ProbeState::pending(),
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
            homogeneous_media_http: build_content_blind_media_cohorts(&records, None),
            voice_owner: None,
            records,
            admission: AdmissionController::new(
                admission,
                [Some(admission), None, None, None, None, None],
            ),
            selector,
            health_client: client.clone(),
            http_client: client,
            metrics: crate::metrics::RouterMetrics::new(),
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
        pool.admission = AdmissionController::new(2, [None, Some(2), None, None, None, None]);
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
    fn speech_websocket_default_resolution_uses_compatible_profiles() {
        let build_record = |ordinal, model, format, stream| {
            let mut record = record_with_profile(
                ordinal,
                "local",
                model,
                speech_websocket_profile(model, format, stream),
            );
            Arc::get_mut(&mut record)
                .expect("new test record is uniquely owned")
                .session_capacity[0] = Some(SessionCapacity {
                limit: 1,
                semaphore: Arc::new(Semaphore::new(1)),
            });
            record
        };
        let pcm = build_record(
            0,
            "pcm-model",
            SpeechResponseFormat::Pcm,
            StreamMode::Streaming,
        );
        let wav = build_record(
            1,
            "wav-model",
            SpeechResponseFormat::Wav,
            StreamMode::NonStreaming,
        );
        let mut compatible = pool(RoutingStrategy::RoundRobin, vec![pcm, wav], 2);
        compatible.admission = AdmissionController::new(2, [None, None, None, None, Some(2), None]);
        let requirement = RouteRequirement::new(
            ProfileRequirement::SpeechWebsocket {
                model: ModelSelection::UnresolvedDefault,
                response_format: Some(SpeechResponseFormat::Pcm),
                stream_mode: StreamMode::Streaming,
                task: Some(SpeechTask::TextToSpeech),
                reference_forms: vec![ReferenceForm::None],
                named_voice: false,
            },
            TrustDomain::new(String::from("local")),
        );
        let lease = compatible
            .dispatch_session(
                compatible
                    .try_admit(CapacityClass::SpeechWebsocket, 1)
                    .expect("admit speech session"),
                &requirement,
            )
            .expect("dispatch compatible default");
        assert_eq!(lease.registration_ordinal(), 0);
        drop(lease);

        let mut ambiguous = pool(
            RoutingStrategy::RoundRobin,
            vec![
                build_record(0, "pcm-a", SpeechResponseFormat::Pcm, StreamMode::Streaming),
                build_record(1, "pcm-b", SpeechResponseFormat::Pcm, StreamMode::Streaming),
            ],
            2,
        );
        ambiguous.admission = AdmissionController::new(2, [None, None, None, None, Some(2), None]);
        assert!(matches!(
            ambiguous.dispatch_session(
                ambiguous
                    .try_admit(CapacityClass::SpeechWebsocket, 1)
                    .expect("admit ambiguous speech session"),
                &requirement,
            ),
            Err(DispatchError::AmbiguousModel)
        ));
    }

    #[test]
    fn speech_websocket_named_voice_policy_follows_the_selected_model() {
        let build_record = |ordinal, model, policy| {
            let mut record = record_with_profile(
                ordinal,
                "local",
                model,
                speech_websocket_profile_with_policy(
                    model,
                    SpeechResponseFormat::Pcm,
                    StreamMode::Streaming,
                    policy,
                ),
            );
            Arc::get_mut(&mut record)
                .expect("new test record is uniquely owned")
                .session_capacity[0] = Some(SessionCapacity {
                limit: 1,
                semaphore: Arc::new(Semaphore::new(1)),
            });
            record
        };
        let mut pool = pool(
            RoutingStrategy::RoundRobin,
            vec![
                build_record(0, "custom", VoiceNamePolicy::Preset),
                build_record(1, "base", VoiceNamePolicy::Uploaded),
            ],
            2,
        );
        pool.admission = AdmissionController::new(2, [None, None, None, None, Some(2), None]);
        let requirement = |model| {
            RouteRequirement::new(
                ProfileRequirement::SpeechWebsocket {
                    model,
                    response_format: Some(SpeechResponseFormat::Pcm),
                    stream_mode: StreamMode::Streaming,
                    task: Some(SpeechTask::TextToSpeech),
                    reference_forms: Vec::new(),
                    named_voice: true,
                },
                TrustDomain::new(String::from("local")),
            )
        };

        let preset = pool
            .dispatch_session(
                pool.try_admit(CapacityClass::SpeechWebsocket, 1)
                    .expect("preset admission"),
                &requirement(ModelSelection::Explicit(String::from("custom"))),
            )
            .expect("preset dispatch");
        assert_eq!(preset.registration_ordinal(), 0);
        drop(preset);

        assert!(matches!(
            pool.dispatch_session(
                pool.try_admit(CapacityClass::SpeechWebsocket, 1)
                    .expect("ambiguous admission"),
                &requirement(ModelSelection::UnresolvedDefault),
            ),
            Err(DispatchError::AmbiguousModel)
        ));
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
        assert_eq!(
            pool.admission.available(),
            (0, [Some(0), None, None, None, None, None])
        );
        assert_eq!(pool.records[0].load(), 1);
        drop(lease);
        assert_eq!(
            pool.admission.available(),
            (1, [Some(1), None, None, None, None, None])
        );
        assert_eq!(pool.records[0].load(), 0);
    }

    #[test]
    fn admission_rejections_are_counted_at_the_saturated_resource() {
        let mut class_limited = pool(
            RoutingStrategy::RoundRobin,
            vec![record(0, "local", "omni")],
            2,
        );
        class_limited.admission =
            AdmissionController::new(2, [Some(1), None, None, None, None, None]);
        let held = class_limited
            .try_admit(CapacityClass::GenerationHttp, 1)
            .expect("hold generation admission");
        assert!(matches!(
            class_limited.try_admit(CapacityClass::GenerationHttp, 1),
            Err(AdmissionError::Overloaded)
        ));
        assert_eq!(
            class_limited
                .metrics
                .rejections(Rejection::GenerationAdmission),
            1
        );
        assert_eq!(
            class_limited.metrics.rejections(Rejection::GlobalAdmission),
            0
        );
        drop(held);

        let global_limited = pool(
            RoutingStrategy::RoundRobin,
            vec![record(0, "local", "omni")],
            1,
        );
        let held = global_limited
            .try_admit_envelope()
            .expect("hold global admission");
        assert!(matches!(
            global_limited.try_admit_envelope(),
            Err(AdmissionError::Overloaded)
        ));
        assert_eq!(
            global_limited
                .metrics
                .rejections(Rejection::GlobalAdmission),
            1
        );
        drop(held);
    }

    #[test]
    fn websocket_sessions_hold_exact_worker_capacity_and_active_load() {
        let mut record = record_with_profile(0, "local", "omni", ServiceProfile::RealtimeWebsocket);
        Arc::get_mut(&mut record)
            .expect("new test record is uniquely owned")
            .session_capacity[1] = Some(SessionCapacity {
            limit: 1,
            semaphore: Arc::new(Semaphore::new(1)),
        });
        let mut pool = pool(RoutingStrategy::LeastRequests, vec![Arc::clone(&record)], 2);
        pool.admission = AdmissionController::new(2, [None, None, None, None, None, Some(2)]);
        let requirement = RouteRequirement::new(
            ProfileRequirement::RealtimeWebsocket { model: None },
            TrustDomain::new(String::from("local")),
        );

        let first = pool
            .dispatch_session(
                pool.try_admit(CapacityClass::RealtimeWebsocket, 1)
                    .expect("first admission"),
                &requirement,
            )
            .expect("first session");
        assert_eq!(record.load(), 1);
        let snapshot = pool.operations_snapshot();
        assert_eq!(snapshot.workers[0].active_requests, 1);
        assert_eq!(snapshot.workers[0].session_capacity.len(), 1);
        assert_eq!(
            snapshot.workers[0].session_capacity[0].class,
            CapacityClass::RealtimeWebsocket
        );
        assert_eq!(snapshot.workers[0].session_capacity[0].limit, 1);
        assert_eq!(snapshot.workers[0].session_capacity[0].in_flight, 1);
        assert!(matches!(
            pool.dispatch_session(
                pool.try_admit(CapacityClass::RealtimeWebsocket, 1)
                    .expect("second admission"),
                &requirement,
            ),
            Err(DispatchError::Overloaded)
        ));
        assert_eq!(
            pool.metrics.rejections(Rejection::RealtimeWebsocketWorker),
            1
        );

        drop(first);
        let reused = pool
            .dispatch_session(
                pool.try_admit(CapacityClass::RealtimeWebsocket, 1)
                    .expect("reused admission"),
                &requirement,
            )
            .expect("released session capacity is reusable");
        assert_eq!(record.load(), 1);
        drop(reused);
        assert_eq!(record.load(), 0);
    }

    #[test]
    fn realtime_explicit_model_never_falls_back_when_its_worker_is_full() {
        let build_record = |ordinal, model| {
            let mut record =
                record_with_profile(ordinal, "local", model, ServiceProfile::RealtimeWebsocket);
            Arc::get_mut(&mut record)
                .expect("new test record is uniquely owned")
                .session_capacity[1] = Some(SessionCapacity {
                limit: 1,
                semaphore: Arc::new(Semaphore::new(1)),
            });
            record
        };
        let alpha = build_record(0, "omni-alpha");
        let beta = build_record(1, "omni-beta");
        let mut pool = pool(
            RoutingStrategy::RoundRobin,
            vec![Arc::clone(&alpha), Arc::clone(&beta)],
            2,
        );
        pool.admission = AdmissionController::new(2, [None, None, None, None, None, Some(2)]);
        let requirement = RouteRequirement::new(
            ProfileRequirement::RealtimeWebsocket {
                model: Some(String::from("omni-beta")),
            },
            TrustDomain::new(String::from("local")),
        );

        let preferred = pool
            .dispatch_session(
                pool.try_admit(CapacityClass::RealtimeWebsocket, 1)
                    .expect("preferred admission"),
                &requirement,
            )
            .expect("select explicitly requested worker");
        assert_eq!(preferred.registration_ordinal(), 1);

        assert!(matches!(
            pool.dispatch_session(
                pool.try_admit(CapacityClass::RealtimeWebsocket, 1)
                    .expect("second admission"),
                &requirement,
            ),
            Err(DispatchError::Overloaded)
        ));
        drop(preferred);

        beta.health.store(WorkerHealth::Unhealthy);
        assert!(matches!(
            pool.dispatch_session(
                pool.try_admit(CapacityClass::RealtimeWebsocket, 1)
                    .expect("unhealthy admission"),
                &requirement,
            ),
            Err(DispatchError::Unavailable)
        ));
        assert_eq!((alpha.load(), beta.load()), (0, 0));
    }

    #[test]
    fn operations_snapshot_reads_exact_permits_and_releases_with_the_lease() {
        let pool = pool(
            RoutingStrategy::RoundRobin,
            vec![record(0, "local", "omni")],
            4,
        );
        let initial = pool.operations_snapshot();
        assert_eq!(initial.admission[0].class, AdmissionClass::Global);
        assert_eq!(initial.admission[0].limit, 4);
        assert_eq!(initial.admission[0].in_flight, 0);
        assert_eq!(
            initial.admission[1].class,
            AdmissionClass::Service(CapacityClass::GenerationHttp)
        );
        assert_eq!(initial.admission[1].limit, 4);
        assert_eq!(initial.admission[2].limit, 0);
        assert_eq!(initial.workers[0].worker_id, "worker-0");
        assert_eq!(initial.workers[0].registration_ordinal, 0);
        assert!(!initial.workers[0].voice_owner);
        assert_eq!(initial.workers[0].health, WorkerHealth::Healthy);
        assert!(initial.workers[0].routable);
        assert_eq!(initial.workers[0].active_requests, 0);
        assert!(initial.workers[0].session_capacity.is_empty());

        let lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::GenerationHttp, 1)
                    .expect("snapshot admission"),
                &requirement("omni", "local"),
            )
            .expect("snapshot dispatch");
        let occupied = pool.operations_snapshot();
        assert_eq!(occupied.admission[0].in_flight, 1);
        assert_eq!(occupied.admission[1].in_flight, 1);
        assert_eq!(occupied.workers[0].active_requests, 1);

        drop(lease);
        let released = pool.operations_snapshot();
        assert_eq!(released.admission[0].in_flight, 0);
        assert_eq!(released.admission[1].in_flight, 0);
        assert_eq!(released.workers[0].active_requests, 0);

        pool.records[0].health.store(WorkerHealth::Unhealthy);
        pool.drain();
        let drained = pool.operations_snapshot();
        assert_eq!(drained.workers[0].health, WorkerHealth::Unhealthy);
        assert!(!drained.workers[0].routable);
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
            session_capacity: [None, None],
            health,
            probe: ProbeState::pending(),
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
            homogeneous_media_http: build_content_blind_media_cohorts(&records, None),
            voice_owner: None,
            records,
            admission: AdmissionController::new(
                8,
                [Some(8), Some(8), Some(8), Some(8), None, None],
            ),
            selector,
            health_client: client.clone(),
            http_client: client,
            metrics: crate::metrics::RouterMetrics::new(),
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

    fn voice_speech_record(ordinal: usize) -> Arc<WorkerRecord> {
        let health = AtomicHealth::unknown();
        health.store(WorkerHealth::Healthy);
        Arc::new(WorkerRecord {
            worker_id: WorkerId::new(format!("voice-{ordinal}")),
            default_model_id: Some(String::from("tts")),
            registration_id: RegistrationId::from_startup_ordinal(ordinal),
            target: ResolvedTarget::from_parts(
                &format!("http://127.0.0.1:{}/", 13_000 + ordinal),
                "/health",
            )
            .expect("voice target"),
            trust_domain: TrustDomain::new(String::from("local")),
            profiles: vec![
                ServiceProfile::SpeechHttp {
                    model_ids: vec![String::from("tts")],
                    response_formats: vec![SpeechResponseFormat::Wav],
                    stream_modes: vec![StreamMode::NonStreaming],
                    tasks: vec![SpeechTask::TextToSpeech],
                    reference_forms: vec![ReferenceForm::None],
                    voice_name_policy: VoiceNamePolicy::Uploaded,
                },
                ServiceProfile::SpeechBatch {
                    model_ids: vec![String::from("tts")],
                    response_formats: vec![SpeechResponseFormat::Wav],
                    tasks: vec![SpeechTask::TextToSpeech],
                    reference_forms: vec![ReferenceForm::None],
                    voice_name_policy: VoiceNamePolicy::Uploaded,
                    max_batch_size: 8,
                },
                ServiceProfile::SpeechWebsocket {
                    model_ids: vec![String::from("tts")],
                    response_formats: vec![SpeechResponseFormat::Pcm],
                    stream_modes: vec![StreamMode::NonStreaming],
                    tasks: vec![SpeechTask::TextToSpeech],
                    reference_forms: vec![ReferenceForm::None],
                    voice_name_policy: VoiceNamePolicy::Uploaded,
                },
            ],
            active_requests: AtomicUsize::new(0),
            session_capacity: [
                Some(SessionCapacity {
                    limit: 1,
                    semaphore: Arc::new(Semaphore::new(1)),
                }),
                None,
            ],
            health,
            probe: ProbeState::pending(),
            immediate_probe: Notify::new(),
        })
    }

    fn speech_requirement(named_voice: bool) -> RouteRequirement {
        RouteRequirement::new(
            ProfileRequirement::SpeechHttp {
                model: ModelSelection::Explicit(String::from("tts")),
                response_format: SpeechResponseFormat::Wav,
                stream_mode: StreamMode::NonStreaming,
                task: Some(SpeechTask::TextToSpeech),
                reference_forms: if named_voice {
                    Vec::new()
                } else {
                    vec![ReferenceForm::None]
                },
                named_voice,
            },
            TrustDomain::new(String::from("local")),
        )
    }

    fn batch_requirement(named_voice: bool) -> RouteRequirement {
        RouteRequirement::new(
            ProfileRequirement::SpeechBatch {
                models: vec![ModelSelection::Explicit(String::from("tts"))],
                response_formats: vec![SpeechResponseFormat::Wav],
                tasks: vec![SpeechTask::TextToSpeech],
                reference_forms: if named_voice {
                    Vec::new()
                } else {
                    vec![ReferenceForm::None]
                },
                named_voice,
                batch_size: 1,
            },
            TrustDomain::new(String::from("local")),
        )
    }

    fn speech_websocket_requirement(named_voice: bool) -> RouteRequirement {
        RouteRequirement::new(
            ProfileRequirement::SpeechWebsocket {
                model: ModelSelection::Explicit(String::from("tts")),
                response_format: Some(SpeechResponseFormat::Pcm),
                stream_mode: StreamMode::NonStreaming,
                task: Some(SpeechTask::TextToSpeech),
                reference_forms: if named_voice {
                    Vec::new()
                } else {
                    vec![ReferenceForm::None]
                },
                named_voice,
            },
            TrustDomain::new(String::from("local")),
        )
    }

    #[test]
    fn voice_owner_dispatch_is_exact_and_mixed_speech_requires_classification() {
        for strategy in [RoutingStrategy::RoundRobin, RoutingStrategy::LeastRequests] {
            let owner = voice_speech_record(0);
            let mut non_owner = voice_speech_record(1);
            let non_owner_record =
                Arc::get_mut(&mut non_owner).expect("new test record is uniquely owned");
            non_owner_record.default_model_id = Some(String::from("other"));
            for profile in &mut non_owner_record.profiles {
                match profile {
                    ServiceProfile::SpeechHttp { model_ids, .. }
                    | ServiceProfile::SpeechBatch { model_ids, .. }
                    | ServiceProfile::SpeechWebsocket { model_ids, .. } => {
                        model_ids.push(String::from("other"));
                    }
                    _ => panic!("voice fixture contains only speech profiles"),
                }
            }
            let mut pool = media_pool(vec![Arc::clone(&owner), Arc::clone(&non_owner)]);
            pool.selector = Selector::new(strategy, pool.records.len());
            pool.voice_owner = Some(Arc::clone(&owner));
            pool.homogeneous_media_http =
                build_content_blind_media_cohorts(&pool.records, pool.voice_owner.as_ref());
            pool.admission =
                AdmissionController::new(8, [None, Some(4), Some(4), None, Some(4), None]);

            assert!(pool.voice_owner_ready());
            let snapshot = pool.operations_snapshot();
            assert!(snapshot.workers[0].voice_owner);
            assert!(!snapshot.workers[1].voice_owner);
            assert!(
                pool.content_blind_media_http(
                    &TrustDomain::new(String::from("local")),
                    crate::config::HttpMediaRoute::Speech,
                )
                .is_none()
            );

            let mut omitted_http = speech_requirement(true);
            let ProfileRequirement::SpeechHttp { model, .. } = &mut omitted_http.profile else {
                panic!("speech requirement")
            };
            *model = ModelSelection::UnresolvedDefault;
            let omitted_http = pool
                .dispatch(
                    pool.try_admit(CapacityClass::SpeechHttp, 1)
                        .expect("owner-bound default admission"),
                    &omitted_http,
                )
                .expect("owner default is deterministic");
            assert_eq!(omitted_http.registration_ordinal(), 0);
            drop(omitted_http);

            for (class, named, stateless) in [
                (
                    CapacityClass::SpeechHttp,
                    speech_requirement(true),
                    speech_requirement(false),
                ),
                (
                    CapacityClass::SpeechBatch,
                    batch_requirement(true),
                    batch_requirement(false),
                ),
            ] {
                let named = pool
                    .dispatch(
                        pool.try_admit(class, 1).expect("named voice admission"),
                        &named,
                    )
                    .expect("named voice dispatch");
                assert_eq!(named.registration_ordinal(), 0);
                drop(named);

                let mut reached_non_owner = false;
                for _ in 0..2 {
                    let lease = pool
                        .dispatch(
                            pool.try_admit(class, 1).expect("stateless admission"),
                            &stateless,
                        )
                        .expect("stateless policy dispatch");
                    reached_non_owner |= lease.registration_ordinal() == 1;
                }
                assert!(reached_non_owner);
            }

            let websocket = speech_websocket_requirement(true);
            let session = pool
                .dispatch_session(
                    pool.try_admit(CapacityClass::SpeechWebsocket, 1)
                        .expect("named voice session admission"),
                    &websocket,
                )
                .expect("named voice session dispatch");
            assert_eq!(session.registration_ordinal(), 0);
            drop(session);

            let mut omitted_websocket = speech_websocket_requirement(true);
            let ProfileRequirement::SpeechWebsocket { model, .. } = &mut omitted_websocket.profile
            else {
                panic!("speech websocket requirement")
            };
            *model = ModelSelection::UnresolvedDefault;
            let session = pool
                .dispatch_session(
                    pool.try_admit(CapacityClass::SpeechWebsocket, 1)
                        .expect("owner-bound session admission"),
                    &omitted_websocket,
                )
                .expect("owner session default is deterministic");
            assert_eq!(session.registration_ordinal(), 0);
            drop(session);

            let control = pool
                .dispatch_voice_owner(pool.try_admit_envelope().expect("voice admission"))
                .expect("exact owner dispatch");
            assert_eq!(control.registration_ordinal(), 0);
            assert_eq!(owner.load(), 1);
            drop(control);
            assert_eq!(owner.load(), 0);

            owner.health.store(WorkerHealth::Unhealthy);
            assert!(!pool.voice_owner_ready());
            assert_eq!(
                pool.dispatch_voice_owner(
                    pool.try_admit_envelope()
                        .expect("unhealthy owner admission"),
                )
                .err(),
                Some(DispatchError::Unavailable)
            );
        }
    }

    #[test]
    fn owner_only_speech_keeps_content_blind_proof() {
        let owner = voice_speech_record(0);
        let mut pool = media_pool(vec![Arc::clone(&owner)]);
        pool.voice_owner = Some(owner);
        pool.homogeneous_media_http =
            build_content_blind_media_cohorts(&pool.records, pool.voice_owner.as_ref());
        assert!(
            pool.content_blind_media_http(
                &TrustDomain::new(String::from("local")),
                crate::config::HttpMediaRoute::Speech,
            )
            .is_some()
        );
    }

    #[test]
    fn uploaded_voice_speech_without_an_owner_requires_classification() {
        let pool = media_pool(vec![voice_speech_record(0)]);

        assert!(
            pool.content_blind_media_http(
                &TrustDomain::new(String::from("local")),
                crate::config::HttpMediaRoute::Speech,
            )
            .is_none()
        );
    }

    #[test]
    fn unrelated_voice_owner_does_not_disable_preset_content_blind_proof() {
        let mut owner = voice_speech_record(0);
        Arc::get_mut(&mut owner)
            .expect("new test record is uniquely owned")
            .trust_domain = TrustDomain::new(String::from("owner"));
        let preset = record_with_profile(
            1,
            "local",
            "custom",
            named_speech_profile("custom", VoiceNamePolicy::Preset),
        );
        let mut pool = media_pool(vec![Arc::clone(&owner), preset]);
        pool.voice_owner = Some(owner);
        pool.homogeneous_media_http =
            build_content_blind_media_cohorts(&pool.records, pool.voice_owner.as_ref());

        assert!(
            pool.content_blind_media_http(
                &TrustDomain::new(String::from("local")),
                crate::config::HttpMediaRoute::Speech,
            )
            .is_some()
        );
    }

    #[test]
    fn preset_voice_uses_normal_selection_with_an_uploaded_voice_owner() {
        let owner = voice_speech_record(0);
        let preset = record_with_profile(
            1,
            "local",
            "custom",
            named_speech_profile("custom", VoiceNamePolicy::Preset),
        );
        let mut pool = media_pool(vec![Arc::clone(&owner), preset]);
        pool.voice_owner = Some(owner);

        let lease = pool
            .dispatch(
                pool.try_admit(CapacityClass::SpeechHttp, 1)
                    .expect("preset admission"),
                &named_speech_requirement(ModelSelection::Explicit(String::from("custom"))),
            )
            .expect("preset dispatch");
        assert_eq!(lease.registration_ordinal(), 1);
    }

    #[test]
    fn uploaded_voice_owner_must_declare_its_default_model() {
        let mut owner = voice_speech_record(0);
        Arc::get_mut(&mut owner)
            .expect("new test record is uniquely owned")
            .default_model_id = None;
        let mut pool = media_pool(vec![Arc::clone(&owner)]);
        pool.voice_owner = Some(owner);
        pool.admission = AdmissionController::new(8, [None, Some(4), Some(4), None, Some(4), None]);

        let mut http = speech_requirement(true);
        let ProfileRequirement::SpeechHttp { model, .. } = &mut http.profile else {
            panic!("speech requirement")
        };
        *model = ModelSelection::UnresolvedDefault;
        assert!(matches!(
            pool.dispatch(
                pool.try_admit(CapacityClass::SpeechHttp, 1)
                    .expect("HTTP admission"),
                &http,
            ),
            Err(DispatchError::AmbiguousModel)
        ));

        let mut websocket = speech_websocket_requirement(true);
        let ProfileRequirement::SpeechWebsocket { model, .. } = &mut websocket.profile else {
            panic!("speech WebSocket requirement")
        };
        *model = ModelSelection::UnresolvedDefault;
        assert!(matches!(
            pool.dispatch_session(
                pool.try_admit(CapacityClass::SpeechWebsocket, 1)
                    .expect("WebSocket admission"),
                &websocket,
            ),
            Err(DispatchError::AmbiguousModel)
        ));
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
        let mut pool = media_pool(vec![preset, Arc::clone(&uploaded)]);
        pool.voice_owner = Some(uploaded);

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

    #[test]
    fn uploaded_voice_requires_an_owner() {
        let uploaded = record_with_profile(
            0,
            "local",
            "base",
            named_speech_profile("base", VoiceNamePolicy::Uploaded),
        );
        let pool = media_pool(vec![uploaded]);

        assert!(matches!(
            pool.dispatch(
                pool.try_admit(CapacityClass::SpeechHttp, 1)
                    .expect("uploaded admission"),
                &named_speech_requirement(ModelSelection::Explicit(String::from("base"))),
            ),
            Err(DispatchError::NoEligibleProfile)
        ));
    }
}
