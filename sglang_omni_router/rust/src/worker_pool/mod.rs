mod admission;
mod health;
pub(crate) mod profile;
mod resolver;
mod selection;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::Notify;

use crate::config::{Config, RoutingStrategy};

pub(crate) use admission::{AdmissionError, AdmissionLease, DispatchError, RequestLease};
pub(crate) use health::{HealthSupervisor, WorkerHealth};
pub(crate) use profile::{
    ChatAudioFormat, MediaPlacement, MessageContentForm, ModelSelection, ProfileRequirement,
    RouteRequirement, ServiceClass, TrustDomain,
};
pub(crate) use resolver::ResolvedTarget;

use admission::AdmissionController;
use health::AtomicHealth;
use profile::{
    MAX_WORKERS, RegistrationId, ServiceProfile, WorkerId, generation_cohort_is_homogeneous,
};
use resolver::{build_generation_client, build_health_client};
use selection::{Selector, SelectorGuard};

/// One static startup registration with independently updated health and load.
pub(super) struct WorkerRecord {
    worker_id: WorkerId,
    default_model_id: String,
    registration_id: RegistrationId,
    target: ResolvedTarget,
    trust_domain: TrustDomain,
    profiles: Vec<ServiceProfile>,
    active_requests: AtomicUsize,
    health: AtomicHealth,
    immediate_probe: Notify,
}

impl WorkerRecord {
    fn has_profile(&self, requirement: &RouteRequirement) -> bool {
        self.profiles
            .iter()
            .any(|profile| profile.matches(&requirement.profile, &self.default_model_id))
    }

    fn is_routable(&self) -> bool {
        self.health.load() == WorkerHealth::Healthy
    }

    fn load(&self) -> usize {
        self.active_requests.load(Ordering::Relaxed)
    }

    fn increment_load(&self) {
        self.active_requests.fetch_add(1, Ordering::Relaxed);
    }

    fn decrement_load(&self) {
        let previous =
            self.active_requests
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                    current.checked_sub(1)
                });
        debug_assert!(previous.is_ok(), "worker load cannot underflow");
    }
}

/// Static-membership generation worker pool with bounded admission,
/// deterministic policy state, and independently owned health and load.
pub(crate) struct WorkerPool {
    records: Vec<Arc<WorkerRecord>>,
    admission: AdmissionController,
    selector: Selector,
    homogeneous_generation_http: Vec<HomogeneousGenerationCohort>,
    health_client: reqwest::Client,
    generation_client: reqwest::Client,
}

struct HomogeneousGenerationCohort {
    trust_domain: TrustDomain,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DefaultModelResolution<'a> {
    NoService,
    Unique(&'a str),
    Ambiguous,
}

/// Startup proof that chat body inspection cannot change the route cohort.
pub(crate) struct ContentBlindGenerationHttp<'a> {
    pool: &'a WorkerPool,
    trust: &'a TrustDomain,
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
        let generation_client = build_generation_client(
            config.http_generation.connect_timeout(),
            config.http_generation.pool_idle_timeout(),
            config.http_generation.pool_max_idle_per_host,
        )
        .map_err(crate::error::RouterError::GenerationClient)?;
        let admission = AdmissionController::new(
            usize::try_from(config.admission.global)
                .map_err(|_| crate::error::RouterError::WorkerPoolInvariant)?,
            usize::try_from(config.admission.generation_http)
                .map_err(|_| crate::error::RouterError::WorkerPoolInvariant)?,
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
        let selector = Selector::new(config.router.strategy, records.len());
        Ok(Self {
            records,
            admission,
            selector,
            homogeneous_generation_http,
            health_client,
            generation_client,
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

    pub(crate) fn generation_client(&self) -> reqwest::Client {
        self.generation_client.clone()
    }

    pub(crate) fn try_admit(&self) -> Result<AdmissionLease, AdmissionError> {
        self.admission.try_admit()
    }

    pub(crate) fn dispatch(
        &self,
        admission: AdmissionLease,
        requirement: &RouteRequirement,
    ) -> Result<RequestLease, DispatchError> {
        let mut matching = [false; MAX_WORKERS];
        for (index, record) in self.records.iter().enumerate() {
            matching[index] = &record.trust_domain == requirement.trust_domain()
                && record.has_profile(requirement);
        }
        let profile_found = matching[..self.records.len()].contains(&true);
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
        .ok_or(DispatchError::Unavailable)?;
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

    pub(crate) fn resolve_default_model_id(
        &self,
        trust: &TrustDomain,
        _service: ServiceClass,
    ) -> DefaultModelResolution<'_> {
        let mut resolved = None;
        for record in &self.records {
            if &record.trust_domain != trust {
                continue;
            }
            match resolved {
                None => resolved = Some(record.default_model_id.as_str()),
                Some(current) if current == record.default_model_id => {}
                Some(_) => return DefaultModelResolution::Ambiguous,
            }
        }
        resolved.map_or(
            DefaultModelResolution::NoService,
            DefaultModelResolution::Unique,
        )
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

    pub(crate) fn generation_http_ready(&self, trust: &TrustDomain) -> bool {
        self.records
            .iter()
            .any(|record| &record.trust_domain == trust && record.is_routable())
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
        self.pool
            .dispatch_matching(admission, true, |record| &record.trust_domain == self.trust)
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
        let members = records
            .iter()
            .filter(|candidate| candidate.trust_domain == record.trust_domain)
            .map(|candidate| {
                (
                    candidate.default_model_id.as_str(),
                    candidate.profiles.as_slice(),
                )
            });
        if generation_cohort_is_homogeneous(members) {
            result.push(HomogeneousGenerationCohort {
                trust_domain: record.trust_domain.clone(),
            });
        }
    }
    result
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::thread;

    use super::profile::{
        InputModality, MessageContentForm, ModelSelection, OutputModality, ProfileRequirement,
        ServiceProfile, StreamMode,
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
            default_model_id: model.to_owned(),
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
            records,
            admission: AdmissionController::new(admission, admission),
            selector,
            health_client: client.clone(),
            generation_client: client,
        }
    }

    fn dispatch_subset(pool: &WorkerPool, ordinals: &[usize]) -> usize {
        let lease = pool
            .dispatch_matching(pool.try_admit().expect("admit subset"), true, |record| {
                ordinals.contains(&record.registration_id.startup_ordinal())
            })
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
            |ServiceProfile::GenerationHttp { model_ids, .. }| {
                model_ids.push(String::from("other"));
            },
            |ServiceProfile::GenerationHttp {
                 message_content_forms,
                 ..
             }| {
                message_content_forms.push(MessageContentForm::TypedParts);
            },
            |ServiceProfile::GenerationHttp {
                 media_placements, ..
             }| {
                media_placements.push(MediaPlacement::TypedParts);
            },
            |ServiceProfile::GenerationHttp {
                 input_modalities, ..
             }| {
                input_modalities.push(InputModality::Image);
            },
            |ServiceProfile::GenerationHttp {
                 output_modalities,
                 chat_audio_formats,
                 ..
             }| {
                output_modalities.push(OutputModality::Audio);
                chat_audio_formats.push(ChatAudioFormat::Wav);
            },
            |ServiceProfile::GenerationHttp { stream_modes, .. }| {
                stream_modes.push(StreamMode::Streaming);
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
                pool.try_admit().expect("admit first"),
                &requirement("omni", "local"),
            )
            .expect("first dispatch");
        let second = pool
            .dispatch(
                pool.try_admit().expect("admit second"),
                &requirement("omni", "local"),
            )
            .expect("second dispatch");
        assert_ne!(first.registration_ordinal(), second.registration_ordinal());
        drop(first);
        drop(second);
        records[0].health.store(WorkerHealth::Unhealthy);
        records[1].health.store(WorkerHealth::Unhealthy);
        let unavailable = pool.dispatch(
            pool.try_admit().expect("admit unavailable"),
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
                    pool.try_admit().expect("admit sparse round robin"),
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
        let trust = TrustDomain::new(String::from("local"));
        let mut selected = Vec::new();
        for _ in 0..6 {
            let lease = pool
                .content_blind_generation_http(&trust)
                .expect("homogeneous cohort")
                .dispatch(pool.try_admit().expect("admit sparse least requests"))
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
        middle.increment_load();
        let pool = pool(RoutingStrategy::LeastRequests, records, 8);
        let trust = TrustDomain::new(String::from("local"));
        let mut selected = Vec::new();
        for _ in 0..6 {
            let lease = pool
                .content_blind_generation_http(&trust)
                .expect("homogeneous cohort")
                .dispatch(pool.try_admit().expect("admit tied least requests"))
                .expect("dispatch tied least requests");
            selected.push(lease.registration_ordinal());
            drop(lease);
        }
        assert_eq!(selected, [0, 2, 0, 2, 0, 2]);
        middle.decrement_load();
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
                let admission = pool.try_admit().expect("concurrent admission");
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
    fn heterogeneous_default_model_routes_to_the_correlated_capable_worker() {
        let mut multimodal = profile("omni");
        let ServiceProfile::GenerationHttp {
            message_content_forms,
            media_placements,
            input_modalities,
            ..
        } = &mut multimodal;
        message_content_forms.push(MessageContentForm::TypedParts);
        media_placements.push(MediaPlacement::TypedParts);
        input_modalities.push(InputModality::Image);
        let pool = pool(
            RoutingStrategy::RoundRobin,
            vec![
                record(0, "local", "omni"),
                record_with_profile(1, "local", "omni", multimodal),
            ],
            2,
        );
        let requirement = RouteRequirement::new(
            ProfileRequirement::GenerationHttp {
                model: ModelSelection::WorkerDefault {
                    expected_model_id: String::from("omni"),
                },
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
                pool.try_admit().expect("admit heterogeneous request"),
                &requirement,
            )
            .expect("dispatch heterogeneous default");
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
                pool.try_admit().expect("admit"),
                &requirement("omni", "local"),
            )
            .expect("dispatch");
        assert_eq!(pool.admission.available(), (0, 0));
        assert_eq!(pool.records[0].load(), 1);
        drop(lease);
        assert_eq!(pool.admission.available(), (1, 1));
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
        let admission = pool.try_admit().expect("admit before drain");

        pool.drain();

        assert!(matches!(pool.try_admit(), Err(AdmissionError::Draining)));
        let lease = pool
            .content_blind_generation_http(&trust)
            .expect("homogeneous cohort")
            .dispatch(admission)
            .expect("admitted request may dispatch during drain");
        drop(lease);
    }
}
