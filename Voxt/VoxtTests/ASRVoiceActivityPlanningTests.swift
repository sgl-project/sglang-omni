// ASRVoiceActivityPlanningTests.swift
// Provides shared ASR voice activity planning tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class ASRVoiceActivityPlanningTests: XCTestCase {
    private func withEphemeralDefaults(
        _ body: (UserDefaults) throws -> Void
    ) rethrows {
        let suiteName = "ASRVoiceActivityPlanningTests.\(UUID().uuidString)"
        guard let defaults = UserDefaults(suiteName: suiteName) else {
            XCTFail("Expected ephemeral UserDefaults suite")
            return
        }
        defaults.removePersistentDomain(forName: suiteName)
        defer {
            defaults.removePersistentDomain(forName: suiteName)
        }
        try body(defaults)
    }

    func testLocalVADModeValuesResolveToUserModes() {
        XCTAssertEqual(LocalVADMode.resolved(rawValue: "automatic"), .automatic)
        XCTAssertEqual(LocalVADMode.resolved(rawValue: "auto"), .automatic)
        XCTAssertEqual(LocalVADMode.resolved(rawValue: "silero"), .silero)
        XCTAssertEqual(LocalVADMode.resolved(rawValue: "mlxSilero"), .silero)
        XCTAssertEqual(LocalVADMode.resolved(rawValue: "omni"), .omni)
        XCTAssertEqual(LocalVADMode.resolved(rawValue: "omnivad"), .omni)
        XCTAssertEqual(LocalVADMode.resolved(rawValue: "omniVAD"), .omni)
        XCTAssertEqual(LocalVADMode.resolved(rawValue: "energy"), .energy)
        XCTAssertEqual(LocalVADMode.resolved(rawValue: "off"), .off)
        XCTAssertEqual(LocalVADMode.resolved(rawValue: "unknown"), .automatic)
    }

    func testSileroModelSupportUsesV6Checkpoint() {
        XCTAssertEqual(SileroVADModelSupport.repo, "mlx-community/silero-vad-v6")
    }

    func testOnlyAutomaticAndSileroModesRequireSileroModel() {
        XCTAssertTrue(ASRVoiceActivityRuntimePolicy.requiresSileroModel(mode: .automatic))
        XCTAssertTrue(ASRVoiceActivityRuntimePolicy.requiresSileroModel(mode: .silero))
        XCTAssertFalse(ASRVoiceActivityRuntimePolicy.requiresSileroModel(mode: .omni))
        XCTAssertFalse(ASRVoiceActivityRuntimePolicy.requiresSileroModel(mode: .energy))
        XCTAssertFalse(ASRVoiceActivityRuntimePolicy.requiresSileroModel(mode: .off))
    }

    func testLocalVADModePersistsInUserDefaults() throws {
        try withEphemeralDefaults { defaults in
            XCTAssertEqual(LocalVADMode.stored(defaults: defaults), .automatic)

            LocalVADMode.save(.energy, defaults: defaults)

            XCTAssertEqual(defaults.string(forKey: AppPreferenceKey.localVADMode), "energy")
            XCTAssertEqual(LocalVADMode.stored(defaults: defaults), .energy)
        }
    }

    func testLocalVADModeSavePostsSettingsChangeNotification() throws {
        try withEphemeralDefaults { defaults in
            let notificationExpectation = expectation(description: "VAD mode change posts settings notification")
            let token = NotificationCenter.default.addObserver(
                forName: .voxtFeatureSettingsDidChange,
                object: nil,
                queue: nil
            ) { _ in
                notificationExpectation.fulfill()
            }
            defer {
                NotificationCenter.default.removeObserver(token)
            }

            LocalVADMode.save(.silero, defaults: defaults)

            wait(for: [notificationExpectation], timeout: 1.0)
        }
    }

    func testVADBackendValuesResolveToLocalBackendKinds() {
        XCTAssertEqual(ASRVoiceActivityBackendKind.resolved(rawValue: "off"), .off)
        XCTAssertEqual(ASRVoiceActivityBackendKind.resolved(rawValue: "energy"), .energy)
        XCTAssertEqual(ASRVoiceActivityBackendKind.resolved(rawValue: "mlxSilero"), .mlxSilero)
        XCTAssertEqual(ASRVoiceActivityBackendKind.resolved(rawValue: "omni"), .omniStream)
        XCTAssertEqual(ASRVoiceActivityBackendKind.resolved(rawValue: "omnivad"), .omniStream)
        XCTAssertEqual(ASRVoiceActivityBackendKind.resolved(rawValue: "omniStream"), .omniStream)
        XCTAssertEqual(ASRVoiceActivityBackendKind.resolved(rawValue: "unknown"), .mlxSilero)
    }

    func testLocalVADModeMapsToEffectiveBackends() {
        XCTAssertEqual(
            ASRVoiceActivityRuntimePolicy.effectiveBackend(mode: .automatic, useCase: .meeting),
            .mlxSilero
        )
        XCTAssertEqual(
            ASRVoiceActivityRuntimePolicy.effectiveBackend(mode: .silero, useCase: .translation),
            .mlxSilero
        )
        XCTAssertEqual(
            ASRVoiceActivityRuntimePolicy.effectiveBackend(mode: .omni, useCase: .transcription),
            .omniStream
        )
        XCTAssertEqual(
            ASRVoiceActivityRuntimePolicy.effectiveBackend(mode: .energy, useCase: .rewrite),
            .energy
        )
        XCTAssertEqual(
            ASRVoiceActivityRuntimePolicy.effectiveBackend(mode: .off, useCase: .transcription),
            .off
        )
    }

    func testUseCaseProfilesRemainDistinctConstants() {
        XCTAssertLessThan(
            ASRVoiceActivityConfiguration.profile(for: .meeting).minSilenceSeconds,
            ASRVoiceActivityConfiguration.profile(for: .translation).minSilenceSeconds
        )
        XCTAssertGreaterThan(
            ASRVoiceActivityConfiguration.profile(for: .rewrite).minSpeechSeconds,
            ASRVoiceActivityConfiguration.profile(for: .transcription).minSpeechSeconds
        )
    }

    func testMeetingSileroSensitivityAdjustsOnlyTheOnsetThreshold() {
        let responsive = MeetingSileroVADSensitivity.responsive.configuration()
        let balanced = MeetingSileroVADSensitivity.balanced.configuration()
        let stable = MeetingSileroVADSensitivity.stable.configuration()

        XCTAssertLessThan(responsive.onsetProbabilityThreshold, balanced.onsetProbabilityThreshold)
        XCTAssertLessThan(balanced.onsetProbabilityThreshold, stable.onsetProbabilityThreshold)
        XCTAssertEqual(responsive.offsetProbabilityThreshold, balanced.offsetProbabilityThreshold)
        XCTAssertEqual(stable.minSilenceSeconds, balanced.minSilenceSeconds)
        // Balanced keeps the historical streaming default (0.5), not the offline meeting
        // profile onset (0.45), so upgrades do not silently increase noise pickup.
        XCTAssertEqual(balanced.onsetProbabilityThreshold, 0.5)
        XCTAssertEqual(MeetingSileroVADSensitivity.responsive.title, AppLocalization.localizedString("Sensitive"))
        XCTAssertEqual(MeetingSileroVADSensitivity.balanced.title, AppLocalization.localizedString("Balanced"))
        XCTAssertEqual(ASRVoiceActivityConfiguration.meeting.onsetProbabilityThreshold, 0.45)
    }

    func testSampleRateConverterResamplesWithoutChangingEmptyOrSameRateInput() {
        XCTAssertEqual(
            ASRVoiceActivitySampleRateConverter.resample(samples: [], from: 48_000, to: 16_000),
            []
        )
        XCTAssertEqual(
            ASRVoiceActivitySampleRateConverter.resample(samples: [0, 1, 0], from: 16_000, to: 16_000),
            [0, 1, 0]
        )
    }

    func testSampleRateConverterUsesLinearInterpolation() {
        let resampled = ASRVoiceActivitySampleRateConverter.resample(
            samples: [0, 1],
            from: 2,
            to: 4
        )

        XCTAssertEqual(resampled.count, 4)
        XCTAssertEqual(resampled[0], 0, accuracy: 0.0001)
        XCTAssertEqual(resampled[1], 0.5, accuracy: 0.0001)
        XCTAssertEqual(resampled[2], 1, accuracy: 0.0001)
        XCTAssertEqual(resampled[3], 1, accuracy: 0.0001)
    }

    func testSampleRateConverterRejectsNonFiniteOrExtremeRates() {
        let samples: [Float] = [0, 1, 0]

        XCTAssertEqual(
            ASRVoiceActivitySampleRateConverter.resample(samples: samples, from: .nan, to: 16_000),
            samples
        )
        XCTAssertEqual(
            ASRVoiceActivitySampleRateConverter.resample(samples: samples, from: 16_000, to: .infinity),
            samples
        )
        XCTAssertEqual(
            ASRVoiceActivitySampleRateConverter.resample(samples: samples, from: 1, to: 16_000),
            samples
        )
    }

    func testSampleRateConverterSanitizesNonFiniteSamplesBeforeModelInput() {
        XCTAssertEqual(
            ASRVoiceActivitySampleRateConverter.resample(
                samples: [0, .nan, .infinity, -.infinity, 1],
                from: 16_000,
                to: 16_000
            ),
            [0, 0, 0, 0, 1]
        )

        let resampled = ASRVoiceActivitySampleRateConverter.resample(
            samples: [0, .nan, 1],
            from: 3,
            to: 6
        )

        XCTAssertEqual(resampled.count, 6)
        XCTAssertTrue(resampled.allSatisfy(\.isFinite))
        XCTAssertEqual(resampled[0], 0, accuracy: 0.0001)
        XCTAssertEqual(resampled[2], 0, accuracy: 0.0001)
        XCTAssertEqual(resampled[4], 1, accuracy: 0.0001)
    }

    func testLocalGatePolicyEnablesLocalMLXWhenModeIsOn() {
        let policy = ASRVoiceActivityRuntimePolicy.localGatePolicy(
            transcriptionEngine: .mlxAudio,
            mode: .automatic
        )

        XCTAssertEqual(policy, .enabled)
        XCTAssertTrue(policy.isEnabled)
    }

    func testLocalGatePolicyDisablesWhenModeIsOff() {
        let policy = ASRVoiceActivityRuntimePolicy.localGatePolicy(
            transcriptionEngine: .mlxAudio,
            mode: .off
        )

        XCTAssertEqual(policy, .disabled(reason: "local-vad-off"))
        XCTAssertFalse(policy.isEnabled)
    }

    func testLocalGatePolicyDisablesForNonLocalASR() {
        let policy = ASRVoiceActivityRuntimePolicy.localGatePolicy(
            transcriptionEngine: .remote,
            mode: .automatic
        )

        XCTAssertEqual(policy, .disabled(reason: "non-local-asr"))
        XCTAssertFalse(policy.isEnabled)
    }

    func testLevelTimingFallsBackWhenLocalVADFramesAreMissing() {
        XCTAssertTrue(
            ASRVoiceActivityRuntimePolicy.shouldUseLevelTiming(
                localVADGateActive: false,
                hasVoiceActivityFrames: false
            )
        )
        XCTAssertFalse(
            ASRVoiceActivityRuntimePolicy.shouldUseLevelTiming(
                localVADGateActive: true,
                hasVoiceActivityFrames: true
            )
        )
        XCTAssertTrue(
            ASRVoiceActivityRuntimePolicy.shouldUseLevelTiming(
                localVADGateActive: true,
                hasVoiceActivityFrames: false
            )
        )
    }

    func testFinalTranscriptionSuppressionRequiresTrustedNoSpeechVAD() {
        XCTAssertTrue(
            ASRVoiceActivityRuntimePolicy.shouldSuppressFinalTranscription(
                localVADGateActive: true,
                observedVoiceActivityFrames: true,
                observedSpeech: false
            )
        )
        XCTAssertFalse(
            ASRVoiceActivityRuntimePolicy.shouldSuppressFinalTranscription(
                localVADGateActive: true,
                observedVoiceActivityFrames: true,
                observedSpeech: true
            )
        )
        XCTAssertFalse(
            ASRVoiceActivityRuntimePolicy.shouldSuppressFinalTranscription(
                localVADGateActive: true,
                observedVoiceActivityFrames: false,
                observedSpeech: false
            )
        )
        XCTAssertFalse(
            ASRVoiceActivityRuntimePolicy.shouldSuppressFinalTranscription(
                localVADGateActive: false,
                observedVoiceActivityFrames: true,
                observedSpeech: false
            )
        )
    }

    func testLocalIntermediateGateRequiresEnabledMLXTrailingSpeechEnd() {
        XCTAssertTrue(
            ASRLocalIntermediateGatePolicy.shouldTriggerIntermediateTranscription(
                transcriptionEngine: .mlxAudio,
                localVADGateActive: true,
                silentDuration: 2.0,
                didTriggerPauseTranscription: false,
                observedSpeechEnd: true
            )
        )
        XCTAssertFalse(
            ASRLocalIntermediateGatePolicy.shouldTriggerIntermediateTranscription(
                transcriptionEngine: .mlxAudio,
                localVADGateActive: false,
                silentDuration: 2.0,
                didTriggerPauseTranscription: false,
                observedSpeechEnd: true
            )
        )
        XCTAssertFalse(
            ASRLocalIntermediateGatePolicy.shouldTriggerIntermediateTranscription(
                transcriptionEngine: .remote,
                localVADGateActive: true,
                silentDuration: 2.0,
                didTriggerPauseTranscription: false,
                observedSpeechEnd: true
            )
        )
        XCTAssertFalse(
            ASRLocalIntermediateGatePolicy.shouldTriggerIntermediateTranscription(
                transcriptionEngine: .mlxAudio,
                localVADGateActive: true,
                silentDuration: 1.99,
                didTriggerPauseTranscription: false,
                observedSpeechEnd: true
            )
        )
        XCTAssertFalse(
            ASRLocalIntermediateGatePolicy.shouldTriggerIntermediateTranscription(
                transcriptionEngine: .mlxAudio,
                localVADGateActive: true,
                silentDuration: 2.0,
                didTriggerPauseTranscription: true,
                observedSpeechEnd: true
            )
        )
        XCTAssertFalse(
            ASRLocalIntermediateGatePolicy.shouldTriggerIntermediateTranscription(
                transcriptionEngine: .mlxAudio,
                localVADGateActive: true,
                silentDuration: 2.0,
                didTriggerPauseTranscription: false,
                observedSpeechEnd: false
            )
        )
    }

    func testEnergyBackendUsesProvidedLevelWhenAvailable() async throws {
        let backend = ASREnergyVoiceActivityBackend(threshold: 0.1)
        let decision = try await backend.decision(
            for: ASRVoiceActivityAudioFrame(
                samples: [0, 0, 0],
                sampleRate: 16_000,
                startSeconds: 0,
                endSeconds: 0.2,
                level: 0.12
            )
        )

        XCTAssertTrue(decision.isSpeech)
        XCTAssertNil(decision.probability)
        XCTAssertEqual(backend.kind, .energy)
    }

    func testEnergyBackendComputesRMSLevelWhenNeeded() async throws {
        let backend = ASREnergyVoiceActivityBackend(threshold: 0.4)
        let decision = try await backend.decision(
            for: ASRVoiceActivityAudioFrame(
                samples: [0.5, -0.5, 0.5, -0.5],
                sampleRate: 16_000,
                startSeconds: 0,
                endSeconds: 0.2
            )
        )

        XCTAssertTrue(decision.isSpeech)
    }

    func testEnergyBackendIgnoresNonFiniteSamples() async throws {
        let backend = ASREnergyVoiceActivityBackend(threshold: 0.4)
        let decision = try await backend.decision(
            for: ASRVoiceActivityAudioFrame(
                samples: [.nan, .infinity, -.infinity, 0.5, -0.5],
                sampleRate: 16_000,
                startSeconds: 0,
                endSeconds: 0.2
            )
        )

        XCTAssertTrue(decision.isSpeech)

        let silentDecision = try await backend.decision(
            for: ASRVoiceActivityAudioFrame(
                samples: [.nan, .infinity, -.infinity],
                sampleRate: 16_000,
                startSeconds: 0,
                endSeconds: 0.2
            )
        )

        XCTAssertFalse(silentDecision.isSpeech)
    }

    func testAudioFrameAndDecisionNormalizeNonFiniteTimingAndLevel() {
        let frame = ASRVoiceActivityAudioFrame(
            samples: [],
            sampleRate: .nan,
            startSeconds: .nan,
            endSeconds: .infinity,
            level: .nan
        )

        XCTAssertEqual(frame.sampleRate, 0)
        XCTAssertEqual(frame.startSeconds, 0)
        XCTAssertEqual(frame.endSeconds, 0)
        XCTAssertNil(frame.level)

        let decision = ASRVoiceActivityFrameDecision(
            startSeconds: .nan,
            endSeconds: .infinity,
            isSpeech: true,
            probability: 0.5
        )

        XCTAssertEqual(decision.startSeconds, 0)
        XCTAssertEqual(decision.endSeconds, 0)
        XCTAssertEqual(decision.durationSeconds, 0)
        XCTAssertEqual(decision.probability, 0.5)
    }

    func testFrameDecisionDropsNonFiniteProbabilities() {
        let nanDecision = ASRVoiceActivityFrameDecision(
            startSeconds: 0,
            endSeconds: 0.2,
            isSpeech: true,
            probability: .nan
        )
        let infiniteDecision = ASRVoiceActivityFrameDecision(
            startSeconds: 0,
            endSeconds: 0.2,
            isSpeech: true,
            probability: .infinity
        )

        XCTAssertNil(nanDecision.probability)
        XCTAssertNil(infiniteDecision.probability)
    }

    func testSegmenterUsesProbabilityHysteresis() {
        var segmenter = ASRVoiceActivitySegmenter(configuration: .balanced)

        XCTAssertEqual(
            segmenter.append(
                ASRVoiceActivityFrameDecision(
                    startSeconds: 0,
                    endSeconds: 0.1,
                    isSpeech: true,
                    probability: 0.49
                )
            ),
            []
        )

        let startEvents = segmenter.append(
            ASRVoiceActivityFrameDecision(
                startSeconds: 0.1,
                endSeconds: 0.4,
                isSpeech: false,
                probability: 0.51
            )
        )

        XCTAssertEqual(startEvents, [.speechStarted(startSeconds: 0)])

        let continuedEvents = segmenter.append(
            ASRVoiceActivityFrameDecision(
                startSeconds: 0.4,
                endSeconds: 0.7,
                isSpeech: false,
                probability: 0.36
            )
        )

        XCTAssertEqual(continuedEvents, [])
    }

    func testSegmenterReturnsResolvedSpeechStateWithHysteresis() {
        var segmenter = ASRVoiceActivitySegmenter(configuration: .balanced)
        _ = segmenter.appendWithResolvedSpeechState(
            ASRVoiceActivityFrameDecision(
                startSeconds: 0,
                endSeconds: 0.2,
                isSpeech: false,
                probability: 0.51
            )
        )

        let continued = segmenter.appendWithResolvedSpeechState(
            ASRVoiceActivityFrameDecision(
                startSeconds: 0.2,
                endSeconds: 0.4,
                isSpeech: false,
                probability: 0.40
            )
        )

        XCTAssertTrue(continued.isSpeech)
        XCTAssertEqual(continued.events, [])
    }

    func testSegmenterEndsSpeechAfterConfiguredSilence() {
        var segmenter = ASRVoiceActivitySegmenter(configuration: .balanced)
        _ = segmenter.append(
            ASRVoiceActivityFrameDecision(
                startSeconds: 1.0,
                endSeconds: 1.4,
                isSpeech: true,
                probability: 0.8
            )
        )

        XCTAssertEqual(
            segmenter.append(
                ASRVoiceActivityFrameDecision(
                    startSeconds: 1.4,
                    endSeconds: 1.7,
                    isSpeech: false,
                    probability: 0.1
                )
            ),
            []
        )

        let endEvents = segmenter.append(
            ASRVoiceActivityFrameDecision(
                startSeconds: 1.7,
                endSeconds: 1.9,
                isSpeech: false,
                probability: 0.1
            )
        )

        guard case .speechEnded(let segment) = endEvents.first else {
            return XCTFail("Expected speechEnded event")
        }
        XCTAssertEqual(segment.startSeconds, 0.82, accuracy: 0.0001)
        XCTAssertEqual(segment.endSeconds, 1.58, accuracy: 0.0001)
        XCTAssertEqual(segment.speechSeconds, 0.4, accuracy: 0.0001)
    }

    func testSegmenterRejectsShortBursts() {
        var segmenter = ASRVoiceActivitySegmenter(configuration: .balanced)
        _ = segmenter.append(
            ASRVoiceActivityFrameDecision(
                startSeconds: 0,
                endSeconds: 0.1,
                isSpeech: true,
                probability: 0.9
            )
        )

        _ = segmenter.append(
            ASRVoiceActivityFrameDecision(
                startSeconds: 0.1,
                endSeconds: 0.4,
                isSpeech: false,
                probability: 0.0
            )
        )

        let events = segmenter.append(
            ASRVoiceActivityFrameDecision(
                startSeconds: 0.4,
                endSeconds: 0.6,
                isSpeech: false,
                probability: 0.0
            )
        )

        guard case .speechRejected(let segment) = events.first else {
            return XCTFail("Expected speechRejected event")
        }
        XCTAssertLessThan(segment.speechSeconds, ASRVoiceActivityConfiguration.balanced.minSpeechSeconds)
        XCTAssertEqual(events.first?.reason, .minSpeechDurationNotMet)
        XCTAssertTrue(events.first?.telemetrySummary.contains("reason=min-speech-duration-not-met") ?? false)
    }

    func testSegmenterForcesLongSegments() {
        let configuration = ASRVoiceActivityConfiguration(
            onsetProbabilityThreshold: 0.5,
            offsetProbabilityThreshold: 0.35,
            minSpeechSeconds: 0.1,
            minSilenceSeconds: 0.5,
            speechPadSeconds: 0,
            maxSegmentSeconds: 1.0
        )
        var segmenter = ASRVoiceActivitySegmenter(configuration: configuration)

        _ = segmenter.append(
            ASRVoiceActivityFrameDecision(
                startSeconds: 0,
                endSeconds: 0.6,
                isSpeech: true,
                probability: 0.9
            )
        )
        let events = segmenter.append(
            ASRVoiceActivityFrameDecision(
                startSeconds: 0.6,
                endSeconds: 1.1,
                isSpeech: true,
                probability: 0.9
            )
        )

        guard case .speechForced(let segment) = events.first else {
            return XCTFail("Expected speechForced event")
        }
        XCTAssertEqual(segment.startSeconds, 0, accuracy: 0.0001)
        XCTAssertEqual(segment.endSeconds, 1.1, accuracy: 0.0001)
        XCTAssertEqual(events.first?.reason, .maxSegmentDurationReached)
        XCTAssertTrue(events.first?.telemetrySummary.contains("reason=max-segment-duration-reached") ?? false)
    }

    func testSegmenterDoesNotEmitSpeechDuringEightHoursOfSilence() {
        var segmenter = ASRVoiceActivitySegmenter(configuration: .realtime)
        let frameDuration = 0.5
        let totalFrames = Int((8 * 60 * 60) / frameDuration)

        for frameIndex in 0..<totalFrames {
            let start = Double(frameIndex) * frameDuration
            let events = segmenter.append(
                ASRVoiceActivityFrameDecision(
                    startSeconds: start,
                    endSeconds: start + frameDuration,
                    isSpeech: false,
                    probability: 0.0
                )
            )
            XCTAssertTrue(events.isEmpty)
        }

        XCTAssertNil(segmenter.finish(at: 8 * 60 * 60))
    }

    func testSegmenterForcesBoundedSegmentsDuringTwoHoursOfContinuousSpeech() {
        let maxSegmentSeconds: TimeInterval = 8
        let frameDuration: TimeInterval = 0.5
        let totalSeconds: TimeInterval = 2 * 60 * 60
        let configuration = ASRVoiceActivityConfiguration(
            onsetProbabilityThreshold: 0.5,
            offsetProbabilityThreshold: 0.35,
            minSpeechSeconds: 0.1,
            minSilenceSeconds: 0.5,
            speechPadSeconds: 0,
            maxSegmentSeconds: maxSegmentSeconds
        )
        var segmenter = ASRVoiceActivitySegmenter(configuration: configuration)
        var forcedSegments: [ASRVoiceActivitySegment] = []
        let totalFrames = Int(totalSeconds / frameDuration) + 1

        for frameIndex in 0..<totalFrames {
            let start = Double(frameIndex) * frameDuration
            let events = segmenter.append(
                ASRVoiceActivityFrameDecision(
                    startSeconds: start,
                    endSeconds: start + frameDuration,
                    isSpeech: true,
                    probability: 0.9
                )
            )
            for event in events {
                switch event {
                case .speechStarted:
                    continue
                case .speechForced(let segment):
                    XCTAssertLessThanOrEqual(segment.durationSeconds, maxSegmentSeconds + frameDuration)
                    forcedSegments.append(segment)
                case .speechEnded, .speechRejected:
                    return XCTFail("Continuous speech should not end or reject before finish, got \(event)")
                }
            }
        }

        let finalEvent = segmenter.finish(at: Double(totalFrames) * frameDuration)
        guard case .speechEnded(let finalSegment) = finalEvent else {
            return XCTFail("Expected final speech segment after continuous speech")
        }

        XCTAssertGreaterThan(forcedSegments.count, 800)
        XCTAssertLessThanOrEqual(finalSegment.durationSeconds, maxSegmentSeconds + frameDuration)
    }

    func testSegmenterRejectsRepeatedShortNoiseBurstsWithoutEmittingSpeechSegments() {
        var segmenter = ASRVoiceActivitySegmenter(configuration: .balanced)
        var rejectedSegments: [ASRVoiceActivitySegment] = []
        let burstCount = 1_200
        let burstSpeechSeconds: TimeInterval = 0.08
        let silenceSeconds: TimeInterval = 0.5
        let cycleSeconds = burstSpeechSeconds + silenceSeconds

        for burstIndex in 0..<burstCount {
            let start = Double(burstIndex) * cycleSeconds
            let speechEvents = segmenter.append(
                ASRVoiceActivityFrameDecision(
                    startSeconds: start,
                    endSeconds: start + burstSpeechSeconds,
                    isSpeech: true,
                    probability: 0.9
                )
            )
            guard case .speechStarted(let paddedStartSeconds) = speechEvents.first else {
                return XCTFail("Expected short burst \(burstIndex) to start a candidate segment, got \(speechEvents)")
            }
            XCTAssertEqual(speechEvents.count, 1)
            XCTAssertEqual(
                paddedStartSeconds,
                max(0, start - ASRVoiceActivityConfiguration.balanced.speechPadSeconds),
                accuracy: 0.0001
            )

            let silenceEvents = segmenter.append(
                ASRVoiceActivityFrameDecision(
                    startSeconds: start + burstSpeechSeconds,
                    endSeconds: start + cycleSeconds,
                    isSpeech: false,
                    probability: 0.0
                )
            )
            guard case .speechRejected(let segment) = silenceEvents.first else {
                return XCTFail("Expected short burst \(burstIndex) to be rejected, got \(silenceEvents)")
            }
            XCTAssertEqual(silenceEvents.count, 1)
            XCTAssertLessThan(segment.speechSeconds, ASRVoiceActivityConfiguration.balanced.minSpeechSeconds)
            XCTAssertEqual(segment.frameCount, 2)
            rejectedSegments.append(segment)
        }

        XCTAssertEqual(rejectedSegments.count, burstCount)
        XCTAssertNil(segmenter.finish(at: Double(burstCount) * cycleSeconds))
    }

    func testSampleFilterKeepsOnlySpeechSegments() {
        let samples = Array(repeating: Float(0), count: 50)
        let decisions = [
            ASRVoiceActivityFrameDecision(startSeconds: 0, endSeconds: 1, isSpeech: false),
            ASRVoiceActivityFrameDecision(startSeconds: 1, endSeconds: 2, isSpeech: true),
            ASRVoiceActivityFrameDecision(startSeconds: 2, endSeconds: 3, isSpeech: false),
            ASRVoiceActivityFrameDecision(startSeconds: 3, endSeconds: 5, isSpeech: false)
        ]
        let result = ASRVoiceActivitySampleFilter.filter(
            samples: samples,
            sampleRate: 10,
            decisions: decisions,
            configuration: ASRVoiceActivityConfiguration(
                onsetProbabilityThreshold: 0.5,
                offsetProbabilityThreshold: 0.35,
                minSpeechSeconds: 0.2,
                minSilenceSeconds: 0.5,
                speechPadSeconds: 0,
                maxSegmentSeconds: nil
            )
        )

        XCTAssertTrue(result.observedFrames)
        XCTAssertTrue(result.observedSpeech)
        XCTAssertEqual(result.speechSegments.count, 1)
        XCTAssertEqual(result.samples.count, 10)
        XCTAssertEqual(result.originalDurationSeconds, 5, accuracy: 0.001)
        XCTAssertEqual(result.filteredDurationSeconds, 1, accuracy: 0.001)
    }

    func testSampleFilterReturnsEmptyWhenNoSpeechObserved() {
        let samples = Array(repeating: Float(0), count: 40)
        let decisions = [
            ASRVoiceActivityFrameDecision(startSeconds: 0, endSeconds: 1, isSpeech: false),
            ASRVoiceActivityFrameDecision(startSeconds: 1, endSeconds: 2, isSpeech: false),
            ASRVoiceActivityFrameDecision(startSeconds: 2, endSeconds: 4, isSpeech: false)
        ]
        let result = ASRVoiceActivitySampleFilter.filter(
            samples: samples,
            sampleRate: 10,
            decisions: decisions,
            configuration: ASRVoiceActivityConfiguration(
                onsetProbabilityThreshold: 0.5,
                offsetProbabilityThreshold: 0.35,
                minSpeechSeconds: 0.2,
                minSilenceSeconds: 0.5,
                speechPadSeconds: 0,
                maxSegmentSeconds: nil
            )
        )

        XCTAssertTrue(result.observedFrames)
        XCTAssertFalse(result.observedSpeech)
        XCTAssertTrue(result.speechSegments.isEmpty)
        XCTAssertTrue(result.samples.isEmpty)
    }
}
