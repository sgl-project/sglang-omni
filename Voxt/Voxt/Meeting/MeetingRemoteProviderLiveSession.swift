import Foundation

@MainActor
struct MeetingRemoteLiveSessionFactory: MeetingLiveSessionFactory {
    let provider: RemoteASRProvider
    let configuration: RemoteProviderConfiguration
    let hintPayload: ResolvedASRHintPayload

    func makeSession(
        for speaker: MeetingSpeaker,
        timelineOffsetSeconds: TimeInterval
    ) throws -> any MeetingLiveTranscribingSession {
        let policy = MeetingLiveSessionPolicy.resolved(
            provider: provider,
            configuration: configuration
        )
        switch provider {
        case .doubaoASR:
            return DoubaoMeetingRemoteLiveSession(
                speaker: speaker,
                configuration: configuration,
                hintPayload: hintPayload,
                timelineOffsetSeconds: timelineOffsetSeconds,
                policy: policy
            )
        case .aliyunBailianASR:
            if RemoteASREndpointSupport.isAliyunQwenRealtimeModel(configuration.model) {
                return AliyunQwenMeetingRemoteLiveSession(
                    speaker: speaker,
                    configuration: configuration,
                    hintPayload: hintPayload,
                    timelineOffsetSeconds: timelineOffsetSeconds,
                    policy: policy
                )
            }
            return AliyunFunMeetingRemoteLiveSession(
                speaker: speaker,
                configuration: configuration,
                hintPayload: hintPayload,
                timelineOffsetSeconds: timelineOffsetSeconds,
                policy: policy
            )
        case .openAIWhisper, .glmASR, .stepFunASR, .xiaomiMiMoASR, .googleGeminiASR:
            throw NSError(
                domain: "Voxt.Meeting",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "This remote provider does not support live meeting sessions."]
            )
        }
    }
}
