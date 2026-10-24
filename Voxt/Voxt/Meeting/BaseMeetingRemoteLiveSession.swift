import Foundation

@MainActor
class BaseMeetingRemoteLiveSession: MeetingLiveTranscribingSession {
    let speaker: MeetingSpeaker
    let configuration: RemoteProviderConfiguration
    let hintPayload: ResolvedASRHintPayload
    let speechThreshold: Float
    let policy: MeetingLiveSessionPolicy

    private(set) var eventHandler: ((MeetingTranscriptEvent) -> Void)?
    private(set) var managedSocket: VoxtNetworkSession.ManagedWebSocketTask?
    private(set) var receiveTask: Task<Void, Never>?
    private(set) var isCancelled = false
    private(set) var isStopping = false
    private(set) var isReadyForAudio = false
    private(set) var pendingAudioPackets: [Data] = []
    private(set) var state: MeetingLiveSessionState = .connecting

    private var finishContinuations: [CheckedContinuation<Void, Never>] = []
    private var finishTimeoutTask: Task<Void, Never>?
    private let finishTimeout: @MainActor @Sendable () async throws -> Void
    private var hasStarted = false
    private var hasFinished = false
    private var hasSentFinishSignal = false
    private var isFlushingPendingAudio = false
    private(set) var currentSegmentID: UUID?
    private(set) var currentSegmentStartSeconds: TimeInterval?
    private(set) var currentTranscriptText: String?
    private(set) var totalAudioSecondsSent: TimeInterval = 0
    private var lastSpeechAudioEndSeconds: TimeInterval?
    private var lastTranscriptEventAt: Date?
    private var transcriptState = MeetingLiveTranscriptState()
    private var lastFinalizedSegmentEndSeconds: TimeInterval?
    private let timelineOffsetSeconds: TimeInterval
    private var hasLoggedFirstAudioPacket = false
    private var hasLoggedFirstServerPacket = false
    private var keepaliveTask: Task<Void, Never>?
    private var lastSpeechAt = Date()
    private var lastKeepaliveAt: Date?
    private var shouldLogNextSpeechAudioPacket = false
    private var hasBegunSpeechStreaming = false
    private var sentAudioPacketCount = 0

    init(
        speaker: MeetingSpeaker,
        configuration: RemoteProviderConfiguration,
        hintPayload: ResolvedASRHintPayload,
        speechThreshold: Float,
        timelineOffsetSeconds: TimeInterval,
        policy: MeetingLiveSessionPolicy,
        finishTimeout: @escaping @MainActor @Sendable () async throws -> Void = {
            try await Task.sleep(for: .milliseconds(1_800))
        }
    ) {
        self.speaker = speaker
        self.configuration = configuration
        self.hintPayload = hintPayload
        self.speechThreshold = speechThreshold
        self.timelineOffsetSeconds = timelineOffsetSeconds
        self.policy = policy
        self.finishTimeout = finishTimeout
    }

    func start(
        timelineOffsetSeconds _: TimeInterval,
        eventHandler: @escaping @MainActor (MeetingTranscriptEvent) -> Void
    ) async throws {
        guard !hasStarted, !hasFinished else {
            throw NSError(domain: "Voxt.Meeting", code: -70, userInfo: [NSLocalizedDescriptionKey: "A live meeting session cannot be restarted."])
        }
        hasStarted = true
        self.eventHandler = eventHandler
        state = .connecting
        lastSpeechAt = Date()
        totalAudioSecondsSent = 0
        hasBegunSpeechStreaming = false
        transcriptState.resetCurrentItem()
        do {
            try await openTransport()
            try Task.checkCancellation()
            guard !isCancelled else { throw CancellationError() }
            if !hasFinished { startKeepaliveLoopIfNeeded() }
        } catch {
            await cancel()
            throw error
        }
    }

    func append(samples: [Float], sampleRate: Double) async {
        guard !Task.isCancelled, !isCancelled, !isStopping, !hasFinished,
              !samples.isEmpty, sampleRate.isFinite, sampleRate > 0 else { return }

        let normalizedLevel = AudioLevelMeter.normalizedLevel(fromSamples: samples)
        let startSeconds = totalAudioSecondsSent
        let duration = Double(samples.count) / max(sampleRate, 1)
        let isSpeech = normalizedLevel >= speechThreshold
        hasBegunSpeechStreaming = true
        if isSpeech {
            lastSpeechAt = Date()
            lastSpeechAudioEndSeconds = startSeconds + duration
            markSpeechIfNeeded(suggestedStartSeconds: startSeconds)
            shouldLogNextSpeechAudioPacket = true
            hasBegunSpeechStreaming = true
        } else if shouldSplitCurrentSegmentForSilence(at: startSeconds) {
            emitPendingFinalSegmentIfNeeded(endSeconds: timelineOffsetSeconds + startSeconds)
        }
        totalAudioSecondsSent += duration

        guard let pcmData = RemoteASRTranscriber.makePCM16MonoData(from: samples, inputSampleRate: sampleRate) else {
            return
        }
        pendingAudioPackets.append(pcmData)
        await flushPendingAudioIfNeeded()
    }

    func finish() async {
        await withTaskCancellationHandler {
            guard !Task.isCancelled else { await cancel(); return }
            guard !isCancelled, !hasFinished else { return }
            if !isStopping {
                isStopping = true
                state = .stopping
                stopKeepaliveLoop()
                let wait = finishTimeout
                // Start the deadline before sending: send/handshake can also stall.
                finishTimeoutTask = Task { @MainActor [weak self] in
                    do {
                        try await wait()
                        try Task.checkCancellation()
                    } catch { return }
                    self?.signalFinished()
                }
            }
            await flushPendingAudioIfNeeded()
            await sendFinishIfReady()
            await waitForFinish()
        } onCancel: {
            Task { @MainActor [weak self] in await self?.cancel() }
        }
    }

    func cancel() async {
        guard !isCancelled, !hasFinished else { return }
        isCancelled = true
        state = .failed
        complete()
    }

    func openTransport() async throws {
        fatalError("Subclasses must override openTransport()")
    }

    func sendAudioPacket(_ pcmData: Data, isLast: Bool) async {
        fatalError("Subclasses must override sendAudioPacket(_:isLast:)")
    }

    func sendFinishSignal() async {
        fatalError("Subclasses must override sendFinishSignal()")
    }

    func flushPendingAudioIfNeeded() async {
        guard isReadyForAudio, !hasFinished, !isFlushingPendingAudio else { return }
        isFlushingPendingAudio = true
        // Appends during an await join the next batch rather than overtaking it.
        while !pendingAudioPackets.isEmpty, !hasFinished {
            let packets = pendingAudioPackets
            pendingAudioPackets.removeAll()
            for packet in packets {
                guard !hasFinished else { break }
                await sendAudioPacket(packet, isLast: false)
            }
        }
        isFlushingPendingAudio = false
        await sendFinishIfReady()
    }

    private func sendFinishIfReady() async {
        guard isStopping, isReadyForAudio, !isFlushingPendingAudio,
              !hasFinished, !hasSentFinishSignal else { return }
        hasSentFinishSignal = true
        await sendFinishSignal()
    }

    func handleReadyForAudio() async {
        guard !isReadyForAudio, !isCancelled, !hasFinished else { return }
        isReadyForAudio = true
        if !isStopping { state = .active }
        await flushPendingAudioIfNeeded()
    }

    func primeTransportForAudio() async {
        guard !isCancelled, !isStopping else { return }
        do {
            try await Task.sleep(for: .milliseconds(180))
        } catch {
            return
        }
        await handleReadyForAudio()
    }

    func emitTranscript(text: String, isFinal: Bool) {
        guard !hasFinished else { return }
        if shouldSplitCurrentSegmentForTextOutputGap() {
            emitPendingFinalSegmentIfNeeded()
        }
        let normalizedText = transcriptState.normalizedVisibleText(for: text)
        guard !normalizedText.isEmpty else {
            if isFinal {
                transcriptState.resetCurrentItem()
                resetCurrentSegment()
            }
            return
        }

        if currentSegmentID == nil {
            currentSegmentID = UUID()
            currentSegmentStartSeconds = max(timelineOffsetSeconds + totalAudioSecondsSent - 0.4, 0)
        }
        currentTranscriptText = normalizedText

        let segment = MeetingTranscriptSegment(
            id: currentSegmentID ?? UUID(),
            speaker: speaker,
            startSeconds: currentSegmentStartSeconds ?? max(timelineOffsetSeconds + totalAudioSecondsSent - 0.4, 0),
            endSeconds: max(timelineOffsetSeconds + totalAudioSecondsSent, currentSegmentStartSeconds ?? 0),
            text: normalizedText,
            preventsAdjacentMerge: true
        )
        eventHandler?(isFinal ? .final(segment) : .partial(segment))
        lastTranscriptEventAt = Date()

        if isFinal {
            resetCurrentSegment()
        }
    }

    func emitProviderPacket(_ packet: MeetingLiveProviderPacket) {
        guard !hasFinished else { return }
        if !packet.units.isEmpty {
            if let activeUnit = latestDisplayableUnit(from: packet.units) {
                emitProviderUnit(activeUnit, forceFinal: packet.isFinal)
            }
            if packet.isFinal {
                signalFinished()
            }
            return
        }

        if let fallbackText = packet.fallbackText,
           !fallbackText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            emitTranscript(text: fallbackText, isFinal: packet.isFinal)
            if packet.isFinal { signalFinished() }
            return
        }

        if packet.isFinal {
            signalFinished()
        }
    }

    private func emitProviderUnit(_ unit: MeetingLiveProviderTranscriptUnit, forceFinal: Bool) {
        let text = unit.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        if shouldSplitCurrentSegmentForTextOutputGap() {
            emitPendingFinalSegmentIfNeeded()
        }

        let normalizedText = transcriptState.normalizedVisibleText(for: text)
        guard !normalizedText.isEmpty else {
            if forceFinal {
                transcriptState.resetCurrentItem()
                resetCurrentSegment()
            }
            return
        }

        if currentSegmentID == nil {
            currentSegmentID = UUID()
            currentSegmentStartSeconds = resolvedProviderSegmentStartSeconds(for: unit)
        }
        currentTranscriptText = normalizedText

        let segment = MeetingTranscriptSegment(
            id: currentSegmentID ?? UUID(),
            speaker: speaker,
            startSeconds: currentSegmentStartSeconds ?? resolvedProviderSegmentStartSeconds(for: unit),
            endSeconds: resolvedProviderSegmentEndSeconds(
                for: unit,
                startSeconds: currentSegmentStartSeconds ?? resolvedProviderSegmentStartSeconds(for: unit)
            ),
            text: normalizedText,
            preventsAdjacentMerge: true
        )
        let shouldFinalizeNow = forceFinal || unit.isFinal
        eventHandler?(shouldFinalizeNow ? .final(segment) : .partial(segment))
        lastTranscriptEventAt = Date()

        if shouldFinalizeNow {
            resetCurrentSegment()
        }
    }

    func emitFailure(_ error: Error) {
        guard !hasFinished, !isCancelled else { return }
        guard !isStopping else { signalFinished(); return }
        state = .failed
        let nsError = error as NSError
        let message = (
            VoxtNetworkSession.directModeConflictMessage(for: error)
            ?? nsError.localizedDescription
        ).trimmingCharacters(in: .whitespacesAndNewlines)
        complete(failureMessage: message.isEmpty ? "Unknown live meeting ASR error." : message)
    }

    func finishReceiveLoop(_ task: Task<Void, Never>) {
        receiveTask = task
    }

    func registerSocket(_ socket: VoxtNetworkSession.ManagedWebSocketTask) {
        managedSocket = socket
    }

    func socketTask() -> URLSessionWebSocketTask? {
        managedSocket?.task
    }

    func cancelTransport(closeCode: URLSessionWebSocketTask.CloseCode) {
        stopKeepaliveLoop()
        receiveTask?.cancel()
        receiveTask = nil
        if let task = managedSocket?.task {
            task.cancel(with: closeCode, reason: nil)
        }
        managedSocket?.session.invalidateAndCancel()
        managedSocket = nil
    }

    private func markSpeechIfNeeded(suggestedStartSeconds: TimeInterval) {
        guard currentSegmentID == nil else { return }
        currentSegmentID = UUID()
        currentSegmentStartSeconds = timelineOffsetSeconds + suggestedStartSeconds
    }

    private func resetCurrentSegment() {
        currentSegmentID = nil
        currentSegmentStartSeconds = nil
        currentTranscriptText = nil
        lastTranscriptEventAt = nil
        transcriptState.resetCurrentItem()
    }

    private func waitForFinish() async {
        // The provider may finish while sendFinishSignal is suspended, before any waiter exists.
        guard !hasFinished else { return }
        await withCheckedContinuation { finishContinuations.append($0) }
    }

    func signalFinished() {
        complete()
    }

    private func complete(failureMessage: String? = nil) {
        guard !hasFinished else { return }
        hasFinished = true
        if state != .failed { state = .stopping }
        stopKeepaliveLoop()
        finishTimeoutTask?.cancel()
        finishTimeoutTask = nil
        if isCancelled {
            resetCurrentSegment()
        } else {
            // Persist a recoverable partial before .failed removes the coordinator's session token.
            emitPendingFinalSegmentIfNeeded()
        }
        if let failureMessage { eventHandler?(.failed(speaker: speaker, message: failureMessage)) }
        pendingAudioPackets.removeAll()
        isReadyForAudio = false
        cancelTransport(closeCode: state == .failed ? .goingAway : .normalClosure)
        eventHandler?(.finished(speaker: speaker))
        eventHandler = nil
        let waiters = finishContinuations
        finishContinuations.removeAll()
        waiters.forEach { $0.resume() }
    }

    private func emitPendingFinalSegmentIfNeeded(endSeconds explicitEndSeconds: TimeInterval? = nil) {
        guard let currentSegmentID,
              let currentSegmentStartSeconds,
              let text = currentTranscriptText?.trimmingCharacters(in: .whitespacesAndNewlines),
              !text.isEmpty
        else {
            return
        }

        let segment = MeetingTranscriptSegment(
            id: currentSegmentID,
            speaker: speaker,
            startSeconds: currentSegmentStartSeconds,
            endSeconds: max(
                explicitEndSeconds ?? timelineOffsetSeconds + totalAudioSecondsSent,
                currentSegmentStartSeconds
            ),
            text: text,
            preventsAdjacentMerge: true
        )
        lastFinalizedSegmentEndSeconds = segment.endSeconds ?? currentSegmentStartSeconds
        transcriptState.freezeCurrentItem(text: text)
        eventHandler?(.final(segment))
        resetCurrentSegment()
    }

    private func latestDisplayableUnit(
        from units: [MeetingLiveProviderTranscriptUnit]
    ) -> MeetingLiveProviderTranscriptUnit? {
        guard currentTranscriptText?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false else {
            return units.last
        }
        if let lastUnit = units.last,
           shouldSuppressAsStaleLeadingUnit(lastUnit) {
            let unitStart = resolvedProviderSegmentStartSeconds(for: lastUnit)
            let unitEnd = resolvedProviderSegmentEndSeconds(for: lastUnit, startSeconds: unitStart)
            let finalizedEndDescription = lastFinalizedSegmentEndSeconds.map { String($0) } ?? "nil"
            VoxtLog.meeting(
                "Meeting live stale leading unit suppressed. speaker=\(speaker.rawValue), unitStart=\(unitStart), unitEnd=\(unitEnd), lastFinalizedEnd=\(finalizedEndDescription)",
                verbose: true
            )
        }
        return units.last(where: { !shouldSuppressAsStaleLeadingUnit($0) })
    }

    private func shouldSuppressAsStaleLeadingUnit(
        _ unit: MeetingLiveProviderTranscriptUnit
    ) -> Bool {
        guard let lastFinalizedSegmentEndSeconds else { return false }
        let unitStart = resolvedProviderSegmentStartSeconds(for: unit)
        let unitEnd = resolvedProviderSegmentEndSeconds(for: unit, startSeconds: unitStart)
        return unitEnd <= lastFinalizedSegmentEndSeconds + 0.05
    }

    private func shouldSplitCurrentSegmentForSilence(at currentAudioSeconds: TimeInterval) -> Bool {
        guard policy.segmentSilenceSplitThreshold > 0,
              currentSegmentID != nil,
              currentTranscriptText?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false,
              let lastSpeechAudioEndSeconds
        else {
            return false
        }
        return currentAudioSeconds - lastSpeechAudioEndSeconds >= policy.segmentSilenceSplitThreshold
    }

    private func shouldSplitCurrentSegmentForTextOutputGap() -> Bool {
        guard policy.segmentSilenceSplitThreshold > 0,
              currentSegmentID != nil,
              currentTranscriptText?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false,
              let lastTranscriptEventAt
        else {
            return false
        }
        return Date().timeIntervalSince(lastTranscriptEventAt) >= policy.segmentSilenceSplitThreshold
    }

    func logFirstAudioPacketIfNeeded(kind: String) {
        guard !hasLoggedFirstAudioPacket else { return }
        hasLoggedFirstAudioPacket = true
        VoxtLog.meeting("Meeting live audio started. provider=\(kind), speaker=\(speaker.rawValue)")
    }

    func logOutgoingAudioPacketIfNeeded(kind: String, sequence: Int32, payloadBytes: Int) {
        guard sentAudioPacketCount < 5 else { return }
        sentAudioPacketCount += 1
        VoxtLog.meeting(
            "Meeting live audio packet sent. provider=\(kind), speaker=\(speaker.rawValue), sequence=\(sequence), payloadBytes=\(payloadBytes)"
        )
    }

    func consumeShouldLogNextSpeechAudioPacket() -> Bool {
        defer { shouldLogNextSpeechAudioPacket = false }
        return shouldLogNextSpeechAudioPacket
    }

    func logServerPacketIfNeeded(kind: String, parsed: (text: String?, isFinal: Bool, sequence: Int32?)?) {
        let textCount = parsed?.text?.trimmingCharacters(in: .whitespacesAndNewlines).count ?? 0
        if !hasLoggedFirstServerPacket {
            hasLoggedFirstServerPacket = true
            VoxtLog.meeting(
                "Meeting live server packet received. provider=\(kind), speaker=\(speaker.rawValue), textChars=\(textCount), isFinal=\(parsed?.isFinal == true), sequence=\(parsed?.sequence.map(String.init) ?? "nil")"
            )
            return
        }
        if textCount > 0 || parsed?.isFinal == true {
            VoxtLog.meeting(
                "Meeting live transcript event. provider=\(kind), speaker=\(speaker.rawValue), textChars=\(textCount), isFinal=\(parsed?.isFinal == true), sequence=\(parsed?.sequence.map(String.init) ?? "nil")",
                verbose: true
            )
        }
    }

    private func startKeepaliveLoopIfNeeded() {
        guard policy.idleKeepaliveEnabled, keepaliveTask == nil else { return }
        keepaliveTask = Task { @MainActor [weak self] in
            guard let self else { return }
            while !Task.isCancelled, !self.isCancelled, !self.isStopping {
                do {
                    try await Task.sleep(for: .milliseconds(800))
                } catch {
                    return
                }
                await self.sendKeepaliveIfNeeded()
            }
        }
    }

    private func stopKeepaliveLoop() {
        keepaliveTask?.cancel()
        keepaliveTask = nil
    }

    private func sendKeepaliveIfNeeded() async {
        guard policy.idleKeepaliveEnabled,
              isReadyForAudio,
              state == .active,
              !isCancelled,
              !isStopping,
              hasBegunSpeechStreaming
        else {
            return
        }

        let now = Date()
        guard now.timeIntervalSince(lastSpeechAt) >= policy.idleKeepaliveInterval else { return }
        if let lastKeepaliveAt,
           now.timeIntervalSince(lastKeepaliveAt) < max(policy.idleKeepaliveInterval - 0.5, 0.5) {
            return
        }

        let sampleCount = max(Int(16_000 * policy.idleKeepaliveFrameDuration), 1)
        let silenceSamples = [Float](repeating: 0, count: sampleCount)
        guard let silenceData = RemoteASRTranscriber.makePCM16MonoData(from: silenceSamples, inputSampleRate: 16_000) else {
            return
        }
        lastKeepaliveAt = now
        await sendAudioPacket(silenceData, isLast: false)
    }

    private func resolvedProviderSegmentStartSeconds(
        for unit: MeetingLiveProviderTranscriptUnit
    ) -> TimeInterval {
        let relativeStart = unit.startSeconds ?? max(totalAudioSecondsSent - 0.4, 0)
        return max(timelineOffsetSeconds + relativeStart, 0)
    }

    private func resolvedProviderSegmentEndSeconds(
        for unit: MeetingLiveProviderTranscriptUnit,
        startSeconds: TimeInterval
    ) -> TimeInterval {
        let relativeEnd = unit.endSeconds ?? totalAudioSecondsSent
        return max(timelineOffsetSeconds + relativeEnd, startSeconds)
    }
}
