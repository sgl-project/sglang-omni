import Foundation

enum MeetingRemoteAudioSupport {
    static func buildDoubaoFullRequestPacket(
        reqID: String,
        sequence: Int32,
        hintPayload: ResolvedASRHintPayload,
        audioFormat: String,
        enableNonstream: Bool = false
    ) throws -> Data {
        let payloadObject = DoubaoASRConfiguration.fullRequestPayload(
            requestID: reqID,
            userID: "voxt-meeting",
            language: hintPayload.language,
            chineseOutputVariant: hintPayload.chineseOutputVariant,
            audioFormat: audioFormat,
            enableNonstream: enableNonstream
        )
        let rawPayload = try JSONSerialization.data(withJSONObject: payloadObject)
        let (compression, payload) = try DoubaoPacketCodec.encodePayload(rawPayload)
        return DoubaoPacketCodec.buildPacket(
            messageType: DoubaoProtocol.messageTypeFullClientRequest,
            messageFlags: DoubaoProtocol.flagPositiveSequence,
            serialization: DoubaoProtocol.serializationJSON,
            compression: compression,
            sequence: sequence,
            payload: payload
        )
    }

    static func parseDoubaoServerPacket(_ data: Data) throws -> MeetingLiveProviderPacket? {
        guard let response = try DoubaoPacketCodec.decodeServerPacket(data) else { return nil }
        let payload = response.payload
        let headerSequence = response.sequence

        guard !payload.isEmpty else {
            return MeetingLiveProviderPacket(units: [], fallbackText: nil, isFinal: response.isFinal, sequence: headerSequence)
        }
        guard let object = try? JSONSerialization.jsonObject(with: payload) else {
            let raw = String(data: payload, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let fallbackText = raw.flatMap { RemoteASRTextSanitizer.isLikelyIdentifierText($0) ? nil : $0 }
            return MeetingLiveProviderPacket(units: [], fallbackText: fallbackText, isFinal: response.isFinal, sequence: headerSequence)
        }
        let sequenceFromJSON = DoubaoPacketCodec.sequence(in: object)
        let isFinal = response.isFinal
            || DoubaoPacketCodec.isLastPackage(in: object) == true
            || (sequenceFromJSON ?? 1) < 0
        let fragment = RemoteASRTextSupport.extractDoubaoText(in: object)
        let units = extractDoubaoUtteranceUnits(in: object, defaultIsFinal: isFinal)
        let hasUtteranceContainers = containsRecursiveKey("utterances", in: object)
        return MeetingLiveProviderPacket(
            units: units,
            fallbackText: hasUtteranceContainers ? nil : fragment,
            isFinal: isFinal,
            sequence: sequenceFromJSON ?? headerSequence
        )
    }

    private static func extractDoubaoUtteranceUnits(
        in object: Any,
        defaultIsFinal: Bool
    ) -> [MeetingLiveProviderTranscriptUnit] {
        var units: [MeetingLiveProviderTranscriptUnit] = []

        func walk(_ node: Any) {
            if let dict = node as? [String: Any] {
                if let utterances = dict["utterances"] as? [[String: Any]] {
                    for (index, utterance) in utterances.enumerated() {
                        if let unit = makeDoubaoUtteranceUnit(
                            utterance,
                            fallbackIndex: index,
                            defaultIsFinal: defaultIsFinal
                        ) {
                            units.append(unit)
                        }
                    }
                }
                for value in dict.values {
                    if value is [String: Any] || value is [Any] {
                        walk(value)
                    }
                }
                return
            }
            if let array = node as? [Any] {
                for item in array {
                    walk(item)
                }
            }
        }

        walk(object)
        units.sort { lhs, rhs in
            let lhsStart = lhs.startSeconds ?? 0
            let rhsStart = rhs.startSeconds ?? 0
            if lhsStart == rhsStart {
                return (lhs.key ?? lhs.text) < (rhs.key ?? rhs.text)
            }
            return lhsStart < rhsStart
        }
        return units
    }

    private static func containsRecursiveKey(_ targetKey: String, in object: Any) -> Bool {
        if let dict = object as? [String: Any] {
            if dict[targetKey] != nil {
                return true
            }
            for value in dict.values {
                if containsRecursiveKey(targetKey, in: value) {
                    return true
                }
            }
        }
        if let array = object as? [Any] {
            for item in array where containsRecursiveKey(targetKey, in: item) {
                return true
            }
        }
        return false
    }

    private static func makeDoubaoUtteranceUnit(
        _ utterance: [String: Any],
        fallbackIndex: Int,
        defaultIsFinal: Bool
    ) -> MeetingLiveProviderTranscriptUnit? {
        let text = RemoteASRTextSupport.extractDoubaoText(in: utterance)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !text.isEmpty, !RemoteASRTextSanitizer.isLikelyIdentifierText(text) else { return nil }

        let startMs = extractTimeMilliseconds(
            in: utterance,
            keys: ["start_time", "begin_time", "start_ms", "begin_ms", "start", "begin"]
        )
        let endMs = extractTimeMilliseconds(
            in: utterance,
            keys: ["end_time", "end_ms", "end"]
        )
        let key = extractString(
            in: utterance,
            keys: ["utterance_id", "id", "uid", "segment_id"]
        ) ?? {
            if let startMs, let endMs {
                return "\(startMs)-\(endMs)"
            }
            return "utterance-\(fallbackIndex)-\(text)"
        }()

        let isFinal = extractBool(
            in: utterance,
            keys: ["is_final", "final", "sentence_end", "definite"]
        ) ?? defaultIsFinal

        return MeetingLiveProviderTranscriptUnit(
            key: key,
            startSeconds: startMs.map { Double($0) / 1000 },
            endSeconds: endMs.map { Double($0) / 1000 },
            text: text,
            isFinal: isFinal
        )
    }

    static func makeAliyunSentenceUnit(
        sentence: [String: Any],
        fallbackText: String,
        isFinal: Bool
    ) -> MeetingLiveProviderTranscriptUnit? {
        let text = fallbackText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        let key = extractString(in: sentence, keys: ["sentence_id", "id", "index"])
        let startMs = extractTimeMilliseconds(in: sentence, keys: ["begin_time", "start_time", "start_ms", "begin_ms"])
        let endMs = extractTimeMilliseconds(in: sentence, keys: ["end_time", "end_ms"])
        guard key != nil || startMs != nil || endMs != nil else { return nil }
        return MeetingLiveProviderTranscriptUnit(
            key: key ?? [startMs, endMs].compactMap { $0 }.map(String.init).joined(separator: "-"),
            startSeconds: startMs.map { Double($0) / 1000 },
            endSeconds: endMs.map { Double($0) / 1000 },
            text: text,
            isFinal: isFinal
        )
    }

    static func makeAliyunQwenUnit(
        object: [String: Any],
        fallbackText: String,
        isFinal: Bool
    ) -> MeetingLiveProviderTranscriptUnit? {
        let text = fallbackText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }
        let key = extractString(in: object, keys: ["item_id", "itemId", "id"])
            ?? ((object["item"] as? [String: Any]).flatMap { extractString(in: $0, keys: ["id", "item_id"]) })
        let startMs = extractTimeMilliseconds(in: object, keys: ["audio_start_ms", "start_ms", "begin_ms"])
        let endMs = extractTimeMilliseconds(in: object, keys: ["audio_end_ms", "end_ms"])
        guard key != nil || startMs != nil || endMs != nil else { return nil }
        return MeetingLiveProviderTranscriptUnit(
            key: key ?? [startMs, endMs].compactMap { $0 }.map(String.init).joined(separator: "-"),
            startSeconds: startMs.map { Double($0) / 1000 },
            endSeconds: endMs.map { Double($0) / 1000 },
            text: text,
            isFinal: isFinal
        )
    }

    private static func extractTimeMilliseconds(
        in object: Any,
        keys: [String]
    ) -> Int? {
        if let dict = object as? [String: Any] {
            for key in keys {
                if let value = dict[key], let parsed = extractInt(in: value) {
                    return parsed
                }
            }
        }
        return nil
    }

    private static func extractString(
        in object: Any,
        keys: [String]
    ) -> String? {
        guard let dict = object as? [String: Any] else { return nil }
        for key in keys {
            if let value = dict[key] as? String {
                let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    return trimmed
                }
            }
            if let value = dict[key] {
                if let intValue = extractInt(in: value) {
                    return String(intValue)
                }
            }
        }
        return nil
    }

    private static func extractBool(
        in object: Any,
        keys: [String]
    ) -> Bool? {
        guard let dict = object as? [String: Any] else { return nil }
        for key in keys {
            if let value = dict[key] as? Bool {
                return value
            }
            if let value = dict[key] as? NSNumber {
                return value.boolValue
            }
        }
        return nil
    }

    private static func extractInt(in object: Any) -> Int? {
        if let value = object as? Int {
            return value
        }
        if let value = object as? Int64 {
            return Int(value)
        }
        if let value = object as? Int32 {
            return Int(value)
        }
        if let value = object as? Double {
            return Int(value.rounded())
        }
        if let value = object as? NSNumber {
            return value.intValue
        }
        if let value = object as? String,
           let parsed = Int(value.trimmingCharacters(in: .whitespacesAndNewlines)) {
            return parsed
        }
        return nil
    }

}
