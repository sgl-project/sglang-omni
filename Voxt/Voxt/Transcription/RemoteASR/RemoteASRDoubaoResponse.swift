import Foundation

extension RemoteASRTranscriber {
    static func parseDoubaoServerPacket(_ data: Data) throws -> (text: String?, isFinal: Bool)? {
        guard let response = try DoubaoPacketCodec.decodeServerPacket(data) else { return nil }
        let payload = response.payload
        guard !payload.isEmpty else { return (nil, response.isFinal) }

        guard let object = try? JSONSerialization.jsonObject(with: payload) else {
            let raw = String(data: payload, encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let visibleText = raw.flatMap { RemoteASRTextSanitizer.isLikelyIdentifierText($0) ? nil : $0 }
            return (visibleText, response.isFinal)
        }

        let sequenceFromJSON = DoubaoPacketCodec.sequence(in: object)
        let isFinal = response.isFinal
            || DoubaoPacketCodec.isLastPackage(in: object) == true
            || (sequenceFromJSON ?? 1) < 0
        let fragment = RemoteASRTextSupport.extractDoubaoText(in: object)
        return (fragment, isFinal)
    }
}
