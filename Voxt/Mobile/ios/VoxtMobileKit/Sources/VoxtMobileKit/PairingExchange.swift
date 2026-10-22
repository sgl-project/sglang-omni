import Foundation

/// Exchanges the QR capability for authorization of this phone's durable key.
public enum PairingExchange {
    public enum Failure: Error { case rejected(Int), invalidResponse }
    private final class NoRedirect: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
        func urlSession(_ session: URLSession, task: URLSessionTask,
                        willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest request: URLRequest,
                        completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
            completionHandler(nil)
        }
    }
    public static func exchange(endpoint: URL, proof: String, publicKey: String, privateKey: String,
                                session injected: URLSession? = nil) async throws -> ServerConfig {
        let session = injected ?? URLSession(configuration: .ephemeral, delegate: NoRedirect(), delegateQueue: nil)
        defer { if injected == nil { session.invalidateAndCancel() } }
        var request = URLRequest(url: endpoint.appendingPathComponent("pair"))
        request.httpMethod = "POST"
        request.timeoutInterval = 30
        request.setValue("Bearer " + proof, forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(["phoneKey": publicKey])
        let (data, response) = try await session.data(for: request)
        guard let response = response as? HTTPURLResponse else { throw Failure.invalidResponse }
        guard response.statusCode == 200 else { throw Failure.rejected(response.statusCode) }
        guard data.count <= 8192 else { throw Failure.invalidResponse }
        struct Reply: Decodable { let invitation: URL }
        let reply = try JSONDecoder().decode(Reply.self, from: data)
        return try TailcatInvitation(pairingLink: reply.invitation).server(clientKey: privateKey, publicKey: publicKey)
    }
}
