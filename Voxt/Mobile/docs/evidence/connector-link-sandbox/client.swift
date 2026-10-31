// Runs Voxt Connector's real ConnectorLink once and prints Voxt's answer.
import Foundation
setbuf(stdout, nil)
print("client team: \(PeerSignature.ownTeamIdentifier() ?? "none")")
print("client candidates: \(ConnectorLink.socketPaths())")
let done = DispatchSemaphore(value: 0)
Task {
    do {
        let grant = try await ConnectorLink(replyTimeout: 10).authorize()
        print("client approved port=\(grant.port) token=\(grant.token) alwaysAllowed=\(grant.alwaysAllowed)")
    } catch {
        print("client failed: \(error)")
    }
    done.signal()
}
done.wait()
