import Foundation
import Testing
@testable import VoxtMobileKit

/// Uses only temporary local stores and self-owned remote UUIDs. The launcher enables
/// the isolated Mac's history-sync preference and provides an ephemeral token.
@Suite(.serialized, .enabled(if: ProcessInfo.processInfo.environment["VOXT_E2E_HISTORY"] == "1"))
struct LiveHistorySyncTests {
    @Test @MainActor func twoPhoneStoresMergeRetryAndPropagateDeletions() async throws {
        let environment = ProcessInfo.processInfo.environment
        let url = try ServerConfig.normalizeBaseURL(try #require(environment["VOXT_E2E_URL"]))
        let client = VoxtClient(server: .init(baseURL: url, token: try #require(environment["VOXT_E2E_TOKEN"])))
        let firstName = "voxt-live-history-a.\(UUID())", secondName = "voxt-live-history-b.\(UUID())"
        let firstDefaults = UserDefaults(suiteName: firstName)!, secondDefaults = UserDefaults(suiteName: secondName)!
        defer {
            firstDefaults.removePersistentDomain(forName: firstName)
            secondDefaults.removePersistentDomain(forName: secondName)
        }
        let firstHistory = TranscriptHistory(defaults: firstDefaults), secondHistory = TranscriptHistory(defaults: secondDefaults)
        let firstStore = HistorySyncStore(defaults: firstDefaults), secondStore = HistorySyncStore(defaults: secondDefaults)
        let firstID = UUID(), secondID = UUID(), owned = [firstID, secondID]
        let stamp = Date().addingTimeInterval(-3600)
        let firstText = "Voxt sync verification A \(firstID)", secondText = "Voxt sync verification B \(secondID)"
        let initial = try await client.syncHistory(.init())
        do {
            firstHistory.append(id: firstID, text: firstText, mode: .translate, elapsedSeconds: 7, now: stamp)
            secondHistory.append(id: secondID, text: secondText, mode: .edit, elapsedSeconds: 11, now: stamp)
            firstStore.setEnabled(true)
            secondStore.setEnabled(true)
            let firstCoordinator = HistorySyncCoordinator(store: firstStore) { try await client.syncHistory($0) }
            let secondCoordinator = HistorySyncCoordinator(store: secondStore) { try await client.syncHistory($0) }
            await firstCoordinator.sync()
            #expect(firstCoordinator.lastError == nil)
            await secondCoordinator.sync()
            #expect(secondCoordinator.lastError == nil)
            await firstCoordinator.sync()
            let firstCopy = try #require(firstHistory.items.first { $0.id == secondID })
            let secondCopy = try #require(secondHistory.items.first { $0.id == firstID })
            #expect(firstCopy.text == secondText && firstCopy.mode == .edit && firstCopy.elapsedSeconds == 11)
            #expect(secondCopy.text == firstText && secondCopy.mode == .translate && secondCopy.elapsedSeconds == 7)
            #expect(abs(secondCopy.date.timeIntervalSince(stamp)) < 0.001)

            // Fail before the request reaches the real server, then recreate the store.
            firstHistory.delete(ids: [firstID])
            let offline = HistorySyncCoordinator(store: firstStore) { _ in throw URLError(.notConnectedToInternet) }
            await offline.sync()
            #expect(offline.lastError != nil)
            #expect(firstStore.makeRequest()?.deletedIDs.contains(firstID) == true)
            firstStore.setEnabled(false)
            let restarted = HistorySyncStore(defaults: firstDefaults)
            restarted.setEnabled(true)
            let recovered = HistorySyncCoordinator(store: restarted) { try await client.syncHistory($0) }
            await recovered.sync()
            #expect(recovered.lastError == nil)
            await secondCoordinator.sync()
            #expect(!firstHistory.items.contains { $0.id == firstID })
            #expect(!secondHistory.items.contains { $0.id == firstID })
            #expect(secondHistory.items.contains { $0.id == secondID })

            // A stale offline upload must not resurrect the deleted record.
            _ = try await client.syncHistory(.init(storeID: initial.storeID, items: [
                .init(id: firstID, text: firstText, createdAt: stamp.timeIntervalSince1970)
            ]))
            secondStore.setEnabled(false)
            secondStore.setEnabled(true)
            await secondCoordinator.sync()
            #expect(!secondHistory.items.contains { $0.id == firstID })
            #expect(secondHistory.items.filter { $0.id == secondID }.count == 1)
        } catch {
            _ = try? await client.syncHistory(.init(storeID: initial.storeID, deletedIDs: owned))
            throw error
        }
        _ = try await client.syncHistory(.init(storeID: initial.storeID, deletedIDs: owned))
    }
}
