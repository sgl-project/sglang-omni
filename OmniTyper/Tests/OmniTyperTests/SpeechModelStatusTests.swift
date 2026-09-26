// SPDX-License-Identifier: Apache-2.0
import Foundation
import Testing
@testable import OmniTyper

struct SpeechModelStatusTests {
    @Test func cacheLocationFollowsTheHubEnvironment() {
        let home = URL(fileURLWithPath: "/Users/tester")
        #expect(SpeechModelState.hubCache(environment: [:], home: home).path == "/Users/tester/.cache/huggingface/hub")
        #expect(SpeechModelState.hubCache(environment: ["XDG_CACHE_HOME": "/xdg"], home: home).path == "/xdg/huggingface/hub")
        #expect(SpeechModelState.hubCache(environment: ["XDG_CACHE_HOME": "/xdg", "HF_HOME": "/hf"], home: home).path == "/hf/hub")
        #expect(SpeechModelState.hubCache(environment: ["HF_HOME": "/hf", "HF_HUB_CACHE": "/cache"], home: home).path == "/cache")
    }

    @Test func onlyCompleteWeightsCountAsDownloaded() throws {
        let home = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: home) }
        let repository = "owner/speech"
        func state(ready: Bool = false) -> SpeechModelState {
            SpeechModelState.resolve(ready: ready, repository: repository, environment: [:], home: home)
        }
        #expect(state() == .missing)

        let model = home.appendingPathComponent(".cache/huggingface/hub/models--owner--speech")
        let snapshot = model.appendingPathComponent("snapshots/revision")
        let blobs = model.appendingPathComponent("blobs")
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: snapshot.appendingPathComponent("model.safetensors"),
                                                   withDestinationURL: blobs.appendingPathComponent("weights"))
        #expect(state() == .missing, "A link to weights that never finished downloading is not a model")

        try Data("weights".utf8).write(to: blobs.appendingPathComponent("weights"))
        #expect(state() == .downloaded)
        #expect(state(ready: true) == .ready)
    }
}
