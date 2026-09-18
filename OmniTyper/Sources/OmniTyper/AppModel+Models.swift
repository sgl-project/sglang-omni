// SPDX-License-Identifier: Apache-2.0
import Foundation

extension AppModel {
    func setKeepModelLoaded(_ enabled: Bool) {
        guard phase == .idle else { return }
        store.preferences.retainsSpeechModel = enabled
        if enabled { prepareModels() }
        else { releaseIdleModel(); notice = L("notice.modelUnloaded") }
    }

    func prepareModels() {
        guard phase == .idle, preloadTask == nil else { return }
        isPreloading = true
        error = ""; notice = ""
        let preferences = store.preferences
        let token = UUID(); preloadGeneration = token
        preloadTask = Task {
            defer {
                if preloadGeneration == token { preloadTask = nil; isPreloading = false }
            }
            do {
                let response = try await worker.request(["op": "prepare", "asr_model": preferences.asrModel],
                                                        python: preferences.pythonExecutable)
                if preloadGeneration == token && phase == .idle { notice = L("notice.modelReady") }
                return response
            } catch {
                if preloadGeneration == token && phase == .idle { self.error = error.localizedDescription }
                throw error
            }
        }
    }

    func stopModelWorker() {
        preloadGeneration = UUID()
        preloadTask?.cancel(); preloadTask = nil; isPreloading = false
        worker.stop()
    }

    func releaseIdleModel() {
        if phase == .idle && !store.preferences.retainsSpeechModel { stopModelWorker() }
    }

    func releaseModels() {
        guard phase == .idle else { return }
        store.preferences.retainsSpeechModel = false
        stopModelWorker()
        notice = L("notice.modelUnloaded")
    }

    func loadTextModels() {
        guard phase == .idle, !isPreloading else { return }
        error = ""; notice = ""
        do {
            var request = try store.preferences.textSettings.payload(apiKey: textAPIKey, requireModel: false)
            request["op"] = "models"
            let python = store.preferences.pythonExecutable
            phase = .preparing
            let token = UUID(); generation = token
            task = Task {
                do {
                    let response = try await worker.request(request, python: python)
                    guard generation == token else { return }
                    textModels = response["models"] as? [String] ?? []
                    notice = textModels.isEmpty ? L("notice.modelsEmpty") : L("notice.modelsListed")
                } catch {
                    guard generation == token else { return }
                    self.error = error.localizedDescription
                }
                phase = .idle
                releaseIdleModel()
            }
        } catch { self.error = error.localizedDescription }
    }
}
