// SPDX-License-Identifier: Apache-2.0
import Foundation

extension AppModel {
    func setKeepModelLoaded(_ enabled: Bool) {
        guard phase == .idle else { return }
        store.preferences.keepModelLoaded = enabled
        if enabled { prepareModels() }
        else { stopModelWorker(); notice = L("notice.modelUnloaded") }
    }

    func prepareModels() {
        guard phase == .idle, preloadTask == nil else { return }
        error = ""; notice = ""
        let preferences = store.preferences
        preloadTask = Task {
            defer {
                if !Task.isCancelled { preloadTask = nil }
            }
            do {
                let response = try await worker.request(["op": "prepare", "asr_model": preferences.asrModel],
                                                        python: preferences.pythonExecutable)
                if !Task.isCancelled && phase == .idle { notice = L("notice.modelReady") }
                return response
            } catch {
                if !Task.isCancelled && phase == .idle { self.error = error.localizedDescription }
                throw error
            }
        }
    }

    func stopModelWorker() {
        preloadTask?.cancel(); preloadTask = nil
        worker.stop()
    }

    func releaseIdleModel() {
        if store.preferences.keepModelLoaded != true { stopModelWorker() }
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
