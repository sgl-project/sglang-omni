// GGUFTranslationModelSupport.swift
// Provides GGUF translation model catalog and storage helpers.

import Foundation

enum GGUFTranslationModelID: String, CaseIterable, Identifiable {
    case hyMT2Q4KM = "tencent/Hy-MT2-1.8B-GGUF#Hy-MT2-1.8B-Q4_K_M.gguf"
    case hyMT2Q8_0 = "tencent/Hy-MT2-1.8B-GGUF#Hy-MT2-1.8B-Q8_0.gguf"

    var id: String { rawValue }
}

struct GGUFTranslationModelOption: Equatable, Identifiable {
    let id: GGUFTranslationModelID
    let title: String
    let shortTitle: String
    let filename: String
    let downloadURL: URL
    let sizeText: String
    let sizeBytes: Int64
    let badgeText: String?
    let ratingText: String
    let tags: [String]
    let description: String
}

enum GGUFTranslationModelCatalog {
    static let defaultModelID: GGUFTranslationModelID = .hyMT2Q4KM

    private static let visibleModels: [GGUFTranslationModelOption] = [
        GGUFTranslationModelOption(
            id: .hyMT2Q4KM,
            title: "Hy-MT2 1.8B (Q4_K_M)",
            shortTitle: "Hy-MT2 1.8B Q4_K_M",
            filename: "Hy-MT2-1.8B-Q4_K_M.gguf",
            downloadURL: URL(string: "https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/resolve/main/Hy-MT2-1.8B-Q4_K_M.gguf?download=true")!,
            sizeText: "1.13 GB",
            sizeBytes: 1_130_000_000,
            badgeText: AppLocalization.localizedString("Recommended"),
            ratingText: "4.5",
            tags: [AppLocalization.localizedString("Local"), AppLocalization.localizedString("Fast"), AppLocalization.localizedString("Translation")],
            description: "Tencent Hy-MT2 translation-specialized GGUF model tuned for low-latency multilingual translation on-device.",
        ),
        GGUFTranslationModelOption(
            id: .hyMT2Q8_0,
            title: "Hy-MT2 1.8B (Q8_0)",
            shortTitle: "Hy-MT2 1.8B Q8_0",
            filename: "Hy-MT2-1.8B-Q8_0.gguf",
            downloadURL: URL(string: "https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF/resolve/main/Hy-MT2-1.8B-Q8_0.gguf?download=true")!,
            sizeText: "1.91 GB",
            sizeBytes: 1_910_000_000,
            badgeText: nil,
            ratingText: "4.8",
            tags: [AppLocalization.localizedString("Local"), AppLocalization.localizedString("Accurate"), AppLocalization.localizedString("Translation")],
            description: "Highest-quality Hy-MT2 translation GGUF variant with the largest local footprint.",
        ),
    ]

    static let allModels: [GGUFTranslationModelOption] = visibleModels
    static let availableModels: [GGUFTranslationModelOption] = visibleModels

    static func option(for id: GGUFTranslationModelID) -> GGUFTranslationModelOption {
        allModels.first(where: { $0.id == id }) ?? allModels[0]
    }

    static func displayModels(includingInstalled _: Set<GGUFTranslationModelID>) -> [GGUFTranslationModelOption] {
        visibleModels
    }

    static func resolvedModelID(_ rawValue: String?) -> GGUFTranslationModelID {
        guard let rawValue,
              let modelID = GGUFTranslationModelID(rawValue: rawValue) else {
            return defaultModelID
        }
        return modelID
    }

    static func storageDirectory(root: URL) -> URL {
        root
            .appendingPathComponent("gguf-translation", isDirectory: true)
            .appendingPathComponent("tencent_Hy-MT2-1.8B-GGUF", isDirectory: true)
    }

    static func modelFileURL(for id: GGUFTranslationModelID, root: URL) -> URL {
        storageDirectory(root: root)
            .appendingPathComponent(option(for: id).filename, isDirectory: false)
    }
}
