// ModelLogoView.swift
// Provides Model Logo View for model settings.

import AppKit
import SwiftUI

enum ModelLogoKey: String {
    case apple
    case openAI
    case anthropic
    case google
    case gemini
    case qwen
    case zhipu
    case deepSeek
    case grok
    case cohere
    case moss
    case granite
    case fireRed
    case sense
    case mistral
    case gemma
    case meta
    case nvidia
    case ollama
    case openRouter
    case minimax
    case kimi
    case doubao
    case xiaomi
    case hunyuan
    case volcengine
    case lmStudio
    case alibaba
    case stepFun
    case liquid
    case generic

    var resourceName: String? {
        switch self {
        case .apple:
            return "apple"
        case .openAI:
            return "openai"
        case .anthropic:
            return "claude"
        case .google:
            return "google"
        case .gemini:
            return "gemini"
        case .qwen:
            return "qwen"
        case .zhipu:
            return "chatglm"
        case .deepSeek:
            return "deepseek"
        case .grok:
            return "grok"
        case .cohere:
            return "cohere"
        case .moss:
            return "moss"
        case .granite:
            return "ibm"
        case .fireRed:
            return "firered"
        case .sense:
            return "sensenova"
        case .mistral:
            return "mistral"
        case .gemma:
            return "gemma"
        case .meta:
            return "meta"
        case .nvidia:
            return "nvidia"
        case .ollama:
            return "ollama"
        case .openRouter:
            return "openrouter"
        case .minimax:
            return "minimax"
        case .kimi:
            return "kimi"
        case .doubao:
            return "doubao"
        case .xiaomi:
            return "xiaomi"
        case .hunyuan:
            return "hunyuan"
        case .volcengine:
            return "volcengine"
        case .lmStudio:
            return "lmstudio"
        case .alibaba:
            return "alibaba"
        case .stepFun:
            return "stepfun"
        case .liquid:
            return "liquid"
        case .generic:
            return nil
        }
    }

    var isTemplate: Bool {
        switch self {
        case .apple, .openAI, .grok, .ollama, .openRouter, .lmStudio, .kimi, .liquid:
            return true
        case .anthropic, .google, .gemini, .qwen, .zhipu, .deepSeek, .cohere, .moss, .granite, .fireRed, .sense,
             .mistral, .gemma, .meta, .nvidia, .minimax, .doubao, .xiaomi, .volcengine, .alibaba, .stepFun, .generic:
            return false
        case .hunyuan:
            return false
        }
    }

    var fallbackText: String {
        switch self {
        case .apple:
            return ""
        case .openAI:
            return "AI"
        case .anthropic:
            return "Cl"
        case .google:
            return "G"
        case .gemini:
            return "Ge"
        case .qwen:
            return "Q"
        case .zhipu:
            return "GL"
        case .deepSeek:
            return "DS"
        case .grok:
            return "x"
        case .cohere:
            return "Co"
        case .moss:
            return "M"
        case .granite:
            return "IBM"
        case .fireRed:
            return "红"
        case .sense:
            return "S"
        case .mistral:
            return "M"
        case .gemma:
            return "Ge"
        case .meta:
            return "L"
        case .nvidia:
            return "N"
        case .ollama:
            return "O"
        case .openRouter:
            return "OR"
        case .minimax:
            return "MM"
        case .kimi:
            return "K"
        case .doubao:
            return "豆"
        case .xiaomi:
            return "Mi"
        case .hunyuan:
            return "HY"
        case .volcengine:
            return "火"
        case .lmStudio:
            return "LM"
        case .alibaba:
            return "阿"
        case .stepFun:
            return "阶"
        case .liquid:
            return "L"
        case .generic:
            return "M"
        }
    }

    static func resolve(title: String, engine: String) -> ModelLogoKey {
        let value = "\(title) \(engine)"
            .folding(options: [.diacriticInsensitive, .caseInsensitive], locale: .current)
            .lowercased()

        if value.contains("direct dictation") || value.contains("系统听写") || value.contains("apple") {
            return .apple
        }
        if value.contains("claude") || value.contains("anthropic") {
            return .anthropic
        }
        if value.contains("deepseek") {
            return .deepSeek
        }
        if value.contains("cohere") {
            return .cohere
        }
        if value.contains("moss") {
            return .moss
        }
        if value.contains("granite") || value.contains("ibm") {
            return .granite
        }
        if value.contains("firered") || value.contains("fire red") || value.contains("小红书") {
            return .fireRed
        }
        if value.contains("sensevoice") || value.contains("sense voice") || value.contains("sensenova") || value.contains("sense") {
            return .sense
        }
        if value.contains("openrouter") {
            return .openRouter
        }
        if value.contains("lm studio") || value.contains("lmstudio") {
            return .lmStudio
        }
        if value.contains("lfm") || value.contains("liquid") {
            return .liquid
        }
        if value.contains("minimax") {
            return .minimax
        }
        if value.contains("doubao") || value.contains("豆包") {
            return .doubao
        }
        if value.contains("xiaomi") || value.contains("mimo") || value.contains("小米") {
            return .xiaomi
        }
        if value.contains("hy-mt2") || value.contains("hunyuan") || value.contains("混元") {
            return .hunyuan
        }
        if value.contains("stepfun") || value.contains("阶跃") {
            return .stepFun
        }
        if value.contains("volc") || value.contains("火山") {
            return .volcengine
        }
        if value.contains("kimi") || value.contains("moonshot") {
            return .kimi
        }
        if value.contains("grok") || value.contains("xai") {
            return .grok
        }
        if value.contains("qwen") || value.contains("funasr") || value.contains("通义") {
            return .qwen
        }
        if value.contains("glm")
            || value.contains("zhipu")
            || value.contains("z.ai")
            || value.contains("zai")
            || value.contains("智谱")
            || value.contains("智普") {
            return .zhipu
        }
        if value.contains("gemma") {
            return .gemma
        }
        if value == "google"
            || value.hasPrefix("google gemini")
            || value == "google remote llm"
            || value.hasPrefix("google remote llm ")
            || value == "google 远程 llm"
            || value.hasPrefix("google 远程 llm ") {
            return .google
        }
        if value.contains("gemini") || value.contains("google") {
            return .gemini
        }
        if value.contains("llama") || value.contains("meta") {
            return .meta
        }
        if value.contains("canary") || value.contains("parakeet") || value.contains("nemotron") || value.contains("nvidia") {
            return .nvidia
        }
        if value.contains("mistral") || value.contains("voxtral") {
            return .mistral
        }
        if value.contains("ollama") || value.contains("omlx") {
            return .ollama
        }
        if value.contains("aliyun") || value.contains("alibaba") || value.contains("bailian") || value.contains("阿里") {
            return .alibaba
        }
        if value.contains("whisper") || value.contains("codex") || value.contains("openai") || value.contains("chatgpt") {
            return .openAI
        }

        return .generic
    }
}

struct ModelLogoView: View {
    let key: ModelLogoKey
    var fallbackTitle: String = ""
    var size: CGFloat = 18

    var body: some View {
        Group {
            if let image = ModelLogoImageStore.image(for: key) {
                Image(nsImage: image)
                    .resizable()
                    .renderingMode(key.isTemplate ? .template : .original)
                    .foregroundStyle(.primary)
                    .scaledToFit()
            } else {
                fallbackBadge
            }
        }
        .frame(width: size, height: size)
        .accessibilityHidden(true)
    }

    private var fallbackBadge: some View {
        ZStack {
            RoundedRectangle(cornerRadius: size * 0.26, style: .continuous)
                .fill(SettingsUIStyle.groupedFillColor)
                .overlay(
                    RoundedRectangle(cornerRadius: size * 0.26, style: .continuous)
                        .strokeBorder(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
                )

            Text(fallbackText)
                .font(.system(size: fallbackText.count > 2 ? size * 0.32 : size * 0.43, weight: .bold, design: .rounded))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.62)
                .padding(.horizontal, 1)
        }
    }

    private var fallbackText: String {
        let titleText = initials(from: fallbackTitle)
        return titleText.isEmpty ? key.fallbackText : titleText
    }

    private func initials(from title: String) -> String {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "" }

        let separators = CharacterSet.alphanumerics.inverted
        let parts = trimmed
            .components(separatedBy: separators)
            .filter { !$0.isEmpty }

        if parts.count >= 2 {
            return parts.prefix(2)
                .compactMap { $0.first.map(String.init) }
                .joined()
                .uppercased()
        }

        let compact = parts.first ?? trimmed
        if compact.unicodeScalars.allSatisfy({ CharacterSet.asciiLetters.contains($0) }) {
            return String(compact.prefix(2)).uppercased()
        }

        return String(compact.prefix(1))
    }
}

private extension CharacterSet {
    static let asciiLetters = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
}

private enum ModelLogoImageStore {
    private static var cache = [ModelLogoKey: NSImage]()

    static func image(for key: ModelLogoKey) -> NSImage? {
        if let cached = cache[key] {
            return cached
        }
        guard let resourceName = key.resourceName,
              let url = resourceURL(named: resourceName),
              let image = NSImage(contentsOf: url) else {
            return nil
        }

        image.size = NSSize(width: 24, height: 24)
        image.isTemplate = key.isTemplate
        cache[key] = image
        return image
    }

    private static func resourceURL(named resourceName: String) -> URL? {
        if let url = Bundle.main.url(
            forResource: resourceName,
            withExtension: "svg",
            subdirectory: "ModelIcons"
        ) {
            return url
        }

        if let url = Bundle.main.url(
            forResource: resourceName,
            withExtension: "svg",
            subdirectory: "Voxt/ModelIcons"
        ) {
            return url
        }

        return Bundle.main.urls(forResourcesWithExtension: "svg", subdirectory: nil)?
            .first { $0.deletingPathExtension().lastPathComponent == resourceName }
    }
}

extension ModelCatalogEntry {
    var modelLogoKey: ModelLogoKey {
        return ModelLogoKey.resolve(title: title, engine: engine)
    }
}

extension ModelCatalogGroupSection {
    var modelLogoKey: ModelLogoKey {
        ModelLogoKey.resolve(title: title, engine: engine)
    }
}

extension FeatureModelSelectorEntry {
    var modelLogoKey: ModelLogoKey {
        return ModelLogoKey.resolve(title: title, engine: engine)
    }
}

extension FeatureModelSelectorGroupSection {
    var modelLogoKey: ModelLogoKey {
        ModelLogoKey.resolve(title: title, engine: engine)
    }
}
