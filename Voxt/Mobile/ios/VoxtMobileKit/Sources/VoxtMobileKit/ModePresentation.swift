import Foundation

extension CompositionOptions.Mode {
    public var keyboardSymbol: String? {
        switch self {
        case .dictate: "mic"
        case .edit, .translate, .write: nil
        }
    }

    public func keyboardLabel(language: String?) -> String? {
        switch self {
        case .translate:
            let code = language ?? "en"
            return "→ " + (code.hasPrefix("zh") ? "中" : String(code.prefix(2)).uppercased())
        case .write: return VoxtL10n.shared.string("Write")
        case .edit: return VoxtL10n.shared.string("Edit")
        case .dictate: return nil
        }
    }
}
