import Foundation
import KoreanInput
import SpanishInput
import TypingInput
import VoxtMobileKit
// Japanese, pinyin and Zhuyin come from ios/InputMethods.yml; a build without it offers the rest.
#if canImport(JapaneseInput)
import JapaneseInput
#endif
#if canImport(PinyinInput)
import PinyinInput
#endif
#if canImport(ZhuyinInput)
import ZhuyinInput
#endif

/// Every typing layout this keyboard offers through the TypingInput contract.
/// Adding a layout is one line here plus its package, which implements the TypingInput contract.
/// Latin layouts without an entry still come from VoxtMobileKit's catalog.
enum TypingLayouts {
    /// Letter keys of a `.keys` layout as the shared key grid draws them, with shift applied.
    static func keys(_ rows: [[KeySpec]], shifted: Bool) -> [[TextKeyboardKey]] {
        rows.map { row in
            row.map { spec in
                let text = spec.text(shifted: shifted)
                return TextKeyboardKey(id: spec.id, kind: kind(spec.role), label: spec.role == .character ? (text ?? spec.label) : spec.label,
                                       text: text, alternates: spec.alternates)
            }
        }
    }

    private static func kind(_ role: KeySpec.Role) -> TextKeyboardKey.Kind {
        switch role {
        case .character, .modifier: .character
        case .shift: .shift
        case .delete: .delete
        case .space: .space
        case .return: .return
        case .numbers: .numbers
        case .voice: .voice
        }
    }

    #if canImport(PinyinInput)
    /// Picks are remembered in the App Group (Settings can clear them); fuzzy
    /// z/zh, c/ch, s/sh follows the Settings switch on every conversion.
    private static let pinyin = PinyinLayout.descriptor(
        learning: PinyinLearning(fileURL: PinyinSettings.learnedWordsURL),
        fuzzy: { PinyinSettings.fuzzy(in: UserDefaults(suiteName: DictationChannel.appGroup) ?? .standard) })
    #endif

    static let registry: LayoutRegistry = {
        var layouts = [SpanishLayout.descriptor]
        #if canImport(JapaneseInput)
        layouts.append(JapaneseLayout.descriptor)
        #endif
        #if canImport(ZhuyinInput)
        layouts.append(ZhuyinLayout.descriptor)
        #endif
        layouts.append(KoreanLayout.descriptor)
        #if canImport(PinyinInput)
        layouts.append(pinyin)
        #endif
        do {
            return try LayoutRegistry(layouts)
        } catch {
            preconditionFailure("Duplicate typing layout id: \(error)")
        }
    }()
}
