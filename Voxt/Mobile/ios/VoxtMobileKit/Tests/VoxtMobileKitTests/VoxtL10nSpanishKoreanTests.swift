import Foundation
import Testing
@testable import VoxtMobileKit

/// Spanish and Korean ship like the other interface languages: the system picks them,
/// and every catalog key has them (everyCatalogKeyIsTranslatedInEveryShippedLanguage).
@Suite struct VoxtL10nSpanishKoreanTests {
    @Test func spanishAndKoreanAreShippedAndPickedFromTheSystem() {
        #expect(VoxtL10n.shippedLanguages.contains("es"))
        #expect(VoxtL10n.shippedLanguages.contains("ko"))
        #expect(VoxtL10n.resolve(preferredLanguages: ["es-MX"]) == "es")
        #expect(VoxtL10n.resolve(preferredLanguages: ["es-419", "en-US"]) == "es")
        #expect(VoxtL10n.resolve(preferredLanguages: ["ko-KR"]) == "ko")
        #expect(VoxtL10n.resolve(preferredLanguages: ["pt-BR"]) == "en")
    }

    @Test func knownStringsInSpanishAndKorean() {
        let l10n = VoxtL10n(defaults: UserDefaults(suiteName: "l10n-es-ko-\(UUID().uuidString)")!)
        #expect(l10n.string("Cancel", language: "es") == "Cancelar")
        #expect(l10n.string("Cancel", language: "ko") == "취소")
        #expect(l10n.string("Tap to speak", language: "es") == "Toca para hablar")
        #expect(l10n.plural("%d words", count: 1, language: "es") == "1 palabra")
        #expect(l10n.plural("%d words", count: 3, language: "es") == "3 palabras")
        #expect(l10n.plural("%d words", count: 3, language: "ko") == "3단어")
        #expect(l10n.format("Connected to Voxt %@", language: "ko", arguments: ["1.16"]) == "Voxt 1.16에 연결됨")
    }
}
