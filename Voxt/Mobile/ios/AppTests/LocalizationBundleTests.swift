import XCTest

final class LocalizationBundleTests: XCTestCase {
    func testMicrophonePermissionTranslationsAreInTheHostAppBundle() throws {
        // Package resources alone cannot localize a system permission dialog.
        let expected = [
            "en": "Voxt records your voice and sends it to your Mac to turn it into text.",
            "zh-Hans": "Voxt 会录制你的语音并发送到你的 Mac，将其转换为文字。",
            "zh-Hant": "Voxt 會錄製你的語音並傳送到你的 Mac，將其轉換為文字。",
            "ja": "Voxt は音声を録音して Mac に送り、テキストに変換します。"
        ]
        for (language, value) in expected {
            let path = try XCTUnwrap(Bundle.main.path(forResource: language, ofType: "lproj"), language)
            let bundle = try XCTUnwrap(Bundle(path: path))
            XCTAssertEqual(bundle.localizedString(forKey: "NSMicrophoneUsageDescription", value: "missing", table: "InfoPlist"), value)
        }
        XCTAssertNil(Bundle.main.path(forResource: "xx-invalid", ofType: "lproj"))
    }

    /// Every shipped interface language carries all three system permission texts, so no
    /// iOS permission prompt falls back to English.
    func testEveryShippedLanguageHasEveryPermissionText() throws {
        let keys = ["NSMicrophoneUsageDescription", "NSCameraUsageDescription", "NSLocalNetworkUsageDescription"]
        for language in ["en", "zh-Hans", "zh-Hant", "ja", "es", "ko"] {
            let path = try XCTUnwrap(Bundle.main.path(forResource: language, ofType: "lproj"), "\(language).lproj is missing")
            let bundle = try XCTUnwrap(Bundle(path: path))
            for key in keys {
                let value = bundle.localizedString(forKey: key, value: "missing", table: "InfoPlist")
                XCTAssertNotEqual(value, "missing", "\(language) has no \(key)")
                XCTAssertTrue(value.contains("Voxt"), "\(language) \(key): \(value)")
            }
        }
        let es = try XCTUnwrap(Bundle(path: try XCTUnwrap(Bundle.main.path(forResource: "es", ofType: "lproj"))))
        XCTAssertEqual(es.localizedString(forKey: "NSMicrophoneUsageDescription", value: nil, table: "InfoPlist"),
                       "Voxt graba tu voz y la envía a tu Mac para convertirla en texto.")
        let ko = try XCTUnwrap(Bundle(path: try XCTUnwrap(Bundle.main.path(forResource: "ko", ofType: "lproj"))))
        XCTAssertEqual(ko.localizedString(forKey: "NSMicrophoneUsageDescription", value: nil, table: "InfoPlist"),
                       "Voxt는 음성을 녹음해 Mac으로 보내 글로 바꿉니다.")
    }
}
