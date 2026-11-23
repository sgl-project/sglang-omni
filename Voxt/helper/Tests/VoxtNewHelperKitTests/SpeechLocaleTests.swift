import XCTest
@testable import VoxtNewHelperKit

/// speechLocale: the user's main language to a locale SFSpeechRecognizer has.
final class SpeechLocaleTests: XCTestCase {
    let supported = ["en-AU", "en-GB", "en-US", "yue-CN", "zh-CN", "zh-HK", "zh-TW", "ja-JP"]

    func testPicksTheRegion() {
        let cases: [(lang: String, region: String?, want: String?)] = [
            ("zh", "US", "zh-CN"),      // no zh-US: Chinese's likely region
            ("zh", "TW", "zh-TW"),      // the user's region
            ("zh-Hant", "CN", "zh-TW"), // the script says the region, not the user's
            ("en", "GB", "en-GB"),
            ("en", "CN", "en-US"),
            ("en-AU", "US", "en-AU"),   // the tag itself
            ("yue", nil, "yue-CN"),     // no yue-HK: the only one of its language
            ("ja", nil, "ja-JP"),
            ("fr", "FR", nil),          // not supported at all
        ]
        for c in cases {
            XCTAssertEqual(speechLocale(c.lang, supported: supported, userRegion: c.region), c.want, "\(c.lang) in \(c.region ?? "-")")
        }
        XCTAssertEqual(speechLocale("zh", supported: ["zh_CN"], userRegion: nil), "zh-CN", "Locale.identifier's underscores")
    }
}
