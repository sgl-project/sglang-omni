// LocalizationResourcesTests.swift
// Provides Localization Resources Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class LocalizationResourcesTests: XCTestCase {
    func testFeatureMenuTitleIsLocalizedInSupportedLanguages() {
        XCTAssertEqual(AppLocalization.localizedString("Feature", localeIdentifier: "en"), "Feature")
        XCTAssertEqual(AppLocalization.localizedString("Feature", localeIdentifier: "zh-Hans"), "功能")
        XCTAssertEqual(AppLocalization.localizedString("Feature", localeIdentifier: "ja"), "機能")
    }

    func testHotkeyLabelsAreLocalizedInSupportedLanguages() {
        XCTAssertEqual(AppLocalization.localizedString("Tap", localeIdentifier: "zh-Hans"), "点按")
        XCTAssertEqual(AppLocalization.localizedString("Unassigned", localeIdentifier: "zh-Hans"), "未设置")
        XCTAssertEqual(AppLocalization.localizedString("Right %@", localeIdentifier: "zh-Hans"), "右侧%@")
        XCTAssertEqual(AppLocalization.localizedString("Tap", localeIdentifier: "ja"), "タップ")
        XCTAssertEqual(AppLocalization.localizedString("Unassigned", localeIdentifier: "ja"), "未設定")
        XCTAssertEqual(AppLocalization.localizedString("Right %@", localeIdentifier: "ja"), "右%@")
    }

    func testBackLabelIsLocalizedInSupportedLanguages() {
        XCTAssertEqual(AppLocalization.localizedString("Back", localeIdentifier: "en"), "Back")
        XCTAssertEqual(AppLocalization.localizedString("Back", localeIdentifier: "zh-Hans"), "返回")
        XCTAssertEqual(AppLocalization.localizedString("Back", localeIdentifier: "ja"), "戻る")
    }

    func testAppEnhancementLabelIsLocalizedInSupportedLanguages() {
        XCTAssertEqual(AppLocalization.localizedString("App Enhancement", localeIdentifier: "en"), "App Branch")

        let chinese = AppLocalization.localizedString("App Enhancement", localeIdentifier: "zh-Hans")
        XCTAssertFalse(chinese.isEmpty)
        XCTAssertNotEqual(chinese, "App Enhancement")

        let japanese = AppLocalization.localizedString("App Enhancement", localeIdentifier: "ja")
        XCTAssertFalse(japanese.isEmpty)
        XCTAssertNotEqual(japanese, "App Enhancement")
    }

    func testAppEnhancementDescriptionIsLocalizedInSupportedLanguages() {
        let key = "Use different enhancement prompts for different apps or browser pages."

        XCTAssertEqual(AppLocalization.localizedString(key, localeIdentifier: "en"), key)

        let chinese = AppLocalization.localizedString(key, localeIdentifier: "zh-Hans")
        XCTAssertFalse(chinese.isEmpty)
        XCTAssertNotEqual(chinese, key)

        let japanese = AppLocalization.localizedString(key, localeIdentifier: "ja")
        XCTAssertFalse(japanese.isEmpty)
        XCTAssertNotEqual(japanese, key)
    }

    func testFeatureLabelsAreLocalizedInSupportedLanguages() {
        let keys = [
            "Transcription",
            "Translation",
            "Rewrite"
        ]

        for key in keys {
            XCTAssertEqual(AppLocalization.localizedString(key, localeIdentifier: "en"), key)

            let chinese = AppLocalization.localizedString(key, localeIdentifier: "zh-Hans")
            XCTAssertFalse(chinese.isEmpty, "Missing zh-Hans localization for \(key)")
            XCTAssertNotEqual(chinese, key, "Expected zh-Hans translation for \(key)")

            let japanese = AppLocalization.localizedString(key, localeIdentifier: "ja")
            XCTAssertFalse(japanese.isEmpty, "Missing ja localization for \(key)")
            XCTAssertNotEqual(japanese, key, "Expected ja translation for \(key)")
        }
    }

    func testModelFilterLabelsAreLocalizedInSupportedLanguages() {
        let keys = ["Local", "Remote", "Installed", "Configured", "In Use"]

        for key in keys {
            XCTAssertEqual(AppLocalization.localizedString(key, localeIdentifier: "en"), key)

            let chinese = AppLocalization.localizedString(key, localeIdentifier: "zh-Hans")
            XCTAssertFalse(chinese.isEmpty, "Missing zh-Hans localization for \(key)")
            XCTAssertNotEqual(chinese, key, "Expected zh-Hans translation for \(key)")

            let japanese = AppLocalization.localizedString(key, localeIdentifier: "ja")
            XCTAssertFalse(japanese.isEmpty, "Missing ja localization for \(key)")
            XCTAssertNotEqual(japanese, key, "Expected ja translation for \(key)")
        }
    }

    func testOpeningUpdateWindowLabelIsLocalizedInSupportedLanguages() {
        let key = "Opening update window…"

        XCTAssertEqual(AppLocalization.localizedString(key, localeIdentifier: "en"), key)
        XCTAssertEqual(AppLocalization.localizedString(key, localeIdentifier: "zh-Hans"), "正在打开更新窗口…")
        XCTAssertEqual(AppLocalization.localizedString(key, localeIdentifier: "ja"), "更新ウィンドウを開いています…")
    }

    func testBetaUpdatesLabelIsLocalizedInSupportedLanguages() {
        XCTAssertEqual(AppLocalization.localizedString("Beta Updates", localeIdentifier: "en"), "Beta Updates")
        XCTAssertEqual(AppLocalization.localizedString("Beta Updates", localeIdentifier: "zh-Hans"), "Beta 更新")
        XCTAssertEqual(AppLocalization.localizedString("Beta Updates", localeIdentifier: "ja"), "ベータ更新")
    }

    func testManualCorrectionActionIsLocalizedInSupportedLanguages() {
        XCTAssertEqual(AppLocalization.localizedString("Correct", localeIdentifier: "en"), "Correct")
        XCTAssertEqual(AppLocalization.localizedString("Correct", localeIdentifier: "zh-Hans"), "纠错")
        XCTAssertEqual(AppLocalization.localizedString("Correct", localeIdentifier: "ja"), "修正")
    }

    func testSettingsDialogLabelsAreLocalizedInSupportedLanguages() {
        XCTAssertEqual(AppLocalization.localizedString("Export Logs", localeIdentifier: "zh-Hans"), "导出日志")
        XCTAssertEqual(AppLocalization.localizedString("Export Logs", localeIdentifier: "ja"), "ログを書き出す")
        XCTAssertEqual(AppLocalization.localizedString("Storage Path", localeIdentifier: "zh-Hans"), "存储路径")
        XCTAssertEqual(AppLocalization.localizedString("Storage Path", localeIdentifier: "ja"), "保存パス")
    }

    func testHistoryAudioUnavailableMessagesAreLocalizedInSupportedLanguages() {
        let keys = [
            "No saved audio file is available for this record. It may have been created before audio saving was enabled, or the audio file may have been removed.",
            "Audio playback is unavailable because history audio saving is turned off. Turn it on before recording to keep audio for future records."
        ]

        for key in keys {
            XCTAssertEqual(AppLocalization.localizedString(key, localeIdentifier: "en"), key)
            XCTAssertNotEqual(
                AppLocalization.localizedString(key, localeIdentifier: "zh-Hans"),
                key,
                "Missing zh-Hans localization for \(key)"
            )
            XCTAssertNotEqual(
                AppLocalization.localizedString(key, localeIdentifier: "ja"),
                key,
                "Missing ja localization for \(key)"
            )
        }
    }

    func testConfirmationDialogCopyIsLocalizedInSupportedLanguages() {
        let keys = [
            "Cancel",
            "Confirm",
            "Delete",
            "Uninstall",
            "Uninstall Model?",
            "Uninstall %@ from this Mac? You can download it again later.",
            "This removes the downloaded model files from this Mac. You can download them again later.",
            "Delete Dictionary Category?",
            "Move Terms to Default",
            "Delete Terms Too",
            "You can move this category's terms to the default category or delete them together.",
            "Archive the current note database and create a new one?",
            "Archive and Rebuild",
            "The existing database will be preserved in a recovery folder. Notes in it will not appear in the new database.",
            "Change Model Storage Path?",
            "After changing the model storage path, previously downloaded local models will need to be downloaded again.",
            "Delete All History?",
            "Delete All %@ History?",
            "Delete All Notes?",
            "This will permanently delete all history entries.",
            "This will permanently delete all entries in %@ history.",
            "This will permanently delete all notes.",
            "End this meeting transcription?",
            "Canceling will discard this meeting; finishing will save it and open Meeting Details.",
            "Cancel Transcription",
            "Finish Transcription"
        ]

        for key in keys {
            XCTAssertFalse(
                AppLocalization.localizedString(key, localeIdentifier: "en").isEmpty,
                "Missing en localization for \(key)"
            )
            XCTAssertNotEqual(
                AppLocalization.localizedString(key, localeIdentifier: "zh-Hans"),
                key,
                "Missing zh-Hans localization for \(key)"
            )
            XCTAssertNotEqual(
                AppLocalization.localizedString(key, localeIdentifier: "ja"),
                key,
                "Missing ja localization for \(key)"
            )
        }
    }

    func testAppleASRSettingsLabelsAreLocalizedInSupportedLanguages() {
        let keys = [
            "Follow User Main Language",
            "Primary language",
            "Resolved locale",
            "Other languages",
            "Prefer On-Device Recognition",
            "Add Punctuation",
            "Report Partial Results",
            "Contextual Phrases",
            "Enter one phrase per line. These phrases bias Apple's recognizer toward names, products, and domain terms.",
            "Phrases count"
        ]

        for key in keys {
            XCTAssertNotEqual(
                AppLocalization.localizedString(key, localeIdentifier: "zh-Hans"),
                key,
                "Expected zh-Hans translation for \(key)"
            )
            XCTAssertNotEqual(
                AppLocalization.localizedString(key, localeIdentifier: "ja"),
                key,
                "Expected ja translation for \(key)"
            )
        }

        XCTAssertEqual(AppLocalization.localizedString("Direct Dictation", localeIdentifier: "ja"), "システム音声認識")
    }

    func testExperimentalBadgeUsesShortEnglishLabel() {
        XCTAssertEqual(AppLocalization.localizedString("Experimental", localeIdentifier: "en"), "Beta")
        XCTAssertEqual(AppLocalization.localizedString("Experimental", localeIdentifier: "zh-Hans"), "实验性")
        XCTAssertEqual(AppLocalization.localizedString("Experimental", localeIdentifier: "ja"), "実験中")
    }

    func testMeetingOverlayEnglishLabelsStayCompact() {
        let expectedValues = [
            "Translate": "Translate",
            "Choose Translation Language": "Choose Language",
            "Realtime translation only translates Them segments.": "Translates Them only.",
            "Start Translation": "Start",
            "Cancel Transcription": "Discard",
            "Finish Transcription": "Finish",
            "End this meeting transcription?": "End meeting?",
            "The transcript timeline for Me / Them will appear here once the meeting starts.":
                "The transcript will appear here when the meeting starts.",
            "Automatic scrolling pauses when you scroll away from the bottom.":
                "Scroll up to pause auto-scroll."
        ]

        for (key, expectedValue) in expectedValues {
            XCTAssertEqual(AppLocalization.localizedString(key, localeIdentifier: "en"), expectedValue)
        }

        XCTAssertEqual(AppLocalization.localizedString("Translate", localeIdentifier: "zh-Hans"), "翻译")
        XCTAssertEqual(AppLocalization.localizedString("Translate", localeIdentifier: "ja"), "翻訳")
    }

    func testDashboardEnglishLabelsStayCompact() {
        let expectedValues = [
            "Total Dictation Time": "Dictation Time",
            "Total Dictation Characters": "Dictation Chars",
            "Total Translation Characters": "Translation Chars",
            "Average Dictation Speed": "Avg. Speed",
            "Daily Characters (Last 7 Days)": "7-Day Characters",
            "Branch": "Branch"
        ]

        for (key, expectedValue) in expectedValues {
            XCTAssertEqual(AppLocalization.localizedString(key, localeIdentifier: "en"), expectedValue)
        }

        XCTAssertEqual(AppLocalization.localizedString("Branch", localeIdentifier: "zh-Hans"), "分支")
        XCTAssertEqual(AppLocalization.localizedString("Branch", localeIdentifier: "ja"), "ブランチ")
    }

    func testHistoryNoteStatusEnglishLabelStaysCompact() {
        XCTAssertEqual(AppLocalization.localizedString("All statuses", localeIdentifier: "en"), "All")
    }

    func testHistorySettingsToggleLabelsAreLocalized() {
        XCTAssertEqual(AppLocalization.localizedString("History Cleanup", localeIdentifier: "zh-Hans"), "历史记录清理")
        XCTAssertEqual(AppLocalization.localizedString("Retention Count", localeIdentifier: "zh-Hans"), "保留条数")
        XCTAssertEqual(AppLocalization.localizedString("Unlimited", localeIdentifier: "zh-Hans"), "不限制")
        XCTAssertEqual(AppLocalization.localizedString("Save history audio", localeIdentifier: "zh-Hans"), "保存历史音频")
        XCTAssertEqual(AppLocalization.localizedString("History Cleanup", localeIdentifier: "ja"), "履歴の自動削除")
        XCTAssertEqual(AppLocalization.localizedString("Retention Count", localeIdentifier: "ja"), "保持件数")
        XCTAssertEqual(AppLocalization.localizedString("Unlimited", localeIdentifier: "ja"), "無制限")
        XCTAssertEqual(AppLocalization.localizedString("Save history audio", localeIdentifier: "ja"), "履歴音声を保存")
    }

    func testMeetingFileConversionNotificationLabelsAreLocalized() {
        XCTAssertEqual(AppLocalization.localizedString("File conversion succeeded", localeIdentifier: "zh-Hans"), "文件转换成功")
        XCTAssertEqual(AppLocalization.localizedString("File conversion failed", localeIdentifier: "zh-Hans"), "文件转换失败")
        XCTAssertEqual(AppLocalization.localizedString("File conversion succeeded", localeIdentifier: "ja"), "ファイル変換が完了しました")
        XCTAssertEqual(AppLocalization.localizedString("File conversion failed", localeIdentifier: "ja"), "ファイル変換に失敗しました")
    }

    func testModelDownloadNotificationLabelsAreLocalized() {
        XCTAssertEqual(AppLocalization.localizedString("Model download succeeded", localeIdentifier: "zh-Hans"), "模型下载成功")
        XCTAssertEqual(AppLocalization.localizedString("Model download failed", localeIdentifier: "zh-Hans"), "模型下载失败")
        XCTAssertEqual(AppLocalization.localizedString("Model download succeeded", localeIdentifier: "ja"), "モデルのダウンロードが完了しました")
        XCTAssertEqual(AppLocalization.localizedString("Model download failed", localeIdentifier: "ja"), "モデルのダウンロードに失敗しました")
    }

    func testCodexConfigurationTextIsLocalizedInSupportedLanguages() {
        let keys = [
            "Codex Credentials",
            "Codex Credentials, Voxt uses local Codex OAuth credentials.",
            "For first-time setup, manually choose the auth.json configuration file once.",
            "Codex auth.json permission denied at %@. Click Choose and select auth.json to grant Voxt access.",
            "Codex auth.json has no ChatGPT OAuth tokens. Run `codex login` first."
        ]

        for key in keys {
            XCTAssertEqual(AppLocalization.localizedString(key, localeIdentifier: "en"), key)

            let chinese = AppLocalization.localizedString(key, localeIdentifier: "zh-Hans")
            XCTAssertFalse(chinese.isEmpty, "Missing zh-Hans localization for \(key)")
            XCTAssertNotEqual(chinese, key, "Expected zh-Hans translation for \(key)")

            let japanese = AppLocalization.localizedString(key, localeIdentifier: "ja")
            XCTAssertFalse(japanese.isEmpty, "Missing ja localization for \(key)")
            XCTAssertNotEqual(japanese, key, "Expected ja translation for \(key)")
        }
    }
}
