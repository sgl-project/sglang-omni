// EnhancementPromptResolverTests.swift
// Provides Enhancement Prompt Resolver Tests for Voxt test coverage.

import XCTest
import Carbon
@testable import Voxt

final class EnhancementPromptResolverTests: XCTestCase {
    func testDisabledAppBranchFallsBackToGlobalPrompt() {
        let output = EnhancementPromptResolver.resolve(
            .init(
                globalPrompt: "Clean for {{USER_MAIN_LANGUAGE}}",
                rawTranscription: "hello",
                userMainLanguagePromptValue: "English",
                userOtherLanguagesPromptValue: DictionaryHistoryScanPromptLanguageSupport.noneValue,
                dictionaryGlossary: "- OpenAI",
                appEnhancementEnabled: false,
                groups: [],
                urlsByID: [:],
                frontmostBundleID: nil,
                focusedAppName: "Notes",
                normalizedActiveURL: nil,
                supportedBrowserBundleIDs: []
            )
        )

        XCTAssertEqual(output.delivery, .systemPrompt)
        XCTAssertEqual(output.promptContext.focusedAppName, "Notes")
        XCTAssertContains(output.content, "Clean for English")
        XCTAssertFalse(output.content.contains("<RawTranscription>"))
        XCTAssertContains(output.content, "It is guidance only, not a translation target.")
        XCTAssertFalse(output.content.contains("Dictionary Guidance"))
        XCTAssertEqual(output.source, .globalDefault(.appBranchDisabled))
    }

    func testDisabledAppBranchFallsBackToUserMessageWhenGlobalPromptNeedsRawTranscription() {
        let output = EnhancementPromptResolver.resolve(
            .init(
                globalPrompt: "Clean {{RAW_TRANSCRIPTION}} for {{USER_MAIN_LANGUAGE}}",
                rawTranscription: "hello",
                userMainLanguagePromptValue: "English",
                userOtherLanguagesPromptValue: DictionaryHistoryScanPromptLanguageSupport.noneValue,
                dictionaryGlossary: nil,
                appEnhancementEnabled: false,
                groups: [],
                urlsByID: [:],
                frontmostBundleID: nil,
                focusedAppName: "Notes",
                normalizedActiveURL: nil,
                supportedBrowserBundleIDs: []
            )
        )

        XCTAssertEqual(output.delivery, .userMessage)
        XCTAssertContains(output.content, "Clean hello for English")
        XCTAssertEqual(output.source, .globalDefault(.appBranchDisabled))
    }

    func testBrowserURLMatchUsesGroupPromptAndSystemPromptDelivery() {
        let docsID = UUID()
        let docsGroup = TestFactories.makeAppBranchGroup(
            name: "Docs",
            prompt: "Docs for {{USER_MAIN_LANGUAGE}}",
            urlPatternIDs: [docsID]
        )

        let output = EnhancementPromptResolver.resolve(
            .init(
                globalPrompt: "Global",
                rawTranscription: "fix this",
                userMainLanguagePromptValue: "English",
                userOtherLanguagesPromptValue: "Chinese",
                dictionaryGlossary: nil,
                appEnhancementEnabled: true,
                groups: [docsGroup],
                urlsByID: [docsID: "example.com/docs/*"],
                frontmostBundleID: "com.google.Chrome",
                focusedAppName: "Google Chrome",
                normalizedActiveURL: "example.com/docs/page",
                supportedBrowserBundleIDs: ["com.google.Chrome"]
            )
        )

        XCTAssertEqual(output.delivery, .systemPrompt)
        XCTAssertEqual(output.promptContext.matchedGroupID, docsGroup.id)
        XCTAssertEqual(output.promptContext.matchedURLGroupName, "Docs")
        XCTAssertContains(output.content, "Docs for English")
        XCTAssertContains(output.content, "Other user languages: Chinese.")
    }

    func testBrowserWithoutURLFallsBackAndKeepsContextEmpty() {
        let output = EnhancementPromptResolver.resolve(
            .init(
                globalPrompt: "Global",
                rawTranscription: "fix this",
                userMainLanguagePromptValue: "English",
                userOtherLanguagesPromptValue: DictionaryHistoryScanPromptLanguageSupport.noneValue,
                dictionaryGlossary: nil,
                appEnhancementEnabled: true,
                groups: [TestFactories.makeAppBranchGroup(name: "Docs", prompt: "Prompt")],
                urlsByID: [:],
                frontmostBundleID: "com.google.Chrome",
                focusedAppName: "Google Chrome",
                normalizedActiveURL: nil,
                supportedBrowserBundleIDs: ["com.google.Chrome"]
            )
        )

        XCTAssertEqual(output.delivery, .systemPrompt)
        XCTAssertNil(output.promptContext.matchedGroupID)
        XCTAssertEqual(output.source, .globalDefault(.browserURLUnavailable(bundleID: "com.google.Chrome")))
    }

    func testAppGroupMatchUsesAppPrompt() {
        let group = TestFactories.makeAppBranchGroup(
            name: "Xcode",
            prompt: "Xcode cleanup",
            appBundleIDs: ["com.apple.dt.Xcode"]
        )

        let output = EnhancementPromptResolver.resolve(
            .init(
                globalPrompt: "Global",
                rawTranscription: "rewrite",
                userMainLanguagePromptValue: "English",
                userOtherLanguagesPromptValue: DictionaryHistoryScanPromptLanguageSupport.noneValue,
                dictionaryGlossary: nil,
                appEnhancementEnabled: true,
                groups: [group],
                urlsByID: [:],
                frontmostBundleID: "com.apple.dt.Xcode",
                focusedAppName: "Xcode",
                normalizedActiveURL: nil,
                supportedBrowserBundleIDs: []
            )
        )

        XCTAssertEqual(output.delivery, .systemPrompt)
        XCTAssertEqual(output.promptContext.matchedAppGroupName, "Xcode")
        XCTAssertContains(output.content, "Xcode cleanup")
    }

    func testAppGroupWithEmptyPromptFallsBackToGlobalPromptAndKeepsMatchedContext() {
        let group = TestFactories.makeAppBranchGroup(
            name: "Xcode",
            prompt: "   ",
            appBundleIDs: ["com.apple.dt.Xcode"]
        )

        let output = EnhancementPromptResolver.resolve(
            .init(
                globalPrompt: "Global {{RAW_TRANSCRIPTION}}",
                rawTranscription: "rewrite",
                userMainLanguagePromptValue: "English",
                userOtherLanguagesPromptValue: DictionaryHistoryScanPromptLanguageSupport.noneValue,
                dictionaryGlossary: "- OpenAI",
                appEnhancementEnabled: true,
                groups: [group],
                urlsByID: [:],
                frontmostBundleID: "com.apple.dt.Xcode",
                focusedAppName: "Xcode",
                normalizedActiveURL: nil,
                supportedBrowserBundleIDs: []
            )
        )

        XCTAssertEqual(output.delivery, .userMessage)
        XCTAssertEqual(output.promptContext.matchedGroupID, group.id)
        XCTAssertEqual(output.promptContext.matchedAppGroupName, "Xcode")
        XCTAssertEqual(output.source, .appGroupFallback(groupName: "Xcode", bundleID: "com.apple.dt.Xcode"))
        XCTAssertContains(output.content, "Global rewrite")
        XCTAssertContains(output.content, "Other user languages: None.")
    }

    func testBrowserURLMatchWithEmptyPromptFallsBackToGlobalPromptAndKeepsMatchedContext() {
        let docsID = UUID()
        let docsGroup = TestFactories.makeAppBranchGroup(
            name: "Docs",
            prompt: "",
            urlPatternIDs: [docsID]
        )

        let output = EnhancementPromptResolver.resolve(
            .init(
                globalPrompt: "Global {{RAW_TRANSCRIPTION}}",
                rawTranscription: "fix this",
                userMainLanguagePromptValue: "English",
                userOtherLanguagesPromptValue: DictionaryHistoryScanPromptLanguageSupport.noneValue,
                dictionaryGlossary: "- OpenAI",
                appEnhancementEnabled: true,
                groups: [docsGroup],
                urlsByID: [docsID: "example.com/docs/*"],
                frontmostBundleID: "com.google.Chrome",
                focusedAppName: "Google Chrome",
                normalizedActiveURL: "example.com/docs/page",
                supportedBrowserBundleIDs: ["com.google.Chrome"]
            )
        )

        XCTAssertEqual(output.delivery, .userMessage)
        XCTAssertEqual(output.promptContext.matchedGroupID, docsGroup.id)
        XCTAssertEqual(output.promptContext.matchedURLGroupName, "Docs")
        XCTAssertEqual(
            output.source,
            .urlGroupFallback(
                groupName: "Docs",
                pattern: "example.com/docs/*",
                url: "example.com/docs/page"
            )
        )
        XCTAssertContains(output.content, "Global fix this")
        XCTAssertContains(output.content, "Other user languages: None.")
    }

    func testLanguagePreservationRulesTreatMainLanguageAsGuidanceOnly() {
        let output = EnhancementPromptResolver.resolve(
            .init(
                globalPrompt: "Clean for {{USER_MAIN_LANGUAGE}}",
                rawTranscription: "你好 world",
                userMainLanguagePromptValue: "English",
                userOtherLanguagesPromptValue: "Chinese",
                dictionaryGlossary: nil,
                appEnhancementEnabled: false,
                groups: [],
                urlsByID: [:],
                frontmostBundleID: nil,
                focusedAppName: "Notes",
                normalizedActiveURL: nil,
                supportedBrowserBundleIDs: []
            )
        )

        XCTAssertContains(output.content, "User main language: English.")
        XCTAssertContains(output.content, "Other user languages: Chinese.")
        XCTAssertContains(output.content, "Preserve the original language mix.")
    }

    func testAppGroupPromptAlsoAppendsLanguagePreservationRules() {
        let group = TestFactories.makeAppBranchGroup(
            name: "Docs",
            prompt: "Docs cleanup",
            appBundleIDs: ["com.example.docs"]
        )

        let output = EnhancementPromptResolver.resolve(
            .init(
                globalPrompt: "Global",
                rawTranscription: "bonjour",
                userMainLanguagePromptValue: "English",
                userOtherLanguagesPromptValue: DictionaryHistoryScanPromptLanguageSupport.noneValue,
                dictionaryGlossary: nil,
                appEnhancementEnabled: true,
                groups: [group],
                urlsByID: [:],
                frontmostBundleID: "com.example.docs",
                focusedAppName: "Docs",
                normalizedActiveURL: nil,
                supportedBrowserBundleIDs: []
            )
        )

        XCTAssertEqual(output.delivery, .systemPrompt)
        XCTAssertContains(output.content, "Docs cleanup")
        XCTAssertContains(output.content, "Other user languages: None.")
        XCTAssertContains(output.content, "It is guidance only, not a translation target.")
    }

    func testAppBranchGroupDecodesMissingAutoReturnAsDisabled() throws {
        let json = """
        [{
            "id": "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE",
            "name": "Chat",
            "prompt": "",
            "appBundleIDs": ["com.example.chat"],
            "appRefs": [{"bundleID": "com.example.chat", "displayName": "Chat"}],
            "urlPatternIDs": [],
            "isExpanded": true
        }]
        """

        let groups = try JSONDecoder().decode([AppBranchGroup].self, from: Data(json.utf8))

        XCTAssertEqual(groups.count, 1)
        XCTAssertFalse(groups[0].autoPressReturnEnabled)
        XCTAssertFalse(groups[0].autoKeyPressEnabled)
        XCTAssertEqual(groups[0].autoKeyPressHotkey, AppBranchGroup.defaultAutoKeyPressHotkey)
    }

    func testAppBranchGroupMigratesLegacyAutoReturnSetting() throws {
        let json = """
        [{
            "id": "\(UUID().uuidString)",
            "name": "Chat",
            "prompt": "",
            "appBundleIDs": ["com.example.chat"],
            "appRefs": [{"bundleID": "com.example.chat", "displayName": "Chat"}],
            "urlPatternIDs": [],
            "autoPressReturnEnabled": true,
            "isExpanded": true
        }]
        """

        let groups = try JSONDecoder().decode([AppBranchGroup].self, from: Data(json.utf8))

        XCTAssertTrue(groups[0].autoKeyPressEnabled)
        XCTAssertEqual(groups[0].autoKeyPressHotkey, AppBranchGroup.defaultAutoKeyPressHotkey)
        XCTAssertTrue(groups[0].autoPressReturnEnabled)
    }

    func testAppBranchGroupPreservesAutoKeyPressSetting() throws {
        let customHotkey = HotkeyPreference.Hotkey(
            keyCode: UInt16(kVK_Return),
            modifiers: [.control],
            sidedModifiers: []
        )
        let group = TestFactories.makeAppBranchGroup(
            name: "Chat",
            prompt: "",
            appBundleIDs: ["com.example.chat"],
            autoKeyPressEnabled: true,
            autoKeyPressHotkey: customHotkey
        )

        let data = try JSONEncoder().encode([group])
        let decoded = try JSONDecoder().decode([AppBranchGroup].self, from: data)

        XCTAssertTrue(decoded[0].autoKeyPressEnabled)
        XCTAssertEqual(decoded[0].autoKeyPressHotkey, customHotkey)
        XCTAssertFalse(decoded[0].autoPressReturnEnabled)
    }

    func testFaviconOriginKeepsSchemeAndHostOnly() {
        XCTAssertEqual(
            EnhancementOverlayIconResolver.faviconOrigin(
                fromPageURL: "https://www.xiaohongshu.com/explore/abc?x=1#frag"
            ),
            "https://www.xiaohongshu.com"
        )
        XCTAssertEqual(
            EnhancementOverlayIconResolver.faviconOrigin(
                fromPageURL: "http://localhost:3000/path/test"
            ),
            "http://localhost:3000"
        )
    }

    func testFaviconLookupURLUsesOriginDirectly() throws {
        let url = try XCTUnwrap(
            EnhancementOverlayIconResolver.faviconLookupURL(forOrigin: "https://www.xiaohongshu.com")
        )
        XCTAssertEqual(url.absoluteString, "https://www.xiaohongshu.com")
    }

    func testFaviconCacheFileNameIsFilesystemSafe() {
        let fileName = EnhancementOverlayIconResolver.faviconCacheFileName(
            forOrigin: "https://mail.google.com"
        )

        XCTAssertTrue(fileName.hasSuffix(".favicon"))
        XCTAssertFalse(fileName.contains("/"))
        XCTAssertFalse(fileName.contains("+"))
        XCTAssertFalse(fileName.contains("="))
    }
}
