import Carbon
import XCTest
@testable import Voxt

final class SettingsPermissionSupportTests: XCTestCase {
    private func makeFeatureSettings(
        transcriptionASR: FeatureModelSelectionID = .mlx(MLXModelManager.defaultModelRepo),
        translationASR: FeatureModelSelectionID = .mlx(MLXModelManager.defaultModelRepo),
        rewriteASR: FeatureModelSelectionID = .mlx(MLXModelManager.defaultModelRepo),
        notesEnabled: Bool = true,
        remindersEnabled: Bool = false
    ) -> FeatureSettings {
        FeatureSettings(
            transcription: .init(
                asrSelectionID: transcriptionASR,
                llmEnabled: false,
                llmSelectionID: .localLLM(CustomLLMModelManager.defaultModelRepo),
                prompt: AppPreferenceKey.defaultEnhancementPrompt,
                notes: .init(
                    enabled: notesEnabled,
                    titleModelSelectionID: .localLLM(CustomLLMModelManager.defaultModelRepo),
                    remindersSync: .init(enabled: remindersEnabled)
                )
            ),
            translation: .init(
                asrSelectionID: translationASR,
                modelSelectionID: .localLLM(CustomLLMModelManager.defaultModelRepo),
                targetLanguageRawValue: TranslationTargetLanguage.english.rawValue,
                prompt: AppPreferenceKey.defaultTranslationPrompt
            ),
            rewrite: .init(
                asrSelectionID: rewriteASR,
                llmSelectionID: .localLLM(CustomLLMModelManager.defaultModelRepo),
                prompt: AppPreferenceKey.defaultRewritePrompt,
                appEnhancementEnabled: false
            )
        )
    }

    func testSidebarRequirementContextPreservesFeatureSettingsSelections() {
        let settings = makeFeatureSettings(remindersEnabled: true)
        let context = SettingsPermissionRequirementResolver.sidebarRequirementContext(
            selectedEngine: .remote, featureSettings: settings
        )
        XCTAssertEqual(context.selectedEngine, .remote)
        XCTAssertEqual(context.featureSettings?.translation.asrSelectionID, settings.translation.asrSelectionID)
        XCTAssertTrue(context.featureSettings?.transcription.notes.remindersSync.enabled == true)
    }

    func testBasicAndSidebarPermissionsDoNotRequireSystemCapture() {
        for engine in [TranscriptionEngine.remote, .mlxAudio] {
            let basic = SettingsPermissionRequirementContext(selectedEngine: engine, featureSettings: nil)
            let sidebar = SettingsPermissionRequirementResolver.sidebarRequirementContext(
                selectedEngine: engine, featureSettings: makeFeatureSettings()
            )
            XCTAssertEqual(SettingsPermissionRequirementResolver.requiredPermissions(context: basic), [.microphone, .accessibility])
            XCTAssertEqual(SettingsPermissionRequirementResolver.requiredPermissions(context: sidebar), [.microphone, .accessibility])
        }
        // Removed authorizations cannot accidentally reappear as permission rows.
        XCTAssertEqual(Set(SettingsPermissionKind.allCases.map(\.rawValue)),
                       ["microphone", "speechRecognition", "accessibility", "reminders"])
    }

    func testSpeechRecognitionForSelectedEngineOrEnabledFeature() {
        let contexts = [
            SettingsPermissionRequirementContext(selectedEngine: .dictation, featureSettings: nil),
            SettingsPermissionRequirementContext(selectedEngine: .remote, featureSettings: makeFeatureSettings(transcriptionASR: .dictation)),
            SettingsPermissionRequirementContext(selectedEngine: .remote, featureSettings: makeFeatureSettings(translationASR: .dictation))
        ]
        for context in contexts {
            XCTAssertEqual(SettingsPermissionRequirementResolver.requiredPermissions(context: context),
                           [.microphone, .accessibility, .speechRecognition])
        }
    }

    func testDisabledFeaturesDoNotAddSpeechRecognitionRequirement() {
        var settings = makeFeatureSettings(translationASR: .dictation, rewriteASR: .dictation)
        settings.availability.translationEnabled = false
        settings.availability.rewriteEnabled = false
        let context = SettingsPermissionRequirementContext(selectedEngine: .remote, featureSettings: settings)
        XCTAssertEqual(SettingsPermissionRequirementResolver.requiredPermissions(context: context), [.microphone, .accessibility])
    }

    func testRemindersOnlyRequiredForEnabledNotesSync() {
        for notesEnabled in [false, true] {
            let context = SettingsPermissionRequirementContext(
                selectedEngine: .remote,
                featureSettings: makeFeatureSettings(notesEnabled: notesEnabled, remindersEnabled: true)
            )
            XCTAssertEqual(SettingsPermissionRequirementResolver.requiredPermissions(context: context),
                           notesEnabled ? [.microphone, .accessibility, .reminders] : [.microphone, .accessibility])
        }
    }
}

final class BrowserAutomationAccessTests: XCTestCase {
    func testPassiveChecksNeverRequestConsentOrExecuteScripts() {
        var requestedConsent: [Bool] = []
        let access = BrowserAutomationAccess(
            queryPermission: { _, askUser in
                requestedConsent.append(askUser)
                return OSStatus(errAEEventWouldRequireUserConsent)
            },
            executeScript: { _ in
                XCTFail("A passive status check must not execute browser scripts")
                return .init(output: nil, errorCode: nil)
            }
        )
        XCTAssertEqual(access.permissionStatus(bundleID: "custom.browser"), OSStatus(errAEEventWouldRequireUserConsent))
        XCTAssertEqual(requestedConsent, [false])
    }

    func testRequestPerformsOneAuthorizationWithoutURLProbing() {
        var requestedConsent: [Bool] = []
        let access = BrowserAutomationAccess(
            queryPermission: { _, askUser in
                requestedConsent.append(askUser)
                return noErr
            },
            executeScript: { _ in
                XCTFail("Authorization must not execute URL or application-name probes")
                return .init(output: nil, errorCode: nil)
            }
        )
        XCTAssertEqual(access.permissionStatus(bundleID: "browser", askUserIfNeeded: true), noErr)
        XCTAssertEqual(requestedConsent, [true])
    }

    func testRequestDenialReturnsWithoutProbingOrRetrying() {
        var requestedConsent: [Bool] = []
        let access = BrowserAutomationAccess(
            queryPermission: { _, askUser in
                requestedConsent.append(askUser)
                return OSStatus(errAEEventNotPermitted)
            },
            executeScript: { _ in
                XCTFail("A denied request must not execute scripts or retry authorization")
                return .init(output: nil, errorCode: nil)
            }
        )
        XCTAssertEqual(access.permissionStatus(bundleID: "browser", askUserIfNeeded: true), OSStatus(errAEEventNotPermitted))
        XCTAssertEqual(requestedConsent, [true])
    }

    func testCancelledOperationsNeverReachNativeAPI() async {
        let gate = DispatchSemaphore(value: 0)
        let access = BrowserAutomationAccess(
            queryPermission: { _, _ in
                XCTFail("A cancelled operation must not query or request permission")
                return noErr
            },
            executeScript: { _ in
                XCTFail("A cancelled operation must not execute scripts")
                return .init(output: nil, errorCode: nil)
            }
        )
        let task = Task.detached {
            gate.wait()
            return (
                access.permissionStatus(bundleID: "browser", askUserIfNeeded: true),
                access.testURLRead(bundleID: "browser", scripts: ["first", "fallback"])
            )
        }
        task.cancel()
        gate.signal()
        let (status, result) = await task.value
        XCTAssertEqual(status, OSStatus(userCanceledErr))
        XCTAssertFalse(result.success)
        XCTAssertEqual(result.lastErrorCode, Int(userCanceledErr))
    }

    func testUnapprovedURLTestNeverExecutesScriptsOrRequestsConsent() {
        for status in [errAEEventWouldRequireUserConsent, errAEEventNotPermitted, errAEPrivilegeError] {
            let access = BrowserAutomationAccess(
                queryPermission: { _, askUser in
                    XCTAssertFalse(askUser)
                    return OSStatus(status)
                },
                executeScript: { _ in
                    XCTFail("An unapproved URL test must not send Apple Events")
                    return .init(output: "unexpected", errorCode: nil)
                }
            )
            let result = access.testURLRead(bundleID: "browser", scripts: ["first", "fallback"])
            XCTAssertFalse(result.success)
            XCTAssertEqual(result.permissionDenied, status != errAEEventWouldRequireUserConsent)
            XCTAssertEqual(result.lastErrorCode, Int(status))
        }
    }

    func testAuthorizedURLReadSucceeds() {
        let access = BrowserAutomationAccess(
            queryPermission: { _, askUser in
                XCTAssertFalse(askUser)
                return noErr
            },
            executeScript: { _ in .init(output: "https://example.com", errorCode: nil) }
        )
        XCTAssertTrue(access.testURLRead(bundleID: "browser", scripts: ["first"]).success)
    }

    func testTimeoutPermissionFailureAndClosedAppStopCandidateFallback() {
        for errorCode in [errAETimeout, errAEEventNotPermitted, errAEPrivilegeError, procNotFound, userCanceledErr] {
            var executed: [String] = []
            let access = BrowserAutomationAccess(
                queryPermission: { _, _ in noErr },
                executeScript: { source in
                    executed.append(source)
                    return .init(output: nil, errorCode: Int(errorCode))
                }
            )
            let result = access.testURLRead(bundleID: "browser", scripts: ["first", "fallback"])
            XCTAssertFalse(result.success)
            XCTAssertEqual(result.lastErrorCode, Int(errorCode))
            XCTAssertEqual(executed, ["first"])
        }
    }

    func testUnsupportedURLSyntaxStillAllowsFallback() {
        var executed: [String] = []
        let access = BrowserAutomationAccess(
            queryPermission: { _, _ in noErr },
            executeScript: { source in
                executed.append(source)
                return source == "first"
                    ? .init(output: nil, errorCode: -1728)
                    : .init(output: "https://example.com", errorCode: nil)
            }
        )
        XCTAssertTrue(access.testURLRead(bundleID: "browser", scripts: ["first", "fallback"]).success)
        XCTAssertEqual(executed, ["first", "fallback"])
    }

    func testUnknownPermissionDoesNotBecomeAuthorized() {
        let access = BrowserAutomationAccess(
            queryPermission: { _, _ in OSStatus(errAETimeout) },
            executeScript: { _ in
                XCTFail("An unresolved authorization check must not execute scripts")
                return .init(output: nil, errorCode: nil)
            }
        )
        XCTAssertEqual(access.permissionStatus(bundleID: "browser", askUserIfNeeded: true), OSStatus(errAETimeout))
        let result = access.testURLRead(bundleID: "browser", scripts: ["first"])
        XCTAssertFalse(result.success)
        XCTAssertFalse(result.permissionDenied)
        XCTAssertEqual(result.lastErrorCode, Int(errAETimeout))
    }

    func testPendingNativeConsentDoesNotBlockOrDuplicateLaterOperations() async {
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let initialCall = DispatchSemaphore(value: 1)
        defer { initialCall.signal() }
        let access = BrowserAutomationAccess(
            queryPermission: { _, askUser in
                XCTAssertTrue(askUser, "Only the initial explicit request should reach the native API")
                if initialCall.wait(timeout: .now()) == .success {
                    entered.signal()
                    release.wait()
                } else {
                    XCTFail("A pending native request must not be duplicated")
                }
                return OSStatus(errAEEventNotPermitted)
            },
            executeScript: { _ in
                XCTFail("No scripts should run while native consent is pending")
                return .init(output: nil, errorCode: nil)
            }
        )
        let pending = Task.detached { access.permissionStatus(bundleID: "browser", askUserIfNeeded: true) }
        XCTAssertEqual(entered.wait(timeout: .now() + 2), .success)
        pending.cancel()
        let laterOperationsReturned = expectation(description: "Later operations return while native consent is pending")
        let later = Task.detached {
            XCTAssertEqual(access.permissionStatus(bundleID: "browser"), OSStatus(errAETimeout))
            XCTAssertEqual(access.permissionStatus(bundleID: "browser", askUserIfNeeded: true), OSStatus(errAETimeout))
            let result = access.testURLRead(bundleID: "browser", scripts: ["first", "fallback"])
            XCTAssertFalse(result.success)
            XCTAssertFalse(result.permissionDenied)
            XCTAssertEqual(result.lastErrorCode, Int(errAETimeout))
            laterOperationsReturned.fulfill()
        }
        // Bounded so the previous blocking implementation fails without hanging the test runner.
        await fulfillment(of: [laterOperationsReturned], timeout: 1)
        release.signal()
        _ = await pending.value
        await later.value
    }

    func testPermissionAndScriptOperationsNeverOverlap() {
        let counter = AutomationConcurrencyCounter()
        let access = BrowserAutomationAccess(
            queryPermission: { _, _ in
                counter.performOperation()
                return noErr
            },
            executeScript: { _ in
                counter.performOperation()
                return .init(output: "https://example.com", errorCode: nil)
            }
        )
        DispatchQueue.concurrentPerform(iterations: 12) { index in
            if index.isMultiple(of: 2) {
                _ = access.permissionStatus(bundleID: "browser", askUserIfNeeded: true)
            } else {
                _ = access.testURLRead(bundleID: "browser", scripts: ["first"])
            }
        }
        XCTAssertEqual(counter.maximumConcurrentOperations, 1)
    }
}

private final class AutomationConcurrencyCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var activeOperations = 0
    private(set) var maximumConcurrentOperations = 0

    func performOperation() {
        lock.withLock {
            activeOperations += 1
            maximumConcurrentOperations = max(maximumConcurrentOperations, activeOperations)
        }
        Thread.sleep(forTimeInterval: 0.005)
        lock.withLock { activeOperations -= 1 }
    }
}
