// OnboardingPreferenceManagerTests.swift
// Provides Onboarding Preference Manager Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class OnboardingPreferenceManagerTests: XCTestCase {
    func testResolvedCompletionStateDefaultsToFalseForFreshInstall() throws {
        let (defaults, suiteName) = makeIsolatedDefaults()
        let directory = try TemporaryDirectory()
        let fileManager = TestAppSupportFileManager(applicationSupportDirectory: directory.url)

        let completed = OnboardingPreferenceManager.resolvedCompletionState(
            defaults: defaults,
            fileManager: fileManager,
            bundleIdentifier: suiteName
        )

        XCTAssertFalse(completed)
        XCTAssertEqual(defaults.object(forKey: AppPreferenceKey.onboardingCompleted) as? Bool, false)
    }

    func testResolvedCompletionStateTreatsExistingPersistentDomainAsCompleted() throws {
        let (defaults, suiteName) = makeIsolatedDefaults()
        let directory = try TemporaryDirectory()
        let fileManager = TestAppSupportFileManager(applicationSupportDirectory: directory.url)
        defaults.setPersistentDomain(
            [AppPreferenceKey.translationTargetLanguage: TranslationTargetLanguage.english.rawValue],
            forName: suiteName
        )

        let completed = OnboardingPreferenceManager.resolvedCompletionState(
            defaults: defaults,
            fileManager: fileManager,
            bundleIdentifier: suiteName
        )

        XCTAssertTrue(completed)
        XCTAssertEqual(defaults.object(forKey: AppPreferenceKey.onboardingCompleted) as? Bool, true)
    }

    func testResolvedCompletionStateTreatsExistingAppSupportDirectoryAsCompleted() throws {
        let (defaults, suiteName) = makeIsolatedDefaults()
        let directory = try TemporaryDirectory()
        let appSupportDirectory = directory.url.appendingPathComponent("Voxt", isDirectory: true)
        try FileManager.default.createDirectory(at: appSupportDirectory, withIntermediateDirectories: true)
        let fileManager = TestAppSupportFileManager(applicationSupportDirectory: directory.url)

        let completed = OnboardingPreferenceManager.resolvedCompletionState(
            defaults: defaults,
            fileManager: fileManager,
            bundleIdentifier: suiteName
        )

        XCTAssertTrue(completed)
        XCTAssertEqual(defaults.object(forKey: AppPreferenceKey.onboardingCompleted) as? Bool, true)
    }

    func testMarkCompletedClearsSavedStep() {
        let defaults = TestDoubles.makeUserDefaults()
        OnboardingPreferenceManager.saveLastStep(.finish, defaults: defaults)

        OnboardingPreferenceManager.markCompleted(defaults: defaults)

        XCTAssertEqual(defaults.object(forKey: AppPreferenceKey.onboardingCompleted) as? Bool, true)
        XCTAssertNil(defaults.string(forKey: AppPreferenceKey.onboardingLastStepID))
    }

    func testGuideStepPersistsAndIsClearedWhenCompleted() {
        let defaults = TestDoubles.makeUserDefaults()
        OnboardingPreferenceManager.saveLastGuideStep(.models, defaults: defaults)

        XCTAssertEqual(OnboardingPreferenceManager.savedLastGuideStep(defaults: defaults), .models)

        OnboardingPreferenceManager.markCompleted(defaults: defaults)

        XCTAssertEqual(defaults.object(forKey: AppPreferenceKey.onboardingCompleted) as? Bool, true)
        XCTAssertNil(OnboardingPreferenceManager.savedLastGuideStep(defaults: defaults))
    }

    func testRemovedMicrophoneGuideStepResumesAtPermissions() {
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set("microphone", forKey: AppPreferenceKey.onboardingLastStepID)

        XCTAssertEqual(OnboardingPreferenceManager.savedLastGuideStep(defaults: defaults), .permissions)
        XCTAssertEqual(OnboardingGuideStep.permissions.next, .models)
    }

    private func makeIsolatedDefaults() -> (defaults: UserDefaults, suiteName: String) {
        let suiteName = "VoxtTests.Onboarding.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        return (defaults, suiteName)
    }
}

final class ModelStorageDirectoryManagerResolutionTests: XCTestCase {
    override func setUp() {
        super.setUp()
        ModelStorageDirectoryManager.resetForTesting()
    }

    override func tearDown() {
        ModelStorageDirectoryManager.resetForTesting()
        super.tearDown()
    }

    func testAutomaticResolutionUsesNewWriteRootAndLegacyReadFallback() {
        let defaults = TestDoubles.makeUserDefaults(testName: "ModelStorageDirectoryManagerTests.automaticResolution")
        defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootPath)
        defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)

        let resolution = ModelStorageDirectoryManager.resolvedRootResolution(defaults: defaults)

        XCTAssertEqual(resolution.writeRootURL.standardizedFileURL.path, ModelStorageDirectoryManager.defaultRootURL.standardizedFileURL.path)
        XCTAssertEqual(
            resolution.readableRootURLs.map(\.standardizedFileURL.path),
            [
                ModelStorageDirectoryManager.defaultRootURL.standardizedFileURL.path,
                ModelStorageDirectoryManager.legacyDefaultRootURL.standardizedFileURL.path,
            ]
        )
    }

    func testStoredPathWithoutBookmarkRequiresAuthorization() throws {
        let defaults = TestDoubles.makeUserDefaults(testName: "ModelStorageDirectoryManagerTests.storedPath")
        let directory = try TemporaryDirectory()
        let customRoot = directory.url.appendingPathComponent("custom-model-root", isDirectory: true)
        defaults.set(customRoot.path, forKey: AppPreferenceKey.modelStorageRootPath)
        defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)

        let resolution = ModelStorageDirectoryManager.resolvedRootResolution(defaults: defaults)

        XCTAssertEqual(resolution.writeRootURL.standardizedFileURL.path, customRoot.standardizedFileURL.path)
        XCTAssertTrue(resolution.readableRootURLs.isEmpty)
        XCTAssertEqual(
            resolution.accessIssue,
            .authorizationRequired(path: customRoot.standardizedFileURL.path)
        )
        XCTAssertThrowsError(try ModelStorageDirectoryManager.requireWriteRootURL(defaults: defaults)) { error in
            XCTAssertEqual(
                error as? ModelStorageDirectoryManager.AccessIssue,
                .authorizationRequired(path: customRoot.standardizedFileURL.path)
            )
        }
    }

    func testDefaultStoredPathDoesNotRequireAuthorizationWithoutValidBookmark() {
        let defaults = TestDoubles.makeUserDefaults(testName: "ModelStorageDirectoryManagerTests.defaultStoredPath")
        defaults.set(
            ModelStorageDirectoryManager.defaultRootURL.standardizedFileURL.path,
            forKey: AppPreferenceKey.modelStorageRootPath
        )

        for bookmarkData in [nil, Data([0x00, 0x01, 0x02])] as [Data?] {
            if let bookmarkData {
                defaults.set(bookmarkData, forKey: AppPreferenceKey.modelStorageRootBookmark)
            } else {
                defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)
            }

            let resolution = ModelStorageDirectoryManager.resolvedRootResolution(defaults: defaults)

            XCTAssertEqual(
                resolution.writeRootURL.standardizedFileURL.path,
                ModelStorageDirectoryManager.defaultRootURL.standardizedFileURL.path
            )
            XCTAssertEqual(
                resolution.readableRootURLs.map(\.standardizedFileURL.path),
                [
                    ModelStorageDirectoryManager.defaultRootURL.standardizedFileURL.path,
                    ModelStorageDirectoryManager.legacyDefaultRootURL.standardizedFileURL.path,
                ]
            )
            XCTAssertNil(resolution.accessIssue)
        }
    }

    func testInvalidBookmarkRequiresReauthorization() throws {
        let defaults = TestDoubles.makeUserDefaults(testName: "ModelStorageDirectoryManagerTests.invalidBookmark")
        let directory = try TemporaryDirectory()
        let customRoot = directory.url.appendingPathComponent("custom-model-root", isDirectory: true)
        defaults.set(customRoot.path, forKey: AppPreferenceKey.modelStorageRootPath)
        defaults.set(Data([0x00, 0x01, 0x02]), forKey: AppPreferenceKey.modelStorageRootBookmark)

        let resolution = ModelStorageDirectoryManager.resolvedRootResolution(defaults: defaults)

        XCTAssertEqual(resolution.writeRootURL.standardizedFileURL.path, customRoot.standardizedFileURL.path)
        XCTAssertTrue(resolution.readableRootURLs.isEmpty)
        XCTAssertEqual(
            resolution.accessIssue,
            .invalidBookmark(path: customRoot.standardizedFileURL.path)
        )
    }

    func testLegacyStoredPathWithoutBookmarkUsesAutomaticResolution() {
        let defaults = TestDoubles.makeUserDefaults(testName: "ModelStorageDirectoryManagerTests.legacyStoredPathAutomaticResolution")
        defaults.set(
            ModelStorageDirectoryManager.legacyDefaultRootURL.standardizedFileURL.path,
            forKey: AppPreferenceKey.modelStorageRootPath
        )
        defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)

        let resolution = ModelStorageDirectoryManager.resolvedRootResolution(defaults: defaults)

        XCTAssertEqual(resolution.writeRootURL.standardizedFileURL.path, ModelStorageDirectoryManager.defaultRootURL.standardizedFileURL.path)
        XCTAssertEqual(
            resolution.readableRootURLs.map(\.standardizedFileURL.path),
            [
                ModelStorageDirectoryManager.defaultRootURL.standardizedFileURL.path,
                ModelStorageDirectoryManager.legacyDefaultRootURL.standardizedFileURL.path,
            ]
        )
    }

    func testResolvedRootURLReturnsWriteRoot() throws {
        let defaults = TestDoubles.makeUserDefaults(testName: "ModelStorageDirectoryManagerTests.resolvedRootURL")
        let directory = try TemporaryDirectory()
        let customRoot = directory.url.appendingPathComponent("another-root", isDirectory: true)
        defaults.set(customRoot.path, forKey: AppPreferenceKey.modelStorageRootPath)
        defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)

        let resolvedRoot = ModelStorageDirectoryManager.resolvedRootURL(defaults: defaults)

        XCTAssertEqual(resolvedRoot.standardizedFileURL.path, customRoot.standardizedFileURL.path)
    }

    func testResolvedDerivedRootURLTracksWriteRoot() throws {
        let defaults = TestDoubles.makeUserDefaults(testName: "ModelStorageDirectoryManagerTests.derivedRootURL")
        let directory = try TemporaryDirectory()
        let customRoot = directory.url.appendingPathComponent("another-root", isDirectory: true)
        defaults.set(customRoot.path, forKey: AppPreferenceKey.modelStorageRootPath)
        defaults.removeObject(forKey: AppPreferenceKey.modelStorageRootBookmark)

        let derivedRoot = ModelStorageDirectoryManager.resolvedDerivedRootURL(defaults: defaults)

        XCTAssertEqual(
            derivedRoot.standardizedFileURL.path,
            customRoot
                .appendingPathComponent(".derived-model-artifacts", isDirectory: true)
                .standardizedFileURL.path
        )
    }
}

private final class TestAppSupportFileManager: FileManager {
    private let applicationSupportDirectory: URL

    init(applicationSupportDirectory: URL) {
        self.applicationSupportDirectory = applicationSupportDirectory
        super.init()
    }

    override func url(
        for directory: SearchPathDirectory,
        in domain: SearchPathDomainMask,
        appropriateFor url: URL?,
        create shouldCreate: Bool
    ) throws -> URL {
        if directory == .applicationSupportDirectory, domain == .userDomainMask {
            return applicationSupportDirectory
        }
        return try super.url(
            for: directory,
            in: domain,
            appropriateFor: url,
            create: shouldCreate
        )
    }
}
