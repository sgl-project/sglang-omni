// AppBranchURLPatternServiceTests.swift
// Provides App Branch URLPattern Service Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class AppBranchURLPatternServiceTests: XCTestCase {
    func testCanonicalizedPatternNormalizesSchemeAndWildcard() {
        XCTAssertEqual(
            AppBranchURLPatternService.canonicalizedPattern("https://Example.com/docs"),
            "example.com/docs"
        )
        XCTAssertEqual(
            AppBranchURLPatternService.canonicalizedPattern("example.com"),
            "example.com/*"
        )
        XCTAssertEqual(
            AppBranchURLPatternService.canonicalizedPattern("example.com/path/"),
            "example.com/path/*"
        )
    }

    func testWildcardValidationRejectsInvalidPatterns() {
        XCTAssertTrue(AppBranchURLPatternService.isValidWildcardURLPattern("example.com/*"))
        XCTAssertFalse(AppBranchURLPatternService.isValidWildcardURLPattern("example"))
        XCTAssertFalse(AppBranchURLPatternService.isValidWildcardURLPattern("example com/*"))
    }

    func testNormalizedURLForMatchingLowercasesHostAndPath() {
        XCTAssertEqual(
            AppBranchURLPatternService.normalizedURLForMatching("HTTPS://Example.COM/Docs/Page"),
            "example.com/docs/page"
        )
    }

    func testFirstPromptAndGroupMatchUseWildcardPatterns() {
        let docsID = UUID()
        let group = TestFactories.makeAppBranchGroup(
            name: "Docs",
            prompt: "Use docs prompt",
            urlPatternIDs: [docsID]
        )
        let urls = [docsID: "example.com/docs/*"]

        let promptMatch = AppBranchURLPatternService.firstPromptMatch(
            groups: [group],
            urlsByID: urls,
            normalizedURL: "example.com/docs/guide"
        )
        let groupMatch = AppBranchURLPatternService.firstGroupMatch(
            groups: [group],
            urlsByID: urls,
            normalizedURL: "example.com/docs/guide"
        )

        XCTAssertEqual(promptMatch?.groupName, "Docs")
        XCTAssertEqual(promptMatch?.pattern, "example.com/docs/*")
        XCTAssertEqual(groupMatch?.groupID, group.id)
    }
}
