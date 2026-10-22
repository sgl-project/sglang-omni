import Foundation
import Testing
@testable import VoxtMobileKit

/// Settings › About › Source code: the commit a release was built from (the release build writes
/// VoxtSourceRevision), linked in the repository the build names (VoxtSourceURL);
/// anything else is a development build.
@Suite struct SourceRevisionTests {
    private let full = "cdfb0e8a1b2c3d4e5f60718293a4b5c6d7e8f901"

    @Test func aReleaseShowsTheShortHashAndLinksTheCommitInItsRepository() throws {
        let revision = try #require(SourceRevision(info: ["VoxtSourceRevision": full, "VoxtSourceURL": "github.com/example/voxt"]))
        #expect(revision.short == "cdfb0e8")
        #expect(revision.url == URL(string: "https://github.com/example/voxt/commit/\(full)"))
    }

    /// xcconfig reads `//` as a comment, so the build setting has no scheme; one with it works too.
    @Test func theRepositoryIsHTTPSWithOrWithoutAScheme() {
        let expected = URL(string: "https://github.com/example/voxt")
        #expect(SourceRevision.repository(info: ["VoxtSourceURL": "github.com/example/voxt"]) == expected)
        #expect(SourceRevision.repository(info: ["VoxtSourceURL": "https://github.com/example/voxt/"]) == expected)
    }

    @Test func withoutARepositoryTheBuildPointsUpstream() {
        let upstream = URL(string: "https://github.com/sgl-project/sglang-omni")
        #expect(SourceRevision.repository(info: nil) == upstream)
        #expect(SourceRevision.repository(info: ["VoxtSourceURL": ""]) == upstream)
        #expect(SourceRevision.repository(info: ["VoxtSourceURL": "$(VOXT_SOURCE_URL)"]) == upstream)
    }

    @Test func anythingElseIsADevelopmentBuild() {
        #expect(SourceRevision(info: nil) == nil)
        #expect(SourceRevision(info: [:]) == nil)
        #expect(SourceRevision(info: ["VoxtSourceRevision": ""]) == nil)
        #expect(SourceRevision(info: ["VoxtSourceRevision": "$(VOXT_SOURCE_REVISION)"]) == nil, "the setting was never filled in")
        #expect(SourceRevision(info: ["VoxtSourceRevision": "main"]) == nil, "not a commit hash")
        #expect(SourceRevision(info: ["VoxtSourceRevision": String(full.prefix(12))]) == nil, "the link needs the full hash")
    }
}
