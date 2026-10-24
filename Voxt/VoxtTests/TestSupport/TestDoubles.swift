// TestDoubles.swift
// Provides Test Doubles for test support.

import Foundation
import XCTest
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif
@testable import Voxt

enum TestDoubles {
    static func canonicalTemporaryDirectory() throws -> URL {
        try canonicalFileURL(for: FileManager.default.temporaryDirectory)
    }

    static func canonicalFileURL(for url: URL) throws -> URL {
        guard let resolved = realpath(url.path, nil) else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno))
        }
        defer { free(resolved) }
        return URL(fileURLWithPath: String(cString: resolved), isDirectory: true)
    }

    static func makeUserDefaults(testName: String = UUID().uuidString) -> UserDefaults {
        let suiteName = "VoxtTests.\(testName)"
        let defaults = UserDefaults(suiteName: suiteName)!
        defaults.removePersistentDomain(forName: suiteName)
        return defaults
    }
}

final class TemporaryDirectory {
    let url: URL

    init() throws {
        let baseURL = FileManager.default.temporaryDirectory
        let directoryURL = baseURL.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        url = directoryURL
    }

    deinit {
        try? FileManager.default.removeItem(at: url)
    }
}
