// VoxtLogRedactorTests.swift
// Provides Voxt Log Redactor Tests for Voxt test coverage.

import XCTest
import Logging
@testable import Voxt

final class VoxtLogRedactorTests: XCTestCase {
    func testRedactsCommonSecretShapes() {
        let text = """
        Authorization: Bearer abc.def.ghi
        apiKey=sk-test-secret
        access_token=token-value
        https://example.com/path?token=query-secret&safe=1
        """

        let redacted = VoxtLogRedactor.redact(text)

        XCTAssertFalse(redacted.contains("abc.def.ghi"))
        XCTAssertFalse(redacted.contains("sk-test-secret"))
        XCTAssertFalse(redacted.contains("token-value"))
        XCTAssertFalse(redacted.contains("query-secret"))
        XCTAssertTrue(redacted.contains("<redacted>"))
        XCTAssertTrue(redacted.contains("safe=1"))
    }

    func testStructuredMetadataRedactsSecretKeysWithoutValuePrefixes() {
        let keys = ["apiKey", "API_KEY", "access-token", "Authorization", "password", "client_secret", "Cookie"]
        for key in keys {
            let metadata: Logger.Metadata = [key: .string("opaque-fixture-secret"), "tokenCount": "42"]
            let result = VoxtLogRedactor.redactedMetadata(metadata)
            XCTAssertEqual(result?[key], .string("<redacted>"), key)
            XCTAssertEqual(result?["tokenCount"], .string("42"))
        }
    }

    func testNestedMetadataAndArraysRedactSecretContainers() {
        let metadata: Logger.Metadata = ["request": .dictionary([
            "headers": .dictionary(["X-API-Key": "opaque-secret"]),
            "items": .array([.dictionary(["refresh_token": "opaque-refresh", "status": "ok"])]),
            "credentials": .array(["not formatted as key=value"])
        ])]
        let expected: Logger.Metadata = ["request": .dictionary([
            "headers": .dictionary(["X-API-Key": "<redacted>"]),
            "items": .array([.dictionary(["refresh_token": "<redacted>", "status": "ok"])]),
            "credentials": "<redacted>"
        ])]
        XCTAssertEqual(VoxtLogRedactor.redactedMetadata(metadata), expected)
    }

    func testMetadataStillRedactsInlineSecretsUnderOrdinaryKeys() {
        let result = VoxtLogRedactor.redactedMetadata(["detail": "apiKey=fixture-secret", "model": "model-name"])
        XCTAssertEqual(result?["detail"], .string("apiKey=<redacted>"))
        XCTAssertEqual(result?["model"], .string("model-name"))
        XCTAssertNil(VoxtLogRedactor.redactedMetadata(nil))
    }

    func testPreviewRedactsAndTruncates() {
        let preview = VoxtLogRedactor.preview(
            "apiKey=secret-value " + String(repeating: "x", count: 80),
            limit: 30
        )

        XCTAssertFalse(preview.contains("secret-value"))
        XCTAssertTrue(preview.contains("<redacted>"))
        XCTAssertTrue(preview.contains("[truncated]"))
    }

    func testRedactsHomeDirectory() {
        let home = NSHomeDirectory()
        let redacted = VoxtLogRedactor.redact("path=\(home)/Documents/private.txt")

        XCTAssertFalse(redacted.contains(home))
        XCTAssertTrue(redacted.contains("~/Documents/private.txt"))
    }

    func testSensitivePrivacyOmitsEntireUserContent() {
        let redacted = VoxtLogRedactor.redact(
            "private transcript and model response",
            privacy: .sensitive
        )

        XCTAssertEqual(redacted, "<redacted>")
        XCTAssertFalse(redacted.contains("transcript"))
    }

    func testLLMContentLoggerDoesNotEvaluateSensitiveMessage() {
        let defaults = UserDefaults.standard
        let previousValue = defaults.object(forKey: AppPreferenceKey.llmDebugLoggingEnabled)
        defaults.set(true, forKey: AppPreferenceKey.llmDebugLoggingEnabled)
        defer {
            if let previousValue {
                defaults.set(previousValue, forKey: AppPreferenceKey.llmDebugLoggingEnabled)
            } else {
                defaults.removeObject(forKey: AppPreferenceKey.llmDebugLoggingEnabled)
            }
        }

        var evaluated = false
        VoxtLog.llm({
            evaluated = true
            return "private prompt and response"
        }())

        XCTAssertFalse(evaluated)
    }

    func testLLMDebugLoggerEvaluatesSafeMetadataWhenEnabled() {
        let defaults = UserDefaults.standard
        let previousValue = defaults.object(forKey: AppPreferenceKey.llmDebugLoggingEnabled)
        defaults.set(true, forKey: AppPreferenceKey.llmDebugLoggingEnabled)
        defer {
            if let previousValue {
                defaults.set(previousValue, forKey: AppPreferenceKey.llmDebugLoggingEnabled)
            } else {
                defaults.removeObject(forKey: AppPreferenceKey.llmDebugLoggingEnabled)
            }
        }

        var evaluated = false
        VoxtLog.llmDebug({
            evaluated = true
            return "model=test, inputChars=42"
        }())

        XCTAssertTrue(evaluated)
    }
}
