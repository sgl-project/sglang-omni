// AliyunMeetingASRClientTests.swift
// Provides Aliyun Meeting ASRClient Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class AliyunMeetingASRClientTests: XCTestCase {
    func testNormalizedTranscriptionURLUpgradesAliyunHTTPToHTTPS() {
        let normalized = AliyunMeetingASRClient.normalizedTranscriptionURL(
            "http://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/path/result.json?foo=bar"
        )

        XCTAssertEqual(
            normalized,
            "https://dashscope-result-bj.oss-cn-beijing.aliyuncs.com/path/result.json?foo=bar"
        )
    }

    func testExtractTranscriptionURLPrefersOutputResult() {
        let object: [String: Any] = [
            "output": [
                "result": [
                    "transcription_url": "https://example.com/result.json"
                ]
            ]
        ]

        XCTAssertEqual(
            AliyunMeetingASRClient.extractTranscriptionURL(from: object),
            "https://example.com/result.json"
        )
    }

    func testExtractResultTextCollectsNestedTranscriptText() {
        let object: [String: Any] = [
            "transcripts": [
                ["text": "第一句"],
                ["text": "第二句"]
            ]
        ]

        XCTAssertEqual(
            AliyunMeetingASRClient.extractResultText(from: object),
            "第一句\n第二句"
        )
    }

    func testExtractTextReadsCompatibleModeResponse() {
        let object: [String: Any] = [
            "choices": [
                [
                    "message": [
                        "content": "测试文本"
                    ]
                ]
            ]
        ]

        XCTAssertEqual(AliyunMeetingASRClient.extractText(from: object), "测试文本")
    }

    func testIsNoValidFragmentRecognizesAliyunStatusCode() {
        let object: [String: Any] = [
            "output": [
                "code": "SUCCESS_WITH_NO_VALID_FRAGMENT"
            ]
        ]

        XCTAssertTrue(AliyunMeetingASRClient.isNoValidFragment(object))
    }
}
