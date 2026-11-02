// TestAssertions.swift
// Provides Test Assertions for test support.

import XCTest

func XCTAssertJSONRoundTrip<T: Codable & Equatable>(
    _ value: T,
    file: StaticString = #filePath,
    line: UInt = #line
) throws {
    let data = try JSONEncoder().encode(value)
    let decoded = try JSONDecoder().decode(T.self, from: data)
    XCTAssertEqual(decoded, value, file: file, line: line)
}

func XCTAssertContains(
    _ text: String,
    _ substring: String,
    file: StaticString = #filePath,
    line: UInt = #line
) {
    XCTAssertTrue(text.contains(substring), "Expected substring not found: \(substring)", file: file, line: line)
}

