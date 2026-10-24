// RemoteProviderConnectivityTestEndpointsTests.swift
// Provides Remote Provider Connectivity Test Endpoints Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class RemoteProviderConnectivityTestEndpointsTests: XCTestCase {
    func testResolvedASRTranscriptionEndpointAppendsDefaultPathForBaseHost() {
        XCTAssertEqual(
            RemoteProviderConnectivityTestEndpoints.resolvedASRTranscriptionEndpoint(
                endpoint: "https://api.openai.com",
                defaultValue: "https://api.openai.com/v1/audio/transcriptions"
            ),
            "https://api.openai.com/v1/audio/transcriptions"
        )
    }

    func testResolvedGLMASRTranscriptionEndpointRemapsModelsPath() {
        XCTAssertEqual(
            RemoteProviderConnectivityTestEndpoints.resolvedGLMASRTranscriptionEndpoint(
                endpoint: "https://open.bigmodel.cn/api/paas/v4/models",
                defaultValue: "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"
            ),
            "https://open.bigmodel.cn/api/paas/v4/audio/transcriptions"
        )
    }

    func testResolvedAliyunRealtimeWebSocketEndpointRemapsCompatibleChatPath() {
        XCTAssertEqual(
            RemoteProviderConnectivityTestEndpoints.resolvedAliyunASRRealtimeWebSocketEndpoint(
                endpoint: "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                defaultValue: "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
            ),
            "https://dashscope.aliyuncs.com/compatible-mode/v1/api-ws/v1/inference"
        )
    }

    func testResolvedAliyunQwenRealtimeEndpointAddsMissingModelQuery() {
        XCTAssertEqual(
            RemoteProviderConnectivityTestEndpoints.resolvedAliyunASRQwenRealtimeWebSocketEndpoint(
                endpoint: "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
                model: "qwen3-asr-flash-realtime"
            ),
            "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-asr-flash-realtime"
        )
    }

    func testResolvedAliyunOmniRealtimeEndpointAddsMissingModelQuery() {
        XCTAssertEqual(
            RemoteProviderConnectivityTestEndpoints.resolvedAliyunASRQwenRealtimeWebSocketEndpoint(
                endpoint: "wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
                model: "qwen3.5-omni-flash-realtime"
            ),
            "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3.5-omni-flash-realtime"
        )
    }
}
