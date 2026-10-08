import XCTest
@testable import Voxt

final class ModelLogoKeyTests: XCTestCase {
    func testNemotronASRUsesNvidiaLogo() {
        XCTAssertEqual(
            ModelLogoKey.resolve(title: "Nemotron", engine: "MLX Audio"),
            .nvidia
        )
    }

    func testCanaryASRUsesNvidiaLogo() {
        XCTAssertEqual(
            ModelLogoKey.resolve(title: "Canary", engine: "MLX Audio"),
            .nvidia
        )
    }

    func testLFMUsesLiquidLogo() {
        XCTAssertEqual(
            ModelLogoKey.resolve(title: "LFM2 1.2B (4bit)", engine: "Local LLM"),
            .liquid
        )
    }



    func testCompactModelBadgeTitleKeepsFamilyWithoutSeries() {
        XCTAssertEqual(FeatureModelCatalogBuilder.compactModelBadgeTitle(from: "Whisper Large v3 Turbo"), "Whisper")
        XCTAssertEqual(FeatureModelCatalogBuilder.compactModelBadgeTitle(from: "Qwen3 1.7B (4bit)"), "Qwen")
        XCTAssertEqual(FeatureModelCatalogBuilder.compactModelBadgeTitle(from: "Qwen3 VL 4B Instruct (4bit)"), "Qwen")
        XCTAssertEqual(FeatureModelCatalogBuilder.compactModelBadgeTitle(from: "Parakeet TDT 0.6B v2"), "Parakeet")
        XCTAssertEqual(FeatureModelCatalogBuilder.compactModelBadgeTitle(from: "FunASR Nano"), "FunASR")
        XCTAssertEqual(FeatureModelCatalogBuilder.compactModelBadgeTitle(from: "Hy-MT2 1.8B (Q4_K_M)"), "Hy-MT2")
        XCTAssertEqual(FeatureModelCatalogBuilder.compactModelBadgeTitle(from: "Meta Llama 3.1 8B Instruct (4bit)"), "Llama")
        XCTAssertEqual(FeatureModelCatalogBuilder.compactModelBadgeTitle(from: "SenseVoice"), "SenseVoice")
        XCTAssertEqual(FeatureModelCatalogBuilder.compactModelBadgeTitle(from: "GLM Nano (4bit)"), "GLM")
    }

    func testGoogleGeminiASRUsesGoogleLogo() {
        XCTAssertEqual(
            ModelLogoKey.resolve(title: RemoteASRProvider.googleGeminiASR.title, engine: "Remote ASR"),
            .google
        )
        XCTAssertEqual(
            ModelLogoKey.resolve(title: "Google", engine: "Remote LLM"),
            .google
        )
    }
}
