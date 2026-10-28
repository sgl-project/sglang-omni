// PrequantizedModelLoadingTests.swift

import MLX
import MLXLMCommon
import MLXNN
import XCTest
@testable import Voxt

@MainActor
final class PrequantizedModelLoadingTests: XCTestCase {
    private final class TestModel: Module {
        @ModuleInfo(key: "linear") var linear = Linear(32, 32, bias: false)
        @ModuleInfo(key: "embedding") var embedding = Embedding(
            embeddingCount: 32,
            dimensions: 32
        )
    }

    private final class TestSwitchModel: Module {
        @ModuleInfo(key: "switch") var switchLinear = SwitchLinear(
            inputDims: 32,
            outputDims: 32,
            numExperts: 2,
            bias: false
        )
    }

    func testReplaceLayersInstallsCheckpointArraysDirectly() throws {
        let model = TestModel()
        let linearWeight = MLXArray(Array(repeating: Float(0.25), count: 1024), [32, 32])
        let embeddingWeight = MLXArray(Array(repeating: Float(-0.5), count: 1024), [32, 32])
        let (linearQuantized, linearScales, linearBiases) = quantized(
            linearWeight,
            groupSize: 32,
            bits: 4
        )
        let (embeddingQuantized, embeddingScales, embeddingBiases) = quantized(
            embeddingWeight,
            groupSize: 32,
            bits: 4
        )
        let unwrappedLinearBiases = try XCTUnwrap(linearBiases)
        let unwrappedEmbeddingBiases = try XCTUnwrap(embeddingBiases)
        eval(
            linearQuantized,
            linearScales,
            unwrappedLinearBiases,
            embeddingQuantized,
            embeddingScales,
            unwrappedEmbeddingBiases
        )

        let weights = [
            "linear.weight": linearQuantized,
            "linear.scales": linearScales,
            "linear.biases": unwrappedLinearBiases,
            "embedding.weight": embeddingQuantized,
            "embedding.scales": embeddingScales,
            "embedding.biases": unwrappedEmbeddingBiases,
        ]
        let summary = PrequantizedModelLoading.replaceLayers(
            in: model,
            weights: weights
        ) { _ in
            (groupSize: 32, bits: 4, mode: .affine)
        }

        XCTAssertEqual(
            summary,
            PrequantizedModelLoading.ReplacementSummary(direct: 2, fallback: 0)
        )
        XCTAssertTrue(model.linear is QuantizedLinear)
        XCTAssertTrue(model.embedding is LoadedQuantizedEmbedding)
        XCTAssertEqual(model.linear.weight.asArray(UInt32.self), linearQuantized.asArray(UInt32.self))
        XCTAssertEqual(
            model.embedding.weight.asArray(UInt32.self),
            embeddingQuantized.asArray(UInt32.self)
        )
    }

    func testReplacementPreservesLinearAndEmbeddingInference() throws {
        let model = TestModel()
        let linearWeight = MLXArray(Array(0..<1024).map { Float($0) / 1024 }, [32, 32])
        let embeddingWeight = MLXArray(Array(0..<1024).map { Float($0) / 512 }, [32, 32])
        let (linearQuantized, linearScales, linearBiases) = quantized(
            linearWeight,
            groupSize: 32,
            bits: 4
        )
        let (embeddingQuantized, embeddingScales, embeddingBiases) = quantized(
            embeddingWeight,
            groupSize: 32,
            bits: 4
        )
        let unwrappedLinearBiases = try XCTUnwrap(linearBiases)
        let unwrappedEmbeddingBiases = try XCTUnwrap(embeddingBiases)
        let weights = [
            "linear.weight": linearQuantized,
            "linear.scales": linearScales,
            "linear.biases": unwrappedLinearBiases,
            "embedding.weight": embeddingQuantized,
            "embedding.scales": embeddingScales,
            "embedding.biases": unwrappedEmbeddingBiases,
        ]
        PrequantizedModelLoading.replaceLayers(in: model, weights: weights) { _ in
            (groupSize: 32, bits: 4, mode: .affine)
        }

        let input = MLXArray(Array(repeating: Float(1), count: 32), [1, 32])
        let expectedLinear = quantizedMM(
            input,
            linearQuantized,
            scales: linearScales,
            biases: unwrappedLinearBiases,
            transpose: true,
            groupSize: 32,
            bits: 4
        )
        let actualLinear = model.linear(input)
        let tokenIDs = MLXArray([1, 3, 6])
        let expectedEmbedding = dequantized(
            embeddingQuantized[tokenIDs],
            scales: embeddingScales[tokenIDs],
            biases: unwrappedEmbeddingBiases[tokenIDs],
            groupSize: 32,
            bits: 4
        )
        let actualEmbedding = model.embedding(tokenIDs)
        eval(expectedLinear, actualLinear, expectedEmbedding, actualEmbedding)

        XCTAssertTrue(allClose(actualLinear, expectedLinear).item(Bool.self))
        XCTAssertTrue(allClose(actualEmbedding, expectedEmbedding).item(Bool.self))
    }

    func testLayerWithoutCheckpointScalesIsNotReplaced() {
        let model = TestModel()
        let summary = PrequantizedModelLoading.replaceLayers(
            in: model,
            weights: ["linear.weight": MLXArray.zeros([32, 16], type: UInt32.self)]
        ) { _ in
            (groupSize: 32, bits: 4, mode: .affine)
        }

        XCTAssertEqual(
            summary,
            PrequantizedModelLoading.ReplacementSummary(direct: 0, fallback: 0)
        )
        XCTAssertFalse(model.linear is QuantizedLinear)
        XCTAssertFalse(model.embedding is LoadedQuantizedEmbedding)
    }

    func testReplacementPreservesSwitchLinearInferenceWithoutFallbackQuantization() throws {
        let model = TestSwitchModel()
        let sourceWeight = MLXArray(
            Array(0..<(2 * 32 * 32)).map { Float($0 % 127) / 127 },
            [2, 32, 32]
        )
        let (weight, scales, biases) = quantized(
            sourceWeight,
            groupSize: 32,
            bits: 4
        )
        let unwrappedBiases = try XCTUnwrap(biases)
        eval(weight, scales, unwrappedBiases)

        let summary = PrequantizedModelLoading.replaceLayers(
            in: model,
            weights: [
                "switch.weight": weight,
                "switch.scales": scales,
                "switch.biases": unwrappedBiases,
            ]
        ) { _ in
            (groupSize: 32, bits: 4, mode: .affine)
        }

        XCTAssertEqual(
            summary,
            PrequantizedModelLoading.ReplacementSummary(direct: 1, fallback: 0)
        )
        let loaded = try XCTUnwrap(model.switchLinear as? QuantizedSwitchLinear)
        let input = MLXArray(Array(repeating: Float(1), count: 64), [2, 1, 32])
        let indices = MLXArray([0, 1])
        let expected = gatherQuantizedMM(
            input,
            weight,
            scales: scales,
            biases: unwrappedBiases,
            rhsIndices: indices,
            transpose: true,
            groupSize: 32,
            bits: 4
        )
        let actual = loaded(input, indices)
        eval(expected, actual)

        XCTAssertTrue(allClose(actual, expected).item(Bool.self))
    }
}
