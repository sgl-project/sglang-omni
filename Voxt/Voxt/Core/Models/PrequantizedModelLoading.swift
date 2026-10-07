// PrequantizedModelLoading.swift
// Installs checkpoint quantization tensors directly instead of quantizing throwaway weights.

import MLX
import MLXLMCommon
import MLXNN

nonisolated final class LoadedQuantizedEmbedding: Embedding, Quantized {
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode
    let scales: MLXArray
    let biases: MLXArray?

    override var shape: (Int, Int) {
        let packedShape = weight.shape2
        return (packedShape.0, packedShape.1 * 32 / bits)
    }

    init(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.scales = scales
        self.biases = biases
        super.init(weight: weight)
        freeze()
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let inputShape = x.shape
        let flattened = x.flattened()
        let output = dequantized(
            weight[flattened],
            scales: scales[flattened],
            biases: biases?[flattened],
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        return output.reshaped(inputShape + [-1])
    }

    override func asLinear(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }
}

nonisolated enum PrequantizedModelLoading {
    typealias Quantization = (groupSize: Int, bits: Int, mode: QuantizationMode)

    struct ReplacementSummary: Equatable {
        let direct: Int
        let fallback: Int
    }

    @discardableResult
    static func replaceLayers(
        in model: Module,
        weights: [String: MLXArray],
        quantizationForPath: (String) -> Quantization?
    ) -> ReplacementSummary {
        var directCount = 0
        var fallbackCount = 0
        let updates = model.leafModules().flattened().compactMap { path, module in
            guard let weight = weights["\(path).weight"],
                  let scales = weights["\(path).scales"],
                  let quantization = quantizationForPath(path)
            else { return nil as (String, Module)? }

            let biases = weights["\(path).biases"]
            if module is Embedding {
                directCount += 1
                return (
                    path,
                    LoadedQuantizedEmbedding(
                        weight: weight,
                        scales: scales,
                        biases: biases,
                        groupSize: quantization.groupSize,
                        bits: quantization.bits,
                        mode: quantization.mode
                    )
                )
            }

            if module is SwitchLinear {
                directCount += 1
                return (
                    path,
                    makeLoadedQuantizedSwitchLinear(
                        weight: weight,
                        bias: weights["\(path).bias"],
                        scales: scales,
                        biases: biases,
                        groupSize: quantization.groupSize,
                        bits: quantization.bits,
                        mode: quantization.mode
                    )
                )
            }

            if let linear = module as? Linear {
                let loadedBias = weights["\(path).bias"] ?? linear.bias
                let quantized = QuantizedLinear(
                    weight: weight,
                    bias: loadedBias,
                    scales: scales,
                    biases: biases,
                    groupSize: quantization.groupSize,
                    bits: quantization.bits,
                    mode: quantization.mode
                )
                quantized.freeze()
                directCount += 1
                return (path, quantized)
            }

            guard let quantized = quantizeSingle(
                layer: module,
                groupSize: quantization.groupSize,
                bits: quantization.bits,
                mode: quantization.mode
            ) else { return nil }
            fallbackCount += 1
            return (path, quantized)
        }

        model.update(modules: ModuleChildren.unflattened(updates))
        return ReplacementSummary(direct: directCount, fallback: fallbackCount)
    }

    private static func makeLoadedQuantizedSwitchLinear(
        weight: MLXArray,
        bias: MLXArray?,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) -> QuantizedSwitchLinear {
        precondition(weight.ndim == 3, "A quantized SwitchLinear weight must have three dimensions.")

        // MLXLMCommon does not expose a checkpoint-array initializer for
        // QuantizedSwitchLinear. Quantize one minimum-sized tensor so the concrete
        // replacement type remains upstream's implementation, then overwrite its
        // parameters before it enters the model. This bounds the throwaway graph to
        // one quantization group instead of the complete MoE expert matrices.
        let placeholder = SwitchLinear(
            inputDims: weight.shape[2] * 32 / bits,
            outputDims: weight.shape[1],
            numExperts: weight.shape[0],
            weight: MLXArray.zeros([1, 1, groupSize]),
            bias: bias
        )
        let quantized = QuantizedSwitchLinear(
            placeholder,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        var parameters = [
            "weight": weight,
            "scales": scales,
        ]
        if let biases {
            parameters["biases"] = biases
        }
        if let bias {
            parameters["bias"] = bias
        }
        quantized.update(
            parameters: ModuleParameters.unflattened(parameters)
        )
        quantized.freeze()
        return quantized
    }
}
