# Copyright (c) 2023 OpenAI. (authors: Whisper Team)
#               2024 Tsinghua Univ. (authors: Xingchen Song)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Speech tokenizer weights for MiniCPM-o."""

from __future__ import annotations

from pathlib import Path

import onnx
import torch


def rename_weights(weights_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new_weight_dict = {}
    for k in weights_dict.keys():
        if "quantizer" in k:
            if k == "/quantizer/rq/model/layers.0/_codebook/Pow_1":
                new_weight_dict["quantizer.codebook.embed"] = weights_dict[k]
            elif "project_down" in k:
                new_weight_dict[k] = weights_dict[k]
            else:
                pass
        elif "positional_embedding" in k:
            new_weight_dict[k] = weights_dict[k]
        elif "conv" in k:
            new_weight_dict[k] = weights_dict[k]
        else:
            assert "blocks" in k
            new_k = (
                k[1:]
                .replace("/", ".")
                .replace("MatMul", "weight")
                .replace("Add_1", "bias")
                .replace("Mul", "weight")
                .replace("Add", "bias")
                .replace("mlp.mlp", "mlp")
                .replace("fsmn_block.Conv", "fsmn_block.weight")
            )
            new_weight_dict[f"encoder.{new_k}"] = weights_dict[k]
    return new_weight_dict


def load_tokenizer_weights(onnx_path: Path) -> dict[str, torch.Tensor]:
    """Load the V2 tokenizer tensors from its original ONNX checkpoint."""
    onnx_model = onnx.load(str(onnx_path))
    weights_dict = {}
    initializer_map = {
        initializer.name: initializer for initializer in onnx_model.graph.initializer
    }
    for node in onnx_model.graph.node:
        for input_name in node.input:
            if input_name in initializer_map:
                ln_bias_name, ln_weight_name = (None, None)
                initializer = initializer_map[input_name]
                if input_name in [
                    "onnx::Conv_1519",
                    "encoders.conv1.weight",
                    "onnx::Conv_2216",
                ]:
                    weight_name = "encoder.conv1.weight"
                elif input_name in [
                    "onnx::Conv_1520",
                    "encoders.conv1.bias",
                    "onnx::Conv_2217",
                ]:
                    weight_name = "encoder.conv1.bias"
                elif input_name in [
                    "onnx::Conv_1521",
                    "encoders.conv2.weight",
                    "onnx::Conv_2218",
                ]:
                    weight_name = "encoder.conv2.weight"
                elif input_name in [
                    "onnx::Conv_1522",
                    "encoders.conv2.bias",
                    "onnx::Conv_2219",
                ]:
                    weight_name = "encoder.conv2.bias"
                elif input_name == "encoders.positional_embedding":
                    weight_name = "encoder.positional_embedding"
                elif input_name == "quantizer.project_in.bias":
                    weight_name = "quantizer.codebook.project_down.bias"
                elif input_name == "onnx::MatMul_2536":
                    weight_name = "quantizer.codebook.project_down.weight"
                elif node.op_type == "LayerNormalization":
                    ln_name = node.name.replace("/LayerNormalization", "")
                    ln_weight_name = ln_name + ".weight"
                    ln_bias_name = ln_name + ".bias"
                else:
                    weight_name = node.name
                if ln_weight_name is not None and ln_bias_name is not None:
                    ln_inputs = node.input
                    scale_name = ln_inputs[1]
                    bias_name = ln_inputs[2]
                    scale = onnx.numpy_helper.to_array(
                        initializer_map[scale_name]
                    ).copy()
                    bias = onnx.numpy_helper.to_array(initializer_map[bias_name]).copy()
                    weight_tensor = torch.from_numpy(scale)
                    bias_tensor = torch.from_numpy(bias)
                    weights_dict[ln_bias_name] = bias_tensor
                    weights_dict[ln_weight_name] = weight_tensor
                else:
                    weight_array = onnx.numpy_helper.to_array(initializer).copy()
                    weight_tensor = torch.from_numpy(weight_array)
                    if len(weight_tensor.shape) > 2 or weight_name in [
                        "encoder.positional_embedding"
                    ]:
                        weights_dict[weight_name] = weight_tensor
                    else:
                        weights_dict[weight_name] = weight_tensor.t()
            else:
                pass
    return rename_weights(weights_dict)
