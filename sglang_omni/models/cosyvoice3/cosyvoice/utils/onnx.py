# Copyright (c) 2024 Alibaba Inc (CosyVoice)
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
#
# Vendored from CosyVoice (https://github.com/FunAudioLLM/CosyVoice).
import os
import random

import onnxruntime
import torch
import torchaudio.compliance.kaldi as kaldi


class SpeechTokenExtractor:
    def __init__(self, model_path):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        option.intra_op_num_threads = 1
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=option,
            providers=[("CUDAExecutionProvider", {"device_id": self.local_rank})],
        )

    def inference(self, feat, feat_lengths, device):
        speech_token = self.speech_tokenizer_session.run(
            None,
            {
                self.speech_tokenizer_session.get_inputs()[0]
                .name: feat.transpose(1, 2)
                .detach()
                .cpu()
                .numpy(),
                self.speech_tokenizer_session.get_inputs()[1]
                .name: feat_lengths.detach()
                .cpu()
                .numpy(),
            },
        )[0]
        return torch.tensor(speech_token).to(torch.int32).to(device), (
            feat_lengths / 4
        ).to(torch.int32).to(device)


class EmbeddingExtractor:
    def __init__(self, model_path):
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        option.intra_op_num_threads = 1
        self.max_len = 10 * 16000
        self.campplus_session = onnxruntime.InferenceSession(
            model_path, sess_options=option, providers=["CPUExecutionProvider"]
        )

    def inference(self, speech):
        if speech.shape[1] > self.max_len:
            start_index = random.randint(0, speech.shape[1] - self.max_len)
            speech = speech[:, start_index : start_index + self.max_len]
        feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = (
            self.campplus_session.run(
                None,
                {
                    self.campplus_session.get_inputs()[0]
                    .name: feat.unsqueeze(dim=0)
                    .cpu()
                    .numpy()
                },
            )[0]
            .flatten()
            .tolist()
        )
        return torch.tensor(embedding).to(speech.device)


# LOCAL MODIFICATION (sglang-omni): upstream read a GENERIC `onnx_path` env var at import and,
# if set, eagerly built an ONNX EmbeddingExtractor (online speaker-embedding extraction). The
# sglang-omni pipeline supplies speech tokens and the campplus x-vector via the frontend and
# never uses this online extractor; a generic env-var name also risks colliding with an
# unrelated host setting and allocating/failing at import. So the online path is disabled and
# nothing is constructed at import time. `onnx_path` is kept defined (None) for the flow import.
onnx_path = None
embedding_extractor, online_feature = None, False
