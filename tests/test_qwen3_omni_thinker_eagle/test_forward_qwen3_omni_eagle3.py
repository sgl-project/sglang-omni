# Test script to verify the collaborative forward pass and generate loop
# of the Qwen3-Omni Thinker Target Model and the Qwen3-Omni Eagle3 Draft Model.
import json
import os

import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

from sglang_omni.models.qwen3_omni.draft import Qwen3OmniEagle3DraftModel
from sglang_omni.models.qwen3_omni.hf_config import Qwen3OmniMoeTextConfig
from sglang_omni.models.qwen3_omni.thinker import Qwen3OmniMoeThinkerTextModel


def test_forward():
    # If running on a GPU machine, use nccl instead of gloo
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    # Mock distributed environment since SGLang components depend on it
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["LOCAL_RANK"] = "0"

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    # SGLang uses custom init functions for its parallel state
    init_distributed_environment(backend=backend)
    initialize_model_parallel(1, 1)

    # Initialize DP Attention state
    from sglang.srt.layers.dp_attention import initialize_dp_attention

    current_dir = os.path.dirname(os.path.abspath(__file__))
    DUMMY_DIR = os.path.join(current_dir, "tiny-qwen3-omni")

    with open(os.path.join(DUMMY_DIR, "config.json"), "r") as f:
        config_dict = json.load(f)

    server_args = ServerArgs(model_path=DUMMY_DIR)
    # Set global server args since some modules fetch it globally
    set_global_server_args_for_scheduler(server_args)

    model_config = ModelConfig(model_path=DUMMY_DIR)

    initialize_dp_attention(server_args, model_config)

    # Extract thinker text config
    text_cfg_dict = config_dict["thinker_config"]["text_config"]
    text_cfg = Qwen3OmniMoeTextConfig(**text_cfg_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing structural forward logic on {device.upper()}...")

    # 1. Initialize Target Model
    target_model = (
        Qwen3OmniMoeThinkerTextModel(
            config=text_cfg,
            quant_config=None,
        )
        .to(device)
        .to(torch.float16)
    )

    # 2. Initialize Draft Model
    draft_model = (
        Qwen3OmniEagle3DraftModel(
            config=text_cfg,
            quant_config=None,
        )
        .to(device)
        .to(torch.float16)
    )

    # 3. Share Embed & Head
    draft_model.set_embed_and_head(target_model.embed_tokens, target_model.lm_head)

    print("Models initialized successfully!")

    if not torch.cuda.is_available():
        print("No GPU detected. Skipping forward pass execution.")
        return

    # ==========================================
    # Mocking Forward Execution (Generate Loop)
    # ==========================================
    print("GPU detected! Mocking forward pass and generate loop...")

    batch_size = 2
    seq_len = 5
    max_gen_steps = 4

    # Create a dummy ForwardBatch
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardBatch,
        ForwardMode,
    )

    class MockAttnBackend:
        def forward(self, q, k, v, layer, forward_batch, *args, **kwargs):
            total_tokens = q.shape[0]
            # Ensure output matches the expected shape from radix_attention
            out_dim = layer.tp_q_head_num * layer.v_head_dim
            return torch.zeros((total_tokens, out_dim), dtype=q.dtype, device=q.device)

    # Initial input ids
    input_ids = torch.randint(0, text_cfg.vocab_size, (batch_size,), device=device)
    current_positions = torch.arange(batch_size, device=device)

    print(f"Initial Target Model input_ids shape: {input_ids.shape}")

    try:
        for step in range(max_gen_steps):
            print(f"\n--- Generate Step {step+1}/{max_gen_steps} ---")

            # Re-initialize ForwardBatch for each step to simulate sequence length growing
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=batch_size,
                input_ids=input_ids,
                req_pool_indices=torch.zeros(
                    batch_size, dtype=torch.long, device=device
                ),
                seq_lens=torch.tensor([seq_len + step] * batch_size, device=device),
                seq_lens_sum=(seq_len + step) * batch_size,
                out_cache_loc=torch.zeros(batch_size, dtype=torch.long, device=device),
                return_logprob=True,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                positions=current_positions,
            )
            forward_batch.attn_backend = MockAttnBackend()

            # 1. Target Model Forward (Verify / initial generate)
            target_output = target_model(
                input_ids=input_ids,
                positions=forward_batch.positions,
                forward_batch=forward_batch,
            )

            target_logits = target_output.next_token_logits
            aux_hidden_states = getattr(target_output, "aux_hidden_states", None)

            # In LogitsProcessorOutput, the last hidden state is stored in hidden_states
            target_hidden = target_output.hidden_states

            if target_hidden is None:
                print("Warning: No hidden states captured from Target Model!")
                target_hidden = torch.randn(
                    batch_size, text_cfg.hidden_size, device=device, dtype=torch.float16
                )

            # Target model makes its prediction (for simulation, we just take argmax)
            # In a real Eagle3 setup, this would be the verification step
            target_next_token = torch.argmax(target_logits, dim=-1)
            print(f"Target predicted tokens: {target_next_token.tolist()}")

            # 2. Draft Model Forward (Speculate next tokens)
            # Draft model takes the target's output token (or just input_ids) and the hidden_states from Target
            # In true Speculative Decoding, Draft might run multiple steps. We simulate 1 step here.
            draft_output = draft_model(
                input_ids=input_ids,
                positions=forward_batch.positions,
                forward_batch=forward_batch,
                hidden_states=target_hidden,
            )

            draft_logits = draft_output.next_token_logits
            draft_next_token = torch.argmax(draft_logits, dim=-1)
            print(f"Draft speculated tokens: {draft_next_token.tolist()}")

            # 3. Prepare for next step
            # We'll feed the target's chosen token back as the input for the next step
            input_ids = target_next_token
            current_positions = current_positions + 1

        print("\nGenerate loop simulation completed successfully!")

    except Exception as e:
        print(f"Generate loop failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_forward()
