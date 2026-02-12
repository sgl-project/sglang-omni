import traceback

import torch

from sglang_omni.models.qwen3_asr.components.audio_encoder import Qwen3ASRAudioEncoder
from sglang_omni.models.qwen3_asr.components.thinker import Qwen3ASRSplitThinker


def run_demo():
    model_id = "/opt/gpfs/home/tianteng/Qwen3-ASR/Qwen3-ASR-0.6B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Using device: {device}, dtype: {dtype}")

    print("\n1. Initializing AudioEncoder...")
    try:
        audio_encoder = Qwen3ASRAudioEncoder(model_id, device=device, dtype=dtype)
        print("AudioEncoder initialized successfully.")
    except Exception as e:
        print(f"AudioEncoder initialization failed: {e}")
        traceback.print_exc()
        return

    print("\n2. Initializing SplitThinker...")
    try:
        thinker = Qwen3ASRSplitThinker(model_id, device=device, dtype=dtype)
        print("SplitThinker initialized successfully.")
    except Exception as e:
        print(f"SplitThinker initialization failed: {e}")
        traceback.print_exc()
        return

    print("\n3. Testing AudioEncoder forward pass...")
    try:
        # Dummy input: (batch=1, mel=128, time=100)
        # Qwen3-ASR expects (batch, mel, time) for audio tower
        dummy_features = torch.randn(1, 128, 100).to(device=device, dtype=dtype)
        dummy_lengths = torch.tensor([100], device=device)
        audio_out = audio_encoder(
            input_features=dummy_features, audio_feature_lengths=dummy_lengths
        )
        audio_embeds = audio_out["audio_embeds"]
        print(f"AudioEncoder forward successful. Output shape: {audio_embeds.shape}")
    except Exception as e:
        print(f"AudioEncoder forward failed: {e}")
        traceback.print_exc()
        return

    print("\n4. Testing SplitThinker forward pass...")
    try:
        audio_token_id = thinker.thinker.config.audio_token_id
        num_audio_tokens = audio_embeds.shape[0]

        # Construct input_ids with audio tokens followed by some text tokens
        input_ids = torch.full(
            (1, num_audio_tokens + 5), 10, dtype=torch.long, device=device
        )
        input_ids[0, :num_audio_tokens] = audio_token_id

        thinker_out = thinker(input_ids=input_ids, audio_embeds=audio_embeds)
        print(
            f"SplitThinker forward successful. Logits shape: {thinker_out['logits'].shape}"
        )
    except Exception as e:
        print(f"SplitThinker forward failed: {e}")
        traceback.print_exc()
        return

    print("\nQwen3-ASR Component Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
