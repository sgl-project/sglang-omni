# Script to generate tiny dummy weights for testing Qwen3-Omni Thinker with Eagle3 Draft Model.
# Used for testing forward passes locally on GPU without full 30B weights.
import json
import os

import safetensors.torch
import torch
from transformers import AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
DUMMY_DIR = os.path.join(current_dir, "tiny-qwen3-omni")
DRAFT_DIR = os.path.join(current_dir, "tiny-qwen3-omni-draft")

os.makedirs(DUMMY_DIR, exist_ok=True)
os.makedirs(DRAFT_DIR, exist_ok=True)

# 1. Create a dummy tokenizer (using an existing one like qwen2.5 just for vocab)
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer.save_pretrained(DUMMY_DIR)
    tokenizer.save_pretrained(DRAFT_DIR)
    vocab_size = len(tokenizer)
except Exception:
    vocab_size = 151936  # Qwen default
    # Create minimal tokenizer config if AutoTokenizer fails

# 2. Dummy config for Target Model (Thinker)
# To keep the total size under 100MB while having 48 layers:
# Let's drastically reduce hidden_size, intermediate_size, vocab_size, etc.
# Note: vocab_size determines embed/lm_head size.
# 48 layers with large vocab could easily exceed 100MB if we aren't careful.
hidden_size = 128  # Reduced from true size
intermediate_size = 256  # Reduced from true size
num_hidden_layers = 48  # Full 48 layers
num_attention_heads = 2  # Keep small
num_key_value_heads = 1
num_experts = 4
num_experts_per_tok = 2
vocab_size = 151936  # Keeping true vocab size, but embed size will be 151936 * 128 * 2 bytes ≈ 38.9MB
# 48 layers * small params will be very small.
# Total size will be dominated by vocab (embed + lm_head = ~77MB).
# Total should be around 90MB.


target_config = {
    "architectures": ["Qwen3OmniMoeForConditionalGeneration"],
    "model_type": "qwen3_omni_moe",
    "thinker_config": {
        "vision_config": {"deepstack_visual_indexes": [8, 16, 24]},
        "text_config": {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "hidden_act": "silu",
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "moe_intermediate_size": intermediate_size,
            "head_dim": hidden_size // num_attention_heads,
            "rope_scaling": {
                "interleaved": True,
                "mrope_interleaved": True,
                "mrope_section": [16, 24, 24],
                "rope_type": "default",
                "type": "default",
            },
        },
    },
}

with open(os.path.join(DUMMY_DIR, "config.json"), "w") as f:
    json.dump(target_config, f, indent=2)

# 3. Dummy config for Draft Model
draft_config = {
    "architectures": ["Qwen3OmniEagle3DraftModel"],
    "model_type": "qwen3_omni_moe",
    "thinker_config": {
        "vision_config": {"deepstack_visual_indexes": [8, 16, 24]},
        "text_config": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": 1,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "vocab_size": vocab_size,
            "max_position_embeddings": 1024,
            "rms_norm_eps": 1e-6,
            "head_dim": hidden_size // num_attention_heads,
            "rope_scaling": {
                "interleaved": True,
                "mrope_interleaved": True,
                "mrope_section": [16, 24, 24],
                "rope_type": "default",
                "type": "default",
            },
        },
    },
}

with open(os.path.join(DRAFT_DIR, "config.json"), "w") as f:
    json.dump(draft_config, f, indent=2)

# 4. Generate random weights for Target Model
target_tensors = {}
# Use torch.float16 directly to save disk space
target_tensors["model.embed_tokens.weight"] = torch.randn(
    vocab_size, hidden_size, dtype=torch.float16
)

# Vision and Audio weights based on Qwen3VLMoE and Qwen3OmniMoe loading structure
target_tensors["visual.merger.mlp.0.weight"] = torch.randn(
    hidden_size, hidden_size, dtype=torch.float16
)
target_tensors["visual.merger.mlp.0.bias"] = torch.randn(
    hidden_size, dtype=torch.float16
)
target_tensors["visual.merger.mlp.2.weight"] = torch.randn(
    hidden_size, hidden_size, dtype=torch.float16
)
target_tensors["visual.merger.mlp.2.bias"] = torch.randn(
    hidden_size, dtype=torch.float16
)
target_tensors["audio.merger.mlp.0.weight"] = torch.randn(
    hidden_size, hidden_size, dtype=torch.float16
)
target_tensors["audio.merger.mlp.0.bias"] = torch.randn(
    hidden_size, dtype=torch.float16
)
target_tensors["audio.merger.mlp.2.weight"] = torch.randn(
    hidden_size, hidden_size, dtype=torch.float16
)
target_tensors["audio.merger.mlp.2.bias"] = torch.randn(
    hidden_size, dtype=torch.float16
)

for i in range(num_hidden_layers):
    prefix = f"model.layers.{i}."
    # Attention QKV and O
    target_tensors[prefix + "self_attn.q_proj.weight"] = torch.randn(
        hidden_size, hidden_size, dtype=torch.float16
    )
    target_tensors[prefix + "self_attn.k_proj.weight"] = torch.randn(
        2 * (hidden_size // num_attention_heads) * num_key_value_heads // 2,
        hidden_size,
        dtype=torch.float16,
    )
    target_tensors[prefix + "self_attn.v_proj.weight"] = torch.randn(
        2 * (hidden_size // num_attention_heads) * num_key_value_heads // 2,
        hidden_size,
        dtype=torch.float16,
    )
    target_tensors[prefix + "self_attn.o_proj.weight"] = torch.randn(
        hidden_size, hidden_size, dtype=torch.float16
    )

    # MoE Gate
    target_tensors[prefix + "mlp.gate.weight"] = torch.randn(
        num_experts, hidden_size, dtype=torch.float16
    )

    # MoE Experts (Unfused in HuggingFace checkpoint, SGLang fuses them on load)
    for e in range(num_experts):
        target_tensors[prefix + f"mlp.experts.{e}.gate_proj.weight"] = torch.randn(
            intermediate_size, hidden_size, dtype=torch.float16
        )
        target_tensors[prefix + f"mlp.experts.{e}.up_proj.weight"] = torch.randn(
            intermediate_size, hidden_size, dtype=torch.float16
        )
        target_tensors[prefix + f"mlp.experts.{e}.down_proj.weight"] = torch.randn(
            hidden_size, intermediate_size, dtype=torch.float16
        )

    # Shared Expert
    target_tensors[prefix + "mlp.shared_expert.gate_proj.weight"] = torch.randn(
        intermediate_size, hidden_size, dtype=torch.float16
    )
    target_tensors[prefix + "mlp.shared_expert.up_proj.weight"] = torch.randn(
        intermediate_size, hidden_size, dtype=torch.float16
    )
    target_tensors[prefix + "mlp.shared_expert.down_proj.weight"] = torch.randn(
        hidden_size, intermediate_size, dtype=torch.float16
    )
    target_tensors[prefix + "mlp.shared_expert_gate.weight"] = torch.randn(
        1, hidden_size, dtype=torch.float16
    )

    # LayerNorms
    target_tensors[prefix + "input_layernorm.weight"] = torch.ones(
        hidden_size, dtype=torch.float16
    )
    target_tensors[prefix + "post_attention_layernorm.weight"] = torch.ones(
        hidden_size, dtype=torch.float16
    )

target_tensors["model.norm.weight"] = torch.ones(hidden_size, dtype=torch.float16)
target_tensors["lm_head.weight"] = torch.randn(
    vocab_size, hidden_size, dtype=torch.float16
)

safetensors.torch.save_file(
    target_tensors, os.path.join(DUMMY_DIR, "model.safetensors")
)

# 5. Generate random weights for Draft Model
draft_tensors = {}
draft_tensors["fc.weight"] = torch.randn(
    hidden_size, hidden_size * 2, dtype=torch.float16
)
draft_tensors["fc.bias"] = torch.randn(hidden_size, dtype=torch.float16)
prefix = "layer."
draft_tensors[prefix + "self_attn.qkv_proj.weight"] = torch.randn(
    hidden_size + 2 * (hidden_size // num_attention_heads) * num_key_value_heads,
    hidden_size,
    dtype=torch.float16,
)
draft_tensors[prefix + "self_attn.o_proj.weight"] = torch.randn(
    hidden_size, hidden_size, dtype=torch.float16
)
draft_tensors[prefix + "mlp.gate_up_proj.weight"] = torch.randn(
    intermediate_size * 2, hidden_size, dtype=torch.float16
)
draft_tensors[prefix + "mlp.down_proj.weight"] = torch.randn(
    hidden_size, intermediate_size, dtype=torch.float16
)
draft_tensors[prefix + "input_layernorm.weight"] = torch.ones(
    hidden_size, dtype=torch.float16
)
draft_tensors[prefix + "post_attention_layernorm.weight"] = torch.ones(
    hidden_size, dtype=torch.float16
)
draft_tensors["norm.weight"] = torch.ones(hidden_size, dtype=torch.float16)

safetensors.torch.save_file(draft_tensors, os.path.join(DRAFT_DIR, "model.safetensors"))

print("Dummy models generated successfully!")
