"""Quick check: model hidden size vs ZImage cap_feat_dim."""
from transformers import AutoTokenizer, AutoConfig
import time

MODEL_PATH = "/data/cache/huggingface/hub/models--inclusionAI--Ming-flash-omni-2.0/snapshots/6a2e1dec07066d20f62a743ac7c34284e4a3932d"

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Tokenizer loaded in {time.time()-t0:.1f}s, vocab_size={tok.vocab_size}")

config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"Model type: {config.architectures}")

tc = getattr(config, "text_config", None)
if tc:
    print(f"text hidden_size: {getattr(tc, 'hidden_size', 'N/A')}")
else:
    print(f"hidden_size: {getattr(config, 'hidden_size', 'N/A')}")

# ZImage transformer expects cap_feat_dim=2560
print(f"\nZImage cap_feat_dim = 2560")
if tc:
    hs = getattr(tc, "hidden_size", 0)
    print(f"LLM hidden_size = {hs}")
    if hs != 2560:
        print("MISMATCH: LLM hidden_size != cap_feat_dim. Need a connector/projection.")
    else:
        print("MATCH: Can use LLM hidden states directly.")
