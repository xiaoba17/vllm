"""Deterministic small Llama 4 checkpoint for execution correctness, not quality."""
from pathlib import Path
import torch
from transformers import Llama4TextConfig, Llama4ForCausalLM

torch.manual_seed(61886)
config = Llama4TextConfig(
    vocab_size=512, hidden_size=256, intermediate_size=512,
    intermediate_size_mlp=512, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2, head_dim=64,
    num_local_experts=4, num_experts_per_tok=1, interleave_moe_layer_step=2,
    no_rope_layers=[1, 1, 1, 0], attention_chunk_size=128,
    max_position_embeddings=1024, bos_token_id=1, eos_token_id=2,
    pad_token_id=0, tie_word_embeddings=False,
)
model = Llama4ForCausalLM(config).to(torch.bfloat16)
path = Path('/data/vllm-gpu-validation/tiny-llama4')
model.save_pretrained(path, safe_serialization=True)
print(config.to_json_string())
print('parameters', sum(p.numel() for p in model.parameters()))

