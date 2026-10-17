"""Keep the original fixture architecture, but exercise replicated KV heads."""
from pathlib import Path
import torch
from transformers import Llama4TextConfig, Llama4ForCausalLM

torch.manual_seed(61886)
config = Llama4TextConfig.from_pretrained('/data/vllm-gpu-validation/tiny-llama4')
config.num_key_value_heads = 1
model = Llama4ForCausalLM(config).to(torch.bfloat16)
model.save_pretrained('/data/vllm-gpu-followup/tiny-llama4-kv1', safe_serialization=True)
print(config.to_json_string())
print('parameters', sum(p.numel() for p in model.parameters()))
