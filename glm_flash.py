"""
GLM-4.7-Flash-FP8-Dynamic Inference Script
==========================================

WARNING: THIS SCRIPT DOES NOT WORK ON BLACKWELL GPUs (SM 120)
----------------------------------------------------------
FP8 quantized models are incompatible with Blackwell GPUs as of Feb 2026.

REASON:
- PyTorch's torch._grouped_mm and torch.baddbmm_cuda don't support Float8_e4m3fn on SM 120
- Error: "baddbmm_cuda not implemented for 'Float8_e4m3fn'"
- Both grouped_mm and batched_mm implementations fail with FP8 tensors

WHEN WILL IT WORK:
- Waiting for PyTorch to add Blackwell FP8 kernel support
- Open feature request: https://github.com/pytorch/pytorch/issues/160891
- vLLM also lacks Blackwell FP8 support (requires SM 90 or SM 100)
- Estimate: PyTorch 2.8+ or a future CUDA/cuDNN update

ALTERNATIVES:
1. Use the BF16 model: zai-org/GLM-4.7-Flash (~62GB VRAM)
2. Use a smaller model: LiquidAI/LFM2.5-1.2B-Thinking (works on Blackwell)
3. Wait for Blackwell FP8 support in PyTorch/transformers

USAGE: python glm_flash.py (will fail on Blackwell)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import os

# Blackwell optimizations
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Local path to GLM-4.7-Flash-FP8-Dynamic model
MODEL_PATH = "/home/jonathan/Models_Transformer/GLM-4.7-Flash-FP8-Dynamic"

# NOTE: Neither grouped_mm nor batched_mm work with FP8 on Blackwell
# This workaround does NOT fix the issue - both fail with Float8_e4m3fn
from transformers import AutoConfig
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
config._experts_implementation = "batched_mm"  # Still fails on Blackwell

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config,
    device_map="auto",
    dtype="bfloat16",
    trust_remote_code=True,
    local_files_only=True,
    # attn_implementation="flash_attention_2"  # Uncomment on compatible GPU
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

prompt = "What is C. elegans?"

input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
    tokenize=True,
).to(model.device)

print(f"Prompt: {prompt}")
print("Generating response...")

output = model.generate(
    input_ids,
    do_sample=True,
    temperature=0.1,
    top_k=50,
    top_p=0.1,
    repetition_penalty=1.05,
    max_new_tokens=512,
    streamer=streamer,
)

print("\nGeneration complete!")
