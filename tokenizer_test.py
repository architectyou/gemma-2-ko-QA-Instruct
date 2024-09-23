from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/data/test/0920_merged"
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation='eager'
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


print(f"Tokenizer vocab size: {len(tokenizer)}")
print(f"Model vocab size: {model.config.vocab_size}")

print(f"EOS token: {tokenizer.eos_token}")
print(f"EOS token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.sep_token}")
print(f"EOS token: {tokenizer.bos_token}")
print(f"EOS token keys: {dir(tokenizer)}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"Model config EOS token ID: {model.config.eos_token_id}")