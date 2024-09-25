base_model_url = "models/ko-gemma-2-9b-it"
new_model_url = "fine_tuned_legal_model_lora"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import torch
from trl import setup_chat_format


# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model_url)

base_model_reload= AutoModelForCausalLM.from_pretrained(
    base_model_url,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cpu",
)

base_model_reload, tokenizer = setup_chat_format(base_model_reload, tokenizer)
model = PeftModel.from_pretrained(base_model_reload, new_model_url)

model = model.merge_and_unload()

model.save_pretrained("results/Gemma-2-9b-it-chat-doctor")
tokenizer.save_pretrained("results/Gemma-2-9b-it-chat-doctor")