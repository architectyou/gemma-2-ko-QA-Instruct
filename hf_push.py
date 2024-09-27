import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin, login
from transformers import AutoTokenizer, AutoModelForCausalLM

login()

def load_model(model_id):
    model=AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer=AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


# push to the hub
if __name__=="__main__":
    model, tokenizer=load_model(model_id="0927_merged")
    model.push_to_hub("architectyou/law-gemma-2-ko-9b-it", private=True)
    tokenizer.push_to_hub("architectyou/law-gemma-2-ko-9b-it")