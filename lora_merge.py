import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel, PeftConfig
from trl import setup_chat_format
import pdb

# 원본 모델 경로 (fine-tuning 전 모델)
base_model_path = "/data/gguf_models/ko-gemma-2-9b-it"
# Fine-tuned 모델 경로
model_name = "/data/test/fine_tuned_legal_model_lora"

# Fine-tuned 모델(PEFT 모델) 설정 로드
peft_config = PeftConfig.from_pretrained(model_name)
# 토크나이저 로드 (원본 모델의 토크나이저 사용)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# 원본 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
)

model = PeftModel.from_pretrained(base_model, model_name)
merged_model = model.merge_and_unload()

print(f"Base model vocab size after resizing: {merged_model.config.vocab_size}")

# 병합된 모델 저장
output_dir = "./0927_merged"
merged_model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

print(f"병합된 모델이 {output_dir}에 저장되었습니다.")

# (선택사항) 저장된 모델 로드 테스트
test_model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
test_tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)

print("저장된 모델 로드 테스트 완료")
print(f"Final model vocab size: {test_model.config.vocab_size}")
print(f"Final tokenizer vocab size: {len(test_tokenizer)}")