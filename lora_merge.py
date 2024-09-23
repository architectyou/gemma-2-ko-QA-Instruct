import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel, PeftConfig
from trl import setup_chat_format

# 원본 모델 경로 (fine-tuning 전 모델)
base_model_path = "/data/gguf_models/ko-gemma-2-9b-it"
# Fine-tuned 모델 경로
model_name = "/data/test/results/checkpoint-4536"

# Fine-tuned 모델(PEFT 모델) 설정 로드
peft_config = PeftConfig.from_pretrained(model_name)

# 토크나이저 로드 (원본 모델의 토크나이저 사용)
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# PEFT 모델의 vocab_size 확인
peft_vocab_size = peft_config.vocab_size if hasattr(peft_config, 'vocab_size') else 256002  # PEFT config에 없다면 오류 메시지의 크기 사용

# 원본 모델 로드 및 vocab 크기 조정
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    attn_implementation='eager', 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# base_model의 vocab 크기를 PEFT 모델과 일치시킴
base_model.resize_token_embeddings(peft_vocab_size)

print(f"Base model vocab size after resizing: {base_model.config.vocab_size}")

# PEFT 모델 로드
model = PeftModel.from_pretrained(base_model, model_name)

# 모델 병합
merged_model = model.merge_and_unload()

# setup_chat_format 적용
tokenizer.pad_token = '<pad>'
tokenizer.eos_token = '<eos>'
tokenizer.bos_token = '<bos>'

tokenizer.add_special_tokens({
    'pad_token': '<pad>',
    'eos_token': '<eos>',
    'bos_token': '<bos>'  # 필요한 경우
})

# 모델의 임베딩 레이어 크기 조정
merged_model.resize_token_embeddings(len(tokenizer))

# 모델 설정 업데이트
merged_model.config.pad_token_id = tokenizer.pad_token_id
merged_model.config.eos_token_id = tokenizer.eos_token_id
merged_model.config.bos_token_id = tokenizer.bos_token_id 

print(f"Tokenizer size after setup_chat_format: {len(tokenizer)}")
print(f"Model vocab size after setup_chat_format: {merged_model.config.vocab_size}")

# 병합된 모델 저장
output_dir = "./0920_merged"
merged_model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

print(f"병합된 모델이 {output_dir}에 저장되었습니다.")

# (선택사항) 저장된 모델 로드 테스트
test_model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)
test_tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)

print("저장된 모델 로드 테스트 완료")
print(f"Final model vocab size: {test_model.config.vocab_size}")
print(f"Final tokenizer vocab size: {len(test_tokenizer)}")

# 특수 토큰 확인
print(f"PAD token: {test_tokenizer.pad_token}, ID: {test_tokenizer.pad_token_id}")
print(f"EOS token: {test_tokenizer.eos_token}, ID: {test_tokenizer.eos_token_id}")
print(f"BOS token: {test_tokenizer.bos_token}, ID: {test_tokenizer.bos_token_id}")