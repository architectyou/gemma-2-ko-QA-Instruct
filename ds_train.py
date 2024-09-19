import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import os
import json
from sklearn.model_selection import train_test_split
from peft import get_peft_model, LoraConfig, TaskType
import wandb
import subprocess

# wandb 초기화
try:
    subprocess.run(["wandb", "login"], check=True)
except subprocess.CalledProcessError:
    print("Wandb 로그인에 실패했습니다. 수동으로 로그인해주세요.")

wandb.init(project="legal-model-finetuning", name="gemma-2-9b-lora-bf16-improved")

# 모델과 토크나이저 로드
model_name = "/data/gguf_models/ko-gemma-2-9b-it/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    attn_implementation='eager', 
    torch_dtype=torch.bfloat16,
    use_cache=False  # 그래디언트 체크포인팅과 호환되도록 설정
)

# LoRA 설정
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# LoRA 모델 생성
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 모델을 명시적으로 훈련 모드로 설정
model.train()

# 데이터 로드 함수
def load_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                data.append(json.load(f))
    return data

# 데이터 로드
data_directory = "./dataset/law_QA_dataset/"
all_data = load_json_files(data_directory)

# 훈련/검증 데이터 분리
train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)

# 데이터셋 생성
def create_dataset(data):
    return {
        "id": [item["id"] for item in data],
        "question": [item["question"] for item in data],
        "answer": [item["answer"] for item in data],
        "context": [f"{item['title']}\n{item['commentary']}" for item in data]
    }

train_dataset = create_dataset(train_data)
val_dataset = create_dataset(val_data)

# 데이터 전처리 함수
def preprocess_function(examples):
    max_length = 512  # 또는 모델에 맞는 적절한 길이
    prompts = [f"Context: {context}\nQuestion: {question}\nAnswer:" 
               for context, question in zip(examples["context"], examples["question"])]
    
    inputs = tokenizer(prompts, padding=False, truncation=True, max_length=max_length)
    labels = tokenizer(examples["answer"], padding=False, truncation=True, max_length=max_length)
    
    # 입력과 라벨의 길이를 맞춤
    for i in range(len(inputs["input_ids"])):
        input_length = len(inputs["input_ids"][i])
        label_length = len(labels["input_ids"][i])
        if input_length > label_length:
            labels["input_ids"][i] += [-100] * (input_length - label_length)
        elif input_length < label_length:
            labels["input_ids"][i] = labels["input_ids"][i][:input_length]
    
    inputs["labels"] = labels["input_ids"]
    return inputs

# 데이터셋 전처리 적용
train_tokenized = Dataset.from_dict(train_dataset).map(preprocess_function, batched=True, remove_columns=train_dataset.keys())
val_tokenized = Dataset.from_dict(val_dataset).map(preprocess_function, batched=True, remove_columns=val_dataset.keys())


# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    learning_rate=1e-4,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    deepspeed="ds_config.json",  # DeepSpeed 설정 파일 경로
    report_to="wandb",
    run_name="gemma-2-9b-lora-bf16-improved",
)

# Wandb에 하이퍼파라미터 로깅
def get_wandb_config(args):
    config = {}
    for k, v in vars(args).items():
        if not k.startswith('_') and k != 'deepspeed':
            if isinstance(v, (int, float, str, bool, list, dict)):
                config[k] = v
            else:
                config[k] = str(v)
    return config

wandb_config = get_wandb_config(training_args)
wandb.config.update(wandb_config)

# 사용자 정의 데이터 콜레이터 (동적 패딩 사용)
def custom_data_collator(features):
    batch = {}
    for key in features[0].keys():
        if key != "labels":
            batch[key] = [feature[key] for feature in features]
        else:
            batch[key] = [feature[key] + [-100] * (max(len(f["input_ids"]) for f in features) - len(feature[key])) for feature in features]
    
    batch = tokenizer.pad(batch, padding=True, return_tensors="pt")
    
    if "label" in batch:
        batch["labels"] = batch["label"]
        del batch["label"]
    if "label_ids" in batch:
        batch["labels"] = batch["label_ids"]
        del batch["label_ids"]
    
    return batch

# Trainer 초기화 및 훈련
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=custom_data_collator,
)

# 훈련 실행
try:
    trainer.train()
except Exception as e:
    print(f"훈련 중 오류 발생: {e}")
    wandb.finish()
    raise

# 모델 저장
trainer.save_model("./fine_tuned_legal_model_lora")

# LoRA 모델 병합 및 전체 모델 저장 (선택사항)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./fine_tuned_legal_model_merged")

# 평가
eval_results = trainer.evaluate()
print(eval_results)
wandb.log({"eval_results": eval_results})

# 새로운 질문에 대한 응답 생성 (예시)
def generate_answer(context, question):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=150, num_return_sequences=1, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 테스트
test_context = "상법 제321조 제2항은 회사 성립시의 발기인의 납입담보책임에 관하여 규정하고 있습니다."
test_question = "회사성립의 경우 발기인의 납입담보책임에 관해서 설명해 주십시오."
test_answer = generate_answer(test_context, test_question)
print(test_answer)
wandb.log({"test_example": {"question": test_question, "answer": test_answer}})

# CUDA 캐시 정리
torch.cuda.empty_cache()

# wandb 실행 종료
wandb.finish()