import pdb
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    pipeline
)
from datasets import load_dataset, Dataset
import os, torch, json, wandb, subprocess
from sklearn.model_selection import train_test_split
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType,
)
import torch.nn as nn
from trl import SFTTrainer, setup_chat_format

# wandb 로그인
try:
    subprocess.run(["wandb", "login"], check=True)
except subprocess.CalledProcessError:
    print("Wandb 로그인에 실패했습니다. 수동으로 로그인해주세요.")

wandb.init(project="legal-model-finetuning", name="gemma-2-9b-lora-bf16-0923")

# 모델과 토크나이저 로드
base_model = "/data/gguf_models/ko-gemma-2-9b-it/"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code = True)
model = AutoModelForCausalLM.from_pretrained(
    base_model, 
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation='eager',
)

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

modules = find_all_linear_names(model)
# LoRA 설정
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=modules,
)

# model, tokenizer = setup_chat_format(model, tokenizer)
# LoRA 모델 생성
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 모델을 명시적으로 훈련 모드로 설정
model.train()

#---------------------------------------------------------------------------------------#
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
    max_length = 512
    
    inputs = []
    labels = []
    
    for context, question, answer in zip(examples["context"], examples["question"], examples["answer"]):
        messages = [
            {"role": "user",
             "content": f"다음 문서를 참고하여 질문에 답변해주세요:\n\nContext: {context}\nQuestion: {question}"}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 입력 인코딩
        encoded_input = tokenizer(prompt, max_length=max_length, padding="max_length", truncation=True)
        
        # 레이블 인코딩
        encoded_label = tokenizer(answer, max_length=max_length, padding="max_length", truncation=True)
        pdb.set_trace()
        
        # 레이블에 대해 -100으로 패딩
        encoded_label["input_ids"] = [-100 if token == tokenizer.pad_token_id else token for token in encoded_label["input_ids"]]
        
        inputs.append(encoded_input)
        labels.append(encoded_label["input_ids"])
    
    batch_inputs = {
        "input_ids": [inp["input_ids"] for inp in inputs],
        "attention_mask": [inp["attention_mask"] for inp in inputs],
        "labels": labels
    }
    
    # pdb.set_trace()
    return batch_inputs

# 데이터셋 전처리 적용
train_tokenized = Dataset.from_dict(train_dataset).map(preprocess_function, batched=True, remove_columns=train_dataset.keys())
val_tokenized = Dataset.from_dict(val_dataset).map(preprocess_function, batched=True, remove_columns=val_dataset.keys())

#---------------------------------------------------------------------------------------------#

# 훈련 인자 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    optim="paged_adamw_32bit",
    eval_strategy="steps",
    eval_steps=0.1,
    logging_dir="./logs",
    logging_steps=11,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=1e-7,
    group_by_length=True,
    bf16=True,
    # report_to="wandb",
    run_name="gemma-2-9b-lora-bf16-0924",
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

# wandb_config = get_wandb_config(training_args)
# wandb.config.update(wandb_config)

# 사용자 정의 데이터 콜레이터 (동적 패딩 사용)
def custom_data_collator(features):
    batch = {}
    for key in features[0].keys():
        if key not in ["labels", "input_ids", "attention_mask"]:
            batch[key] = [feature[key] for feature in features]
        else:
            # 동적 패딩을 위해 최대 길이 계산
            max_length = max(len(feature[key]) for feature in features)
            
            # input_ids와 attention_mask는 오른쪽에 패딩
            if key in ["input_ids", "attention_mask"]:
                batch[key] = [feature[key] + [tokenizer.pad_token_id if key == "input_ids" else 0] * (max_length - len(feature[key])) for feature in features]
            
            # labels는 오른쪽에 -100으로 패딩
            elif key == "labels":
                batch[key] = [feature[key] + [-100] * (max_length - len(feature[key])) for feature in features]
    
    # 텐서로 변환
    for key in batch:
        batch[key] = torch.tensor(batch[key])
    
    return batch

# 사용 예시:
# train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_data_collator)

# Trainer 초기화 및 훈련
trainer = SFTTrainer(
    model=model,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    peft_config=peft_config,
    max_seq_length = 512,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    data_collator=custom_data_collator,
)

# 훈련 실행
try:
    model.config.use_cache = False
    trainer.train()
except Exception as e:
    print(f"훈련 중 오류 발생: {e}")
    wandb.finish()
    raise

# 모델 저장
trainer.model.save_pretrained("./fine_tuned_legal_model_lora")

# # LoRA 모델 병합 및 전체 모델 저장 (선택사항)
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained("./fine_tuned_legal_model_merged")

# 평가
eval_results = trainer.evaluate()
print(eval_results)
wandb.log({"eval_results": eval_results})

# # 새로운 질문에 대한 응답 생성 (예시)
# def generate_answer(context, question):
#     prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_length=150, num_return_sequences=1, temperature=0.7)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # 테스트
# test_context = "상법 제321조 제2항은 회사 성립시의 발기인의 납입담보책임에 관하여 규정하고 있습니다."
# test_question = "회사성립의 경우 발기인의 납입담보책임에 관해서 설명해 주십시오."
# test_answer = generate_answer(test_context, test_question)
# print(test_answer)
# wandb.log({"test_example": {"question": test_question, "answer": test_answer}})

# CUDA 캐시 정리
torch.cuda.empty_cache()

# wandb 실행 종료
wandb.finish()