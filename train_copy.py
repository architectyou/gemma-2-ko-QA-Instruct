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

wandb.init(project="legal-model-finetuning", name="gemma-2-9b-lora-bf16-0926")

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
def load_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                data.append(json.load(f))
    return data

# 데이터셋 생성
def create_dataset(data):
    dataset_dict = {
        "id": [],
        "question": [],
        "answer": [],
        "context": []
    }
    
    for item in data:
        dataset_dict["id"].append(item["id"])
        dataset_dict["question"].append(item["question"])
        dataset_dict["answer"].append(item["answer"])
        dataset_dict["context"].append(f"{item['title']}\n{item['commentary']}")
    
    return Dataset.from_dict(dataset_dict)

# dataset load & preprocessing
data_directory = "./dataset/law_QA_dataset/"
all_data = load_json_files(data_directory)

train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)

train_dataset = create_dataset(train_data)
val_dataset = create_dataset(val_data)

def generate_prompts(examples):
    prompt_list=[]
    for context, question, answer in zip(examples["context"], examples["question"], examples["answer"]):
        prompt_list.append(
            f"""<bos><start_of_turn>user
            다음 문서를 참고하여 질문에 답변해주세요:
            
            Context: {context}
            Question: {question}
            <end_of_turn>
            <start_of_turn>model
            {answer}<end_of_turn><eos>"""
        )
    return prompt_list
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
    learning_rate=2e-4,
    group_by_length=True,
    bf16=True,
    report_to="wandb",
    run_name="gemma-2-9b-lora-bf16-0926",
)

# Trainer 초기화 및 훈련
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    max_seq_length = 512,
    args=training_args,
    formatting_func=generate_prompts,
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