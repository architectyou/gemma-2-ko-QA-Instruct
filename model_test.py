import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_id):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        return tokenizer, model
    except Exception as e:
        print(f"모델 또는 토크나이저 로딩 중 오류 발생: {e}")
        return None, None

def generate_response(model, tokenizer, instruction):
    try:
        messages = [{"role": "user", "content": instruction}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<end_of_turn>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        generated_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return generated_text.strip()
    except Exception as e:
        print(f"텍스트 생성 중 오류 발생: {e}")
        return None
    
def generate_answer(model, tokenizer, context, question):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=500, num_return_sequences=1, temperature=0.56)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    model_id = "/data/test/0920_merged"
    
    origin_model = "/data/gguf_models/ko-gemma-2-9b-it"
    
    tokenizer, model = load_model_and_tokenizer(model_id)
    if tokenizer is None or model is None:
        return

    model.eval()
    
    instruction = "서울의 유명한 관광 코스를 만들어줄래?"
    response = generate_response(model, tokenizer, instruction)
    
    # # 테스트
    # test_context = "상법 제321조 제2항은 회사 성립시의 발기인의 납입담보책임에 관하여 규정하고 있습니다."
    # test_question = "회사성립의 경우 발기인의 납입담보책임에 관해서 설명해 주십시오."

    
    # response = generate_answer(model, tokenizer, test_context, test_question)
    
    if response:
        print("생성된 응답:")
        print(response)
    else:
        print("응답 생성에 실패했습니다.")
        


if __name__ == "__main__":
    main()