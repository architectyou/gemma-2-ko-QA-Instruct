import os
import pdb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_id, origin_model):
    try:
        tokenizer = AutoTokenizer.from_pretrained(origin_model)
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
        prompt = f"Human: {instruction}\n\nAssistant: "
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        inputs = inputs.to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            num_return_sequences=1,
        )

        pdb.set_trace() # 토크나이저가 이상하다는 것은 확실히 알겠어...
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text.split("Assistant: ")[-1].strip()
        
        print(f"생성된 응답: {response}")

        return response
    except Exception as e:
        print(f"텍스트 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def test_tokenizer(tokenizer):
    test_text = "안녕하세요. 서울의 유명한 관광 코스를 알려주세요."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"원본 텍스트: {test_text}")
    print(f"인코딩된 토큰: {encoded}")
    print(f"디코딩된 텍스트: {decoded}")
    
    # 특수 토큰 확인
    print(f"\n특수 토큰:")
    print(f"PAD 토큰: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
    print(f"EOS 토큰: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    print(f"BOS 토큰: {tokenizer.bos_token}, ID: {tokenizer.bos_token_id}")
    print(f"UNK 토큰: {tokenizer.unk_token}, ID: {tokenizer.unk_token_id}")
    
    # 어휘 크기 확인
    print(f"\n어휘 크기: {len(tokenizer)}")

def generate_answer(model, tokenizer, context, question):
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=500, num_return_sequences=1, temperature=0.56)
        
    pdb.set_trace()
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    
    model_id = "/data/test/0923_merged"
    origin_model = "/data/gguf_models/ko-gemma-2-9b-it"
    
    tokenizer, model = load_model_and_tokenizer(model_id, origin_model)
    # orig_tokenizer, orig_model = load_model_and_tokenizer(origin_model)
    if tokenizer is None or model is None:
        return

    model.eval()

    # test_tokenizer(tokenizer)

    instruction = "서울의 유명한 관광 코스를 만들어줄래?"
    # response = generate_response(model, tokenizer, instruction)
    
    # # 테스트
    test_context = "상법 제321조 제2항은 회사 성립시의 발기인의 납입담보책임에 관하여 규정하고 있습니다."
    test_question = "회사성립의 경우 발기인의 납입담보책임에 관해서 설명해 주십시오."

    
    response = generate_answer(model, tokenizer, test_context, test_question)
    
    if response:
        print("생성된 응답:")
        print(response)
    else:
        print("응답 생성에 실패했습니다.")
        


if __name__ == "__main__":
    main()