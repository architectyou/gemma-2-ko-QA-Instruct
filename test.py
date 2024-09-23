# pip install bitsandbytes
import pdb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = "/data/test/0923_merged-2"
# model_id = "/data/gguf_models/ko-gemma-2-9b-it"
# quantization_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
# quantization_config_4bit = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # quantization_config=quantization_config_8bit,
    # quantization_config=quantization_config_4bit,
    # low_cpu_mem_usage=True,
)

model.eval()
# instruction = "서울의 유명한 관광 코스를 만들어줄래?"
prompt = """
<bos><bos><start_of_turn>user\nContext: 잔고증명서\n상법 제318조 제1항과 제3항은 납입금 보관자의 증명과 책임과 관련하여 다음과 같이 규정하고 있습니다:\n① 납입금을 보관한 은행이나 그 밖의 금융기관은 발기인 또는 이사의 청구를 받으면 그 보관금액에 관하여 증명서를 발급하여야 한다.\n③ 자본금 총액이 10억원 미만인 회사를 상법 제295조 제1항에 따라 발기설립하는 경우에는 제1항의 증명서를 은행이나 그 밖의 금융기관의 잔고증명서로 대체할 수 있다(상법 제318조 제3항, 상업등기법 제24조 제3항, 상업등기규칙 제129조 참조).\n한편, 상업등기법 제24조 제3항와 상업등기규칙 제129조는 자본금 총액이 10억원 미만인 주식회사가 발기설립을 위하여 등기신청을 하는 경우에는 "주금의 납입을 맡은 은행, 그 밖의 금융기관의 납입금 보관을 증명하는 정보" 대신 "은행이나 그 밖의 금융기관의 잔고를 증명하는 정보로 대체할 수 있다"고 규정하고 있습니다.\nQuestion: 발기인 A와 B는 로봇 청소기 제조 및 판매를 목적으로 하는 자본금 5억원인 갑 주식회사를 발기설립의 방법으로 설립하고 있습니다. 갑 주식회사는 설립 등기 신청을 하면서 주금의 납입을 맡은 은행의 납입금 보관 증명서 대신 은행의 잔고증명서를 제공하였습니다. 갑 주식회사의 설립등기신청은 적법한가요?<end_of_turn>\n<start_of_turn>model\nAnswer: 적법합니다.<end_of_turn><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
"""

messages = [
    {"role": "user", 
     "content": f"{prompt}"}
]

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
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    # repetition_penalty=1.2,
    # no_repeat_ngram_size=2
)

pdb.set_trace()
print(tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True))