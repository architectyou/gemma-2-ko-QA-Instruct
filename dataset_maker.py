import PyPDF2
import json, os
import re
import openai
from openai import OpenAI
from tqdm import tqdm
from time import time
from dotenv import load_dotenv

load_dotenv()

# OpenAI API 키 설정
token = os.getenv("GITHUB_TOKEN")
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o"

client = OpenAI(
    base_url = endpoint,
    api_key = token,
)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def split_text_into_chunks(text, max_chunk_size=2000):
    chunks = []
    current_chunk = ""
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_qa_pair(chunk, max_retries=1):
    prompt = f"""Given the following legal text, generate a question-answer pair. 
    The answer should be factual and based solely on the information provided in the text. 
    Also, identify any specific legal reference (e.g., article number, law name) mentioned in the text.
    Please Answer in Korean.

    Text: {chunk}

    Provide the output in the following JSON format:
    {{
        "question": "Generated question",
        "answer": "Generated answer",
        "reference_rule": "Identified legal reference or null if not found",
        "reference_file" : "Identified reference file or null if not found"
    }}
    """

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.56,
                max_tokens=600
            )
            content = response.choices[0].message.content.strip()
            
            print(f"API Response (Attempt {attempt + 1}):")
            print(content)
            print("Response type:", type(content))
            print("Response length:", len(content))
            
            if not content:
                print("Empty response received")
                if attempt == max_retries - 1:
                    return {
                        "question": "Failed to generate question",
                        "answer": "Failed to generate answer",
                        "reference_rule": None,
                        "reference_file" : None
                    }
                continue
            
            try:
                json_str = re.search(r'\{.*\}', content, re.DOTALL)
                if json_str:
                    parsed_content = json.loads(json_str.group())
                    # 필요한 키만 추출
                    return {
                        "question": parsed_content.get("question", "Failed to parse question"),
                        "answer": parsed_content.get("answer", "Failed to parse answer"),
                        "reference_rule": parsed_content.get("reference_rule"),
                        "reference_file" : parsed_content.get("reference_file")
                    }
                else:
                    raise ValueError("No JSON object found in the response")
                
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                if attempt == max_retries - 1:
                    # JSON 파싱에 실패하면 수동으로 파싱 시도
                    question = re.search(r'"question":\s*"(.*?)"', content, re.DOTALL)
                    answer = re.search(r'"answer":\s*"(.*?)"', content, re.DOTALL)
                    reference = re.search(r'"reference_rule":\s*"(.*?)"', content, re.DOTALL)
                    return {
                        "question": question.group(1) if question else "Failed to parse question",
                        "answer": answer.group(1) if answer else "Failed to parse answer",
                        "reference_rule": reference.group(1) if reference else None
                    }
        except Exception as e:
            print(f"Error during API call: {e}")
            if attempt == max_retries - 1:
                raise
        
        time.sleep(2)

def generate_qa_pairs(text):
    chunks = split_text_into_chunks(text)
    qa_pairs = []
    
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Generating QA pairs"):
        try:
            qa = generate_qa_pair(chunk)
            qa_pair = {
                "id": f"QA_{i:05d}",
                "question": qa['question'],
                "answer": qa['answer'],
                "context": chunk,
                "reference_rule": qa['reference_rule'],
                "reference_file" : qa['reference_file']
            }
            qa_pairs.append(qa_pair)
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
    
    return qa_pairs


def save_qa_pairs_to_json(qa_pairs, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    
    pdf_path = "/data/test/pdf_files/표준 개인정보 보호지침(개인정보보호위원회고시)(제2024-1호)(20240104).pdf"
    output_path = "output_data/test_qa_pairs.json"

    # PDF에서 텍스트 추출
    text = extract_text_from_pdf(pdf_path)

    # QA 쌍 생성
    qa_pairs = generate_qa_pairs(text)

    # JSON 파일로 저장
    print(qa_pairs)
    save_qa_pairs_to_json(qa_pairs, output_path)

    print(f"Generated {len(qa_pairs)} QA pairs and saved to {output_path}")