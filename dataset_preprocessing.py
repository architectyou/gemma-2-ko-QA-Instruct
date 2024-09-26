from datasets import Dataset
import os, json, pdb
from sklearn.model_selection import train_test_split

# 데이터 로드 함수
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
    
if __name__=="__main__":

    data_directory = "./dataset/law_QA_dataset/"
    all_data = load_json_files(data_directory)

    train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42)

    train_dataset = create_dataset(train_data)
    val_dataset = create_dataset(val_data)
    pdb.set_trace()