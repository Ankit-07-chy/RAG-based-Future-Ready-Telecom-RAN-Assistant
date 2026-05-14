import json
import pandas as pd
from pathlib import Path

def parse_teleqna():
    data_path = "data/teleqna"
    questions = []
    answers = []
    contexts = []
    
    for json_file in Path(data_path).glob("**/*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    question = item.get('question', '')
                    answer = item.get('answer', '')
                    explanation = item.get('explanation', '')
                    source = item.get('source', '')
                    
                    context = f"Question: {question}\nAnswer: {answer}\nExplanation: {explanation}\nSource: {source}"
                    
                    questions.append(question)
                    answers.append(answer)
                    contexts.append(context)
    
    df = pd.DataFrame({
        'question': questions,
        'answer': answers,
        'context': contexts
    })
    
    df.to_csv('data/processed/teleqna_processed.csv', index=False)
    print(f"✅ Processed {len(df)} Q&A pairs")
    return df

if __name__ == "__main__":
    parse_teleqna()