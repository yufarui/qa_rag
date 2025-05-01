import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import src.constant as constant

model_path = constant.rerank_model_path

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path)
max_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict(query: str, extracted_contents: list[str]) -> list[str]:
    pairs = [(query, content) for content in extracted_contents]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=max_length).to(device)
    with torch.no_grad():
        scores = model(**inputs).logits
    scores = scores.detach().cpu().clone().numpy()
    response = [doc for score, doc in sorted(zip(scores, extracted_contents), reverse=True, key=lambda x: x[0])]
    return response
