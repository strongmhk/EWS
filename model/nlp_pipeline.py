from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

def make_pipeline():    
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
    return nlp

def classification_text(text):
    result = nlp()