import re

def cleansing(text: str) -> str:
    if not text or str(text).strip() == '':
        return ''
    text = str(text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace('-', ' ')
    text = re.sub(r'[^a-zA-Z0-9\s\(\)]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def case_folding(text: str) -> str:
    return text.lower()

def stopword_removal(text: str, stopword_remover) -> str:
    return stopword_remover.remove(text)

def stemming(text: str, stemmer) -> str:
    tokens  = text.split()
    stemmed = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed)

def preprocess(text: str, stemmer, stopword_remover) -> str:
    text = cleansing(text)
    text = case_folding(text)
    text = stopword_removal(text, stopword_remover)
    text = stemming(text, stemmer)
    return text