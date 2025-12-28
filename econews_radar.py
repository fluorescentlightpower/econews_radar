"""
EEcoNews Radar - это NLP-движок на базе EnvironmentalBERT, предназначенный для
автоматического выявления экологически значимых новостей в больших потоках медиа-контента.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================
# Загрузка модели и токенизатора
# ============================

MODEL = "ESGBERT/EnvironmentalBERT-environmental"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()

# ============================
# Параметры sliding-window
# ============================

MAX_LEN = 512
STRIDE = 128
THRESHOLD = 0.70


# ============================
# Фрагменты по token
# ============================

def split_into_chunks(text):
    """
    Разбивает длинный текст на перекрывающиеся фрагменты токенов.
    Используется для корректной обработки документов длиннее 512 токенов.
    """
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]

    chunks = []
    start = 0

    while start < len(tokens):
        part = tokens[start:start + (MAX_LEN - 2)]
        chunks.append(part)
        start += (MAX_LEN - 2) - STRIDE

    return chunks


# ============================
# OR-логика, классификация ESG
# ============================

def classify_news_or_logic(text):
    """
    Классифицирует длинную новость.
    Если хотя бы один фрагмент имеет eco_prob >= THRESHOLD,
    вся новость считается экологической.
    """
    chunks = split_into_chunks(text)

    for ch in chunks:

        # добавляем CLS/SEP и делаем padding до MAX_LEN
        input_ids = tokenizer.build_inputs_with_special_tokens(ch)
        input_ids = input_ids[:MAX_LEN]

        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        input_ids = input_ids + [pad_id] * (MAX_LEN - len(input_ids))
        attention_mask = [1 if t != pad_id else 0 for t in input_ids]

        inputs = {
            "input_ids": torch.tensor([input_ids]),
            "attention_mask": torch.tensor([attention_mask])
        }

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        eco_prob = probs[1].item()

        if eco_prob >= THRESHOLD:
            return {
                "is_environmental": True,
                "confidence": eco_prob,
                "trigger_chunk": tokenizer.decode(ch[:300]) + "..."
            }

    return {
        "is_environmental": False,
        "confidence": float(probs[1].item()),
        "trigger_chunk": None
    }