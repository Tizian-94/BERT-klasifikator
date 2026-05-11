import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib

MODEL_PATH = "models/final_model"

def load_model(model_dir=MODEL_PATH):
    if not os.path.isabs(model_dir):
        pass
    # load tokenizer, model i label encoder s diska
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    return tokenizer, model, label_encoder

def predict(text, tokenizer, model, label_encoder, max_len=256):
    inputs = tokenizer(
        text,
        truncation = True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return label_encoder.inverse_transform([pred_id])[0]

def main():
    parser = argparse.ArgumentParser(description="Predict fixed asset accounting code.")
    parser.add_argument("text", type=str, help="Asset description (EG. 'Izrada projekta za zgradu')")
    args = parser.parse_args()

    tokenizer, model, le = load_model()
    prediction = predict (args.text, tokenizer, model, le)
    print(f"Input: {args.text}")
    print(f"Predicted OznakaKonta: {prediction}")

if __name__ == "__main__":
    main()