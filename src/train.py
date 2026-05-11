import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

# CONFIG
DATA_PATH = "./data/raw/osnovna_sredstva.csv"
MODEL_OUTPUT_DIR = "./models/final_model"
CHECKPOINT_DIR = "./models/results"
LOG_DIR = "./models/logs"
RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_LEN = 256
BATCH_SIZE = 4
EPOCHS = 1 # prebacit na 3 (6h)
LEARNING_RATE = 5e-5 #optimalno

class FixedAssetDataset(Dataset):
#dataset config
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation = True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
def compute_metrics(eval_pred):
#metrika
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {'accuracy': acc, 'f1':f1}

def main():
#main
    # load raw
    print("Loading data...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    print(f"Raw shape: {df.shape}")

    # dedupliciraj po IDu
    text_cols = ['NazivSredstva', 'OpisMjestaTroska']
    for col in text_cols:
        df[col] = df[col].fillna('')
    df['combined_text'] = (df['NazivSredstva'] + " " + df['OpisMjestaTroska']).str.replace(r'\s+',' ',regex=True).str.strip()

    #filtriranje po rare klasama
    temp_encoder = LabelEncoder()
    df['label_temp'] = temp_encoder.fit_transform(df['OznakaKonta'])
    class_counts = df['label_temp'].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df['label_temp'].isin(valid_classes)]
    print(f"After removing rare classes: {df.shape}")

    # final label encoding
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['OznakaKonta'])
    num_labels = df['label'].nunique()
    print(f"Number of classes: {num_labels}")

    # train split
    X = df['combined_text']
    y = df['label']
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")

    model_name = "classla/bcms-bertic"
    print(f"Loading tokenizer & model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # disable symlink upozorenje (samo za Windows masine)
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    #create pytorch dataset
    train_dataset = FixedAssetDataset(X_train, y_train, tokenizer, max_len=MAX_LEN)
    val_dataset = FixedAssetDataset(X_val, y_val, tokenizer, max_len=MAX_LEN)

    #training args
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # train start
    print("Starting training...")
    trainer.train()

    #eval
    eval_results = trainer.evaluate()
    print(f"Validation accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Validation F1: {eval_results['eval_f1']:.4f}")

    # save model
    print(f"Saving model to {MODEL_OUTPUT_DIR}")
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    joblib.dump(label_encoder, os.path.join(MODEL_OUTPUT_DIR, "label_encoder.pkl"))
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()