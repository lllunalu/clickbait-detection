import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
import evaluate
import os

# Load data
def load_json(filename):
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f]

train_data = load_json("data/train.jsonl")
val_data = load_json("data/val.jsonl")
test_data = load_json("data/test.jsonl")

# Extract text and labels
train_texts = [item["postText"][0] for item in train_data]
train_labels = [item["tags"][0] for item in train_data]
val_texts = [item["postText"][0] for item in val_data]
val_labels = [item["tags"][0] for item in val_data]
test_texts = [item["postText"][0] for item in test_data]

# Label encoding
le = LabelEncoder()
train_labels_encoded = le.fit_transform(train_labels)
val_labels_encoded = le.transform(val_labels)

train_df = pd.DataFrame({"text": train_texts, "labels": train_labels_encoded})
val_df = pd.DataFrame({"text": val_texts, "labels": val_labels_encoded})
test_df = pd.DataFrame({"text": test_texts})

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)
test_ds = Dataset.from_pandas(test_df)

# Tokenization
checkpoint = "microsoft/deberta-v3-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize(example):
    return tokenizer(example["text"], truncation=True)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Model
class CustomDeberta(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(self.base.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.relu(self.dropout(pooled)))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return SequenceClassifierOutput(logits=logits, loss=loss)

# Trainer
num_labels = len(le.classes_)
model = CustomDeberta(checkpoint, num_labels)
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return f1.compute(predictions=preds, references=labels, average="macro")

training_args = TrainingArguments(
    output_dir="./deberta_output",
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="epoch",
    report_to="none"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

os.environ["WANDB_DISABLED"] = "true"
trainer.train()

# Prediction
preds = trainer.predict(test_ds)
pred_ids = np.argmax(preds.predictions, axis=1)
pred_labels = le.inverse_transform(pred_ids)

# Output
os.makedirs("results", exist_ok=True)
submission = pd.DataFrame({
    "id": [item["id"] for item in test_data],
    "spoilerType": pred_labels
})
submission.to_csv("results/task1_output.csv", index=False)
print("Task 1 complete → results/task1_output.csv")