import json
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import os

# Load data
def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

train_data = load_jsonl("data/train.jsonl")
val_data = load_jsonl("data/val.jsonl")
test_data = load_jsonl("data/test.jsonl")

# Preprocess: postText + targetParagraphs → spoiler
def preprocess(data, is_test=False):
    inputs, targets, ids = [], [], []
    for item in data:
        post = item['postText'][0]
        para = " ".join(item['targetParagraphs'])[:1024]
        text_input = f"generate spoiler: {post} </s> {para}"
        inputs.append(text_input)
        ids.append(item["id"])
        if not is_test:
            targets.append(item['spoiler'][0])
    if is_test:
        return pd.DataFrame({'id': ids, 'input': inputs})
    return pd.DataFrame({'input': inputs, 'target': targets})

train_df = preprocess(train_data)
val_df = preprocess(val_data)
test_df = preprocess(test_data, is_test=True)

# Tokenization
model_name = "t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def tokenize_function(example):
    model_inputs = tokenizer(example["input"], max_length=512, truncation=True, padding="max_length")
    if "target" in example:
        labels = tokenizer(example["target"], max_length=64, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_ds = Dataset.from_pandas(train_df)
val_ds = Dataset.from_pandas(val_df)
tokenized_train = train_ds.map(tokenize_function, batched=True)
tokenized_val = val_ds.map(tokenize_function, batched=True)

# Training setup
training_args = TrainingArguments(
    output_dir="./t5_spoiler",
    evaluation_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="no",
    logging_dir="./logs",
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# Inference
test_inputs = tokenizer(test_df["input"].tolist(), return_tensors="pt", padding=True, truncation=True, max_length=512)
test_outputs = model.generate(
    input_ids=test_inputs["input_ids"],
    attention_mask=test_inputs["attention_mask"],
    max_new_tokens=64
)
decoded = tokenizer.batch_decode(test_outputs, skip_special_tokens=True)

# Save results
os.makedirs("results", exist_ok=True)
pd.DataFrame({
    "id": test_df["id"],
    "spoiler": decoded
}).to_csv("results/task2_output.csv", index=False)
print("Task 2 complete → results/task2_output.csv")