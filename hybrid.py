from datasets import load_dataset, concatenate_datasets
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from extractive import extractive_summary
from preprocessing import normalize_and_merge
import os


model_name = "google/pegasus-cnn_dailymail"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

dataset = normalize_and_merge()

splits = dataset.train_test_split(test_size=0.2, seed=42)
train_val_split = splits["train"].train_test_split(test_size=0.1, seed=42)
train_dataset_raw = train_val_split["train"]
val_dataset_raw = train_val_split["test"]
test_dataset_raw = splits["test"]

def tokenize(example):
    model_inputs = tokenizer(example["article"], truncation=True, padding="max_length", max_length=512)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["summary"], truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset_raw.map(tokenize, batched=True, remove_columns=train_dataset_raw.column_names)
tokenized_val = val_dataset_raw.map(tokenize, batched=True, remove_columns=val_dataset_raw.column_names)
tokenized_test = test_dataset_raw.map(tokenize, batched=True, remove_columns=test_dataset_raw.column_names)

training_args = Seq2SeqTrainingArguments(
    output_dir="pegasus_hybrid_model",
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    num_train_epochs=1,
    logging_dir="logs",
    save_total_limit=1,
    save_steps=1000,
    eval_steps=500,
    predict_with_generate=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("pegasus_hybrid_model")
tokenizer.save_pretrained("pegasus_hybrid_model")

metrics = trainer.evaluate(tokenized_test)
print("\n--- Final Test Set Evaluation ---")
print(metrics)

def hybrid_summary(text, extractive_ratio=1.0):
    extractive = extractive_summary(text)

    if 0 < extractive_ratio < 1.0:
        words = extractive.split()
        cutoff = max(1, int(len(words) * extractive_ratio))
        truncated_extractive = " ".join(words[:cutoff])
    else:
        truncated_extractive = extractive

    inputs = tokenizer(truncated_extractive, return_tensors="pt", truncation=True, padding="longest")
    with torch.no_grad():
        summary_ids = model.generate(
            inputs.input_ids,
            max_length=120,
            min_length=60,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)