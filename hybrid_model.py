from all_import import *
from baseline_model import evaluate_summary
from data import tokenize_function
from transformers import Seq2SeqTrainingArguments, TfidfVectorizer, pipeline
from transformers import PegasusForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_from_disk
from sklearn.metrics.pairwise import cosine_similarity

combined_dataset = load_from_disk("./combined_summary_dataset")
final_dataset = {}

def get_split_data():

    tokenized_dataset = combined_dataset.map(tokenize_function, batched=True, remove_columns=combined_dataset.column_names)

    tokenized_dataset = tokenized_dataset.select(range(300))
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
    val_test = split_dataset['test'].train_test_split(test_size=0.5)

    final_dataset['train'] = split_dataset['train']
    final_dataset['test'] = val_test['test']
    final_dataset['validation'] = val_test['train']
    # final_dataset = {
    #     'train': split_dataset['train'],
    #     'validation': val_test['train'],
    #     'test': val_test['test']
    # }

def extract_top_sentences(text, top_k=3):
    sentences = text.split('. ')
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)
    scores = cosine_similarity(tfidf[0:1], tfidf).flatten()
    ranked = np.argsort(scores)[::-1][:top_k]

    return '. '.join([sentences[i] for i in sorted(ranked)])

# Abstractive summarizer using Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def hybrid_summarize(text, num_extract=3):
    extractive = extract_top_sentences(text, top_k=num_extract)
    result = summarizer(extractive, max_length=100, min_length=25, do_sample=False)
    return result[0]['summary_text']


def main():
    results = []

    for sample in final_dataset['test']:
        original = sample['input_text']
        reference = sample['reference_summary']

        # Hybrid summary
        generated_summary = hybrid_summarize(original)

        # Evaluate
        scores = evaluate_summary(original, generated_summary)

        results.append({
            'reference': reference,
            'generated': generated_summary,
            'rouge1': scores['rouge1'],
            'rouge2': scores['rouge2'],
            'rougeL': scores['rougeL'],
            'bertscore': scores['bertscore'],
            'flesch': scores['flesch'],
            'smog': scores['smog'],
            'compression': scores['compression'],
            'domain': sample['input_text'].split()[0]  # First token is the domain tag
        })

    results = pd.DataFrame(results)
    results.to_csv("./final_results.csv")

    # also the training model

    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail")


    training_args = TrainingArguments(
        output_dir="./pegasus-multi",
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        report_to="none",
        # predict_with_generate=True,
        logging_dir="./logs",
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        # predict_with_generate=True,
        train_dataset=final_dataset["train"],
        eval_dataset=final_dataset["validation"]
    )

    trainer.train()

    metrics = trainer.evaluate(eval_dataset=final_dataset["test"])
    print("📊 Test Set Metrics:\n", metrics)

    predictions = trainer.predict(test_dataset=final_dataset["test"])

    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
    decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)



    return 0


