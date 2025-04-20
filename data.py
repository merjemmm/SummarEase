from datasets import load_dataset, concatenate_datasets
from all_import import *

def extractive_summary_h(text, num_sentences=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary_sentences)

def format_sample(example, source_tag, article_key, summary_key):
    article = example.get(article_key)
    summary = example.get(summary_key)
    return {
        "input_text": f"[{source_tag}] {article.strip() if article else ''}",
        "target_summary": summary.strip() if summary else ''
    }

def format_sample_hybrid(example, source_tag, article_key, summary_key):
    article = example.get(article_key)
    summary = example.get(summary_key)

    extractive = extractive_summary_h(article)
    return {
        "input_text": f"[{source_tag}] {extractive}",
        "target_summary": summary.strip() if summary else ''
    }

def tokenize_function(batch):
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")


    inputs = tokenizer(batch["input_text"], truncation=True, padding="max_length", max_length=512)
    targets = tokenizer(batch["target_summary"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

def main():
    
    cnn = load_dataset("cnn_dailymail", "3.0.0", split='train[:10%]')
    pubmed = load_dataset("ccdv/pubmed-summarization", split='train[:10%]')
    xsum = load_dataset("xsum", split="train[:10%]")
    arxiv = load_dataset("ccdv/arxiv-summarization", split="train[:10%]")
    print("CNN/DailyMail:", len(cnn))
    print("PubMed:", len(pubmed))
    print("XSum:", len(xsum))
    print("arXiv:", len(arxiv))


    # adding tags for domain-aware learning
    cnn = cnn.map(lambda x: format_sample_hybrid(x, "CNN", "article", "highlights"), remove_columns=cnn.column_names)
    pubmed = pubmed.map(lambda x: format_sample_hybrid(x, "PUBMED", "article", "abstract"), remove_columns=pubmed.column_names)
    xsum = xsum.map(lambda x: format_sample_hybrid(x, "XSUM", "document", "summary"), remove_columns=xsum.column_names)
    arxiv = arxiv.map(lambda x: format_sample_hybrid(x, "ARXIV", "article", "abstract"), remove_columns=arxiv.column_names)

    combined_dataset = concatenate_datasets([cnn, pubmed, xsum, arxiv])
    combined_dataset.save_to_disk("./combined_summary_dataset")

    tokenized_dataset = combined_dataset.map(tokenize_function, batched=True, remove_columns=combined_dataset.column_names)

    tokenized_dataset = tokenized_dataset.select(range(300))
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
    val_test = split_dataset['test'].train_test_split(test_size=0.5)

    final_dataset = {
        'train': split_dataset['train'],
        'validation': val_test['train'],
        'test': val_test['test']
    }

    # split_dataset = combined_dataset.train_test_split(test_size=0.1)
    # train_data = split_dataset["train"]
    # test_data = split_dataset["test"]

    # tokenized_train = train_data.map(tokenize_function, batched=True, remove_columns=train_data.column_names)
    # tokenized_test = test_data.map(tokenize_function, batched=True, remove_columns=test_data.column_names)

    # tokenized_train.save_to_disk("/content/drive/MyDrive/tokenized_hybrid_train")
    # tokenized_test.save_to_disk("/content/drive/MyDrive/tokenized_hybrid_test")