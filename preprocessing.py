import re
import pandas as pd
from datasets import load_dataset


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n", " ", text)
    return text.strip()


# def load_cleaned_dataset(name="cnn_dailymail", split="test[:1%]", version="3.0.0"):
#     dataset = load_dataset(name, version, split=split)
#     articles = [clean_text(item['article']) for item in dataset]
#     summaries = [clean_text(item['highlights']) for item in dataset]
#     return pd.DataFrame({"article": articles, "summary": summaries})


def add_domain_tags(example, domain):
    tagged_article = f"<domain:{domain}> {example['article']}"
    return {"article": tagged_article, "summary": example["summary"]}


def normalize_and_merge():
    cnn = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
    pubmed = load_dataset("ccdv/pubmed-summarization", split="train[:1%]")
    xsum = load_dataset("xsum", split="train[:1%]", trust_remote_code=True)
    arxiv = load_dataset("ccdv/arxiv-summarization", split="train[:1%]")

    cnn = cnn.map(lambda x: {"article": x["article"], "summary": x["highlights"]})
    pubmed = pubmed.rename_columns({"article": "article", "abstract": "summary"})
    xsum = xsum.rename_columns({"document": "article", "summary": "summary"})
    arxiv = arxiv.rename_columns({"article": "article", "abstract": "summary"})

    cnn = cnn.map(lambda x: add_domain_tags(x, "news"))
    pubmed = pubmed.map(lambda x: add_domain_tags(x, "medical"))
    xsum = xsum.map(lambda x: add_domain_tags(x, "news"))
    arxiv = arxiv.map(lambda x: add_domain_tags(x, "academic"))

    return cnn, pubmed, xsum, arxiv