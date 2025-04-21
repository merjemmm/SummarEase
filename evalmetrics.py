from rouge_score import rouge_scorer
from bert_score import score as bert_score
import textstat


def compute_rouge(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return {
        'ROUGE-1': scores['rouge1'].fmeasure,
        'ROUGE-2': scores['rouge2'].fmeasure,
        'ROUGE-L': scores['rougeL'].fmeasure
    }


def compute_bertscore(cands, refs, lang="en"):
    P, R, F1 = bert_score(cands, refs, lang=lang, verbose=False)
    return F1.mean().item()


def compute_readability(text):
    return {
        'Flesch Reading Ease': textstat.flesch_reading_ease(text),
        'SMOG Index': textstat.smog_index(text)
    }


def compute_compression_ratio(original, summary):
    return len(original) / len(summary) if len(summary) > 0 else 0