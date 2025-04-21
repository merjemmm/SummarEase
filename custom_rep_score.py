from collections import Counter
from nltk.util import ngrams


def repetition_score(summary, n=3):
    tokens = summary.split()
    total_ngrams = list(ngrams(tokens, n))
    total = len(total_ngrams)
    counter = Counter(total_ngrams)
    repeated = sum(count - 1 for count in counter.values() if count > 1)
    unique = len(counter)
    return {
        'Total n-grams': total,
        'Unique n-grams': unique,
        'Repeated n-grams': repeated,
        'Repetition Score': repeated / total if total > 0 else 0
    }
