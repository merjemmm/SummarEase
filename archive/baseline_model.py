from all_import import *

def evaluate_summary(original, summary):
    print("\n--- Evaluation Scores ---")

    # 1. ROUGE Scores
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(original, summary)
    print("ROUGE-1:", rouge_scores['rouge1'].fmeasure)
    print("ROUGE-2:", rouge_scores['rouge2'].fmeasure)
    print("ROUGE-L:", rouge_scores['rougeL'].fmeasure)

    # 2. BERTScore
    _, _, F1 = bert_score.score([summary], [original], lang="en", verbose=False)
    print("BERTScore (F1):", F1[0].item())

    # 3. Readability Scores
    print("Flesch Reading Ease:", textstat.flesch_reading_ease(summary))
    print("SMOG Index:", textstat.smog_index(summary))

    # 4. Compression Ratio
    ratio = len(original.split()) / len(summary.split())
    print("Compression Ratio:", round(ratio, 2))

# Sample article text (climate change example)
sample_text = """
Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural,
such as through variations in the solar cycle. But since the 1800s, human activities have been the main driver
of climate change, primarily due to burning fossil fuels like coal, oil and gas. Burning these materials releases
what are called greenhouse gases into Earth’s atmosphere. These emissions act like a blanket wrapped around the Earth,
trapping the sun’s heat and raising temperatures. Examples of greenhouse gases include carbon dioxide and methane.
Climate change has many impacts on the environment and human health. It leads to more extreme weather, rising sea levels,
and biodiversity loss. To reduce the worst effects of climate change, urgent and transformative action is needed to cut emissions,
switch to renewable energy, and protect natural ecosystems.
"""

def self_repetition_score(summary, n=3):
    tokens = summary.split()
    ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)

    m = sum(count - 1 for count in counts.values() if count > 1)

    return math.log(m + 1)

# Extractive summary (top 3 sentences)
def extractive_summary(text, sentence_count=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return " ".join(str(sentence) for sentence in summary)

def abstractive_summary(text):
    model_name = "google/pegasus-cnn_dailymail"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)


    inputs = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=260,
        min_length=60,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Run the summary

def main():

    #extractive
    extract_summary = extractive_summary(sample_text, sentence_count=3)
    
    evaluate_summary(sample_text, extract_summary)

    #abstractive
    abstract_summary = abstractive_summary(sample_text)

    evaluate_summary(sample_text, abstract_summary)
