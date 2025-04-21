# Main Orchestration Notebook Script
from abstractive_baseline import generate_summary
from extractive import extractive_summary
from hybrid import hybrid_summary
from evalmetrics import compute_rouge, compute_bertscore, compute_readability, compute_compression_ratio
from custom_rep_score import repetition_score

sample_text = """
Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be
natural, such as through variations in the solar cycle. But since the 1800s, human activities have
been the main driver of climate change, primarily due to burning fossil fuels like coal, oil and
gas. Burning these materials releases what are called greenhouse gases into Earth's atmosphere.
These emissions act like a blanket wrapped around the Earth, trapping the sun's heat and raising
temperatures. Examples of greenhouse gases include carbon dioxide and methane. Climate change has
many impacts on the environment and human health. It leads to more extreme weather, rising sea
levels, and biodiversity loss. To reduce the worst effects of climate change, urgent and
transformative action is needed to cut emissions, switch to renewable energy, and protect natural
ecosystems. 
"""

print("===== BASELINE: Extractive (TextRank) =====")
extractive_summary = extractive_summary(sample_text)
print(extractive_summary)
print("\n--- METRICS ---")
print("ROUGE:", compute_rouge(sample_text, extractive_summary))
print("BERTScore:", compute_bertscore([extractive_summary], [sample_text]))
print("Readability:", compute_readability(extractive_summary))
print("Compression Ratio:", compute_compression_ratio(sample_text, extractive_summary))
print("Repetition Score:", repetition_score(extractive_summary))

print("\n===== BASELINE: Abstractive (PEGASUS-XSum) =====")
xsum_abstractive = generate_summary(sample_text, model_key="xsum")
print(xsum_abstractive)
print("\n--- METRICS ---")
print("ROUGE:", compute_rouge(sample_text, xsum_abstractive))
print("BERTScore:", compute_bertscore([xsum_abstractive], [sample_text]))
print("Readability:", compute_readability(xsum_abstractive))
print("Compression Ratio:", compute_compression_ratio(sample_text, xsum_abstractive))
print("Repetition Score:", repetition_score(xsum_abstractive))

print("\n===== BASELINE: Abstractive (PEGASUS-CNN/DailyMail) =====")
cnn_abstractive = generate_summary(sample_text, model_key="cnn_dailymail")
print(cnn_abstractive)
print("\n--- METRICS ---")
print("ROUGE:", compute_rouge(sample_text, cnn_abstractive))
print("BERTScore:", compute_bertscore([cnn_abstractive], [sample_text]))
print("Readability:", compute_readability(cnn_abstractive))
print("Compression Ratio:", compute_compression_ratio(sample_text, cnn_abstractive))
print("Repetition Score:", repetition_score(cnn_abstractive))

print("\n===== HYBRID (Extractive → Fine-tuned PEGASUS) =====")
hybrid_output = hybrid_summary(sample_text)
print(hybrid_output)
print("\n--- METRICS ---")
print("ROUGE:", compute_rouge(sample_text, hybrid_output))
print("BERTScore:", compute_bertscore([hybrid_output], [sample_text]))
print("Readability:", compute_readability(hybrid_output))
print("Compression Ratio:", compute_compression_ratio(sample_text, hybrid_output))
print("Repetition Score:", repetition_score(hybrid_output))