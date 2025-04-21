from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import torch

model_map = {
    "xsum": "google/pegasus-xsum",
    "cnn_dailymail": "google/pegasus-cnn_dailymail"
}

def load_pegasus_model(model_key):
    assert model_key in model_map, f"Unsupported model: {model_key}"
    model_name = model_map[model_key]
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def generate_summary(text, model_key="cnn_dailymail", max_len=120, min_len=60):
    tokenizer, model = load_pegasus_model(model_key)
    inputs = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_len,
            min_length=min_len,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
