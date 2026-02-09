import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "./models/bert-ag-news"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

label_names = ["World", "Sports", "Business", "Sci/Tech"]


def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    return {label_names[i]: float(probs[i]) for i in range(len(label_names))}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Enter news text"),
    outputs=gr.Label(label="Predictions"),
    title="News Classifier",
    description="Bert Fine-Tuned on AG News dataset",
)
demo.launch()
