# Transformers
## Project Structure

```
TransformersPractice/
├── models/
│   └── bert-ag-news/          # Fine-tuned BERT model (not in git)
├── src/
│   ├── tokenization.ipynb     # Day 1: Tokenization basics
│   ├── hidden_states.ipynb    # Day 2: Extracting hidden states
│   ├── attention.ipynb        # Day 3: Attention visualization
│   ├── classification.ipynb   # Day 4: Baseline (frozen BERT + LogReg)
│   ├── finetuning.ipynb       # Day 5: Fine-tuning BERT
│   ├── inference.ipynb        # Day 6: Inference and comparison
│   ├── error_analysis.ipynb   # Day 7: Error analysis
│   └── app.py                 # Day 7: Gradio demo app
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/TransformersPractice.git
cd TransformersPractice

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers datasets scikit-learn pandas matplotlib seaborn gradio
```

## Results

Text classification on AG News dataset (4 classes: World, Sports, Business, Sci/Tech):

| Model | Accuracy |
|-------|----------|
| Baseline (frozen BERT + LogReg) | 86% |
| Fine-tuned BERT | 91% |

## Training the Model

The `models/` folder is not included in git (too large). To reproduce the fine-tuned model:

1. Open `src/finetuning.ipynb` in Google Colab
2. Enable GPU: Runtime -> Change runtime type -> T4 GPU
3. Run all cells
4. Save and download the model:

```python
model.save_pretrained("./bert-ag-news")
tokenizer.save_pretrained("./bert-ag-news")

!zip -r bert-ag-news.zip bert-ag-news
from google.colab import files
files.download("bert-ag-news.zip")
```

5. Unzip and place in `models/bert-ag-news/`

## Run the Demo App

```bash
python src/app.py
```

Opens a web interface at `http://localhost:7860` where you can classify news articles.
# bert-news-classifier
