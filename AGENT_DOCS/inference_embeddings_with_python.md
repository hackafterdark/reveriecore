# Local Inference: BART & Sentence-Transformers

ReverieCore performs all semantic intelligence locally. We avoid the simplified `pipeline()` API from Hugging Face in favor of direct `AutoModel` loading. This provides greater control over lazy-loading and resource management.

## 1. Summarization (DistilBART)

We use `sshleifer/distilbart-cnn-12-6` for fast, CPU-efficient summarization.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Lazy loading implementation
tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")

def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=150, 
        min_length=20, 
        num_beams=2, 
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 2. Zero-Shot Classification (BART Large MNLI)

The core "brain" of our enrichment layer is `facebook/bart-large-mnli`. It handles both **Memory Classification** and **Importance Scoring**.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

def classify(text, labels):
    # Construct hypothesis for each label
    # e.g. "This text is about TASK"
    results = {}
    for label in labels:
        premise = text
        hypothesis = f"This text is about {label}."
        
        # Tokenize pair
        x = tokenizer.encode(premise, hypothesis, return_tensors='pt', truncation=True)
        logits = model(x)[0]
        
        # Convert to entailment probability
        # BART's 2nd logit [index 2] is 'entailment'
        entail_logit = logits[:, [0, 1, 2]]
        probs = F.softmax(entail_logit, dim=1)
        results[label] = probs[:, 2].item()
    
    return results
```

## 3. Vector Embeddings (all-MiniLM-L6-v2)

For vector search, we use the industry-standard `SentenceTransformer` which produces 384-dimension embeddings optimized for semantic similarity.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
vector = model.encode(["Your memory gist here"])[0]
# Returns a 384-length float array compatible with sqlite-vec
```