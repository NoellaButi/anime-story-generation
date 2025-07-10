# Anime Story Generation System

**CS495 Data Science Capstone – Spring 2025**  
**Team Members:** Noella Buti & Yuki Rivera  
**Institution:** Bellevue College – B.S. Computer Science (Data Science Emphasis)

---

## Overview

The Anime Story Generation System is a deep learning-based pipeline that generates original anime-style synopses from structured prompts (genre, theme, demographic). The project explores a variety of architectures and fine-tunes pre-trained models like BART, T5, and GPT-2 for conditional generation tasks.

---

## Objectives

- Build and evaluate a variety of neural architectures for conditional text generation
- Optimize the best-performing model (BART) based on:
  - **Perplexity (PPL)** – fluency
  - **Cosine Similarity (CS)** – semantic alignment
  - **Manual Inspection (MI)** – human-rated relevance and coherence

---

## Tech Stack

- Python, JupyterLab, Google Colab Pro  
- Hugging Face Transformers (GPT2, T5, BART)  
- Keras/TensorFlow, PyTorch  
- sentence-transformers, NumPy, pandas, matplotlib  
- UMAP for semantic projection  

---

## Models Directory

Located in `models/`:

- `bart/` – Final model with best performance  
- `basic_transformer_keras/` – Transformer from scratch (Keras)  
- `basic_transformer_pytorch/` – Transformer from scratch (PyTorch)  
- `gpt2/` – GPT-2 fine-tuning and evaluation  
- `t5/` – T5-small and T5-base runs  
- `lstm_1st_version/`, `lstm_2nd_version/` – Sequential LSTM models  
- `rnn_1st_version/`, `rnn_2nd_version/` – RNN baselines  
- `hybrid/` – GRU-RNN and GRU-LSTM combined models  

---

## Data Workflow

- Located in `cleaning_and_eda/`: Data cleaning, EDA, and demographic analysis  
- Located in `input_output_pair_creation/`: Prompt-synopsis pair generation using templates  

### Dataset

- **Top 15,000 Ranked Anime Dataset** (via Kaggle/MyAnimeList API)  
- ~10,000 clean entries after preprocessing  
- Used to create 15 prompt structures using genre, theme, demographic, title  

---

## Final Results (BART – Optimized Trial 6)

| Metric              | Value     |
|---------------------|-----------|
| Perplexity (PPL)    | 47.53     |
| Cosine Similarity   | 0.293     |
| Manual Score (MI)   | 52 / 80   |

> Tuned decoding parameters: `top_k=30`, `top_p=0.9`, `temperature=0.8`

---

## Plot & Evaluation Tools

Located in `plot_creation/`:

- `comparison_graph_creation.ipynb` – Model metric comparisons  
- `testing_graphs.ipynb` – Final BART performance plots  

---

## Team Contributions

### Noella Buti  
- Data cleaning & EDA (`cleaning_and_eda/`)  
- Feature engineering and selection  
- Built & evaluated: LSTM (v1), Hybrid, T5, GPT-2  
- Implemented cosine similarity & perplexity metrics  
- Report sections: Data Visualization, Feature Engineering  

### Yuki Rivera  
- Prompt-template generation (`input_output_pair_creation/`)  
- Built & optimized: RNN (v2), Transformer, BART  
- BART hyperparameter tuning and decoding refinement  
- Final report writing, model comparison graphs, GitHub structure  

---

## How to Run

```bash
# Example: run BART training or evaluation
cd models/bart/
open bart_final_training.ipynb

# For comparison graphs
cd plot_creation/
open comparison_graph_creation.ipynb
```
Recommended to run on Google Colab or a GPU-enabled environment

## Files

- models/ – All models tested and tuned
- plot_creation/ – Graph notebooks for final comparisons
- cleaning_and_eda/ – Dataset cleaning and EDA
- input_output_pair_creation/ – Prompt-synopsis pairing logic

## Contact

Noella Buti – noellabuti@gmail.com

Yuki Rivera – yukiko.rivera@bellevuecollege.edu
