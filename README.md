# Anime Story Generation System 🎬✨  
Deep learning pipeline to generate original **anime-style synopses** from structured prompts (genre, theme, demographic).  

![Language](https://img.shields.io/badge/language-Python-blue.svg) 
![Notebook](https://img.shields.io/badge/tool-Jupyter-orange.svg) 
![Framework](https://img.shields.io/badge/framework-Transformers-black.svg) 
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)  

---

## ✨ Overview  
This project was completed as part of the **CS495 Data Science Capstone (Spring 2025)** at Bellevue College.  
It explores multiple architectures (LSTM, RNN, GPT-2, T5, BART, Transformers from scratch) for **conditional text generation**.  

The final optimized model (**BART**) produces the most coherent results, evaluated by:  
- **Perplexity (PPL)** – fluency  
- **Cosine Similarity (CS)** – semantic alignment  
- **Manual Inspection (MI)** – human-rated relevance  

---

## 🛠️ Workflow  
- Cleaned and analyzed ~10,000 anime entries (genres, themes, demographics, titles)  
- Generated prompt–synopsis pairs using structured templates  
- Trained and compared multiple models (LSTM, Hybrid GRU, T5, GPT-2)  
- Evaluated using quantitative + qualitative metrics  
- Visualized comparisons and semantic projections with UMAP  

## 📁 Repository Layout 
```bash
models/                  # model runs (BART, GPT-2, T5, RNN, LSTM, Hybrid, Transformers)
cleaning_and_eda/        # dataset cleaning & EDA
input_output_pair_creation/ # prompt–synopsis templates
plot_creation/           # evaluation plots & comparisons
README.md
```

## 📊 Results (BART – Optimized Trial 6)

| Metric            | Value   |
|-------------------|---------|
| Accuracy (Manual Score) | 52 / 80 |
| Perplexity (PPL)  | 47.53   |
| Cosine Similarity | 0.293   |

Tuned decoding: `top_k=30`, `top_p=0.9`, `temperature=0.8`

## 👥 Team Contributions
This was a two-person group project.

**Project Lead – Yuki Rivera**
- Feature engineering
- Prompt–synopsis pair preparation
- Built & evaluated: RNN, LSTM (2nd version), Basic Transformer, BART

**Team Member – Noëlla Buti**
- Data analysis & visualization
- Feature engineering and selection
- Built & evaluated: LSTM (1st version), Hybrid (GRU/RNN, GRU/LSTM), T5, GPT-2

## ▶️ How to Run
Run in Google Colab or any GPU-enabled environment.
```bash
# Example: run BART training or evaluation
cd models/bart/
open bart_final_training.ipynb

# For comparison graphs
cd plot_creation/
open comparison_graph_creation.ipynb
```

## 🔮 Roadmap
- Experiment with larger pre-trained models (T5-Base, GPT-J)
- Expand datasets with character traits for richer prompts
- Integrate Stable Diffusion for optional visual generation

## 📜 License
MIT (see LICENSE)

---
