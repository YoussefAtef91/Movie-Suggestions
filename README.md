# 🎬 Movie Recommendation System  

**A content-based movie recommender using semantic search with customizable filters**  

## 🔍 Overview  
This project recommends movies based on:  
- **Content similarity** (plot + keywords) using `all-mpnet-base-v2` embeddings  
- **Hybrid scoring** (customizable blend of similarity and popularity)  
- **Filters** (language, genre, rating thresholds)  

Built with TMDB dataset (top 100K movies) and hosted via Gradio.  

## 🚀 Try the Demo  
[![Gradio Demo](https://img.shields.io/badge/🤗-Gradio%20Demo-blue.svg)](https://huggingface.co/spaces/YoussefAtef91/MovieSuggestions)  

## 🛠️ Features  
### 🔧 Tunable Parameters  
| Control | Effect |  
|---------|--------|  
| **Description** | Finds movies with similar plots |  
| **Keywords** | Boosts genre/tag relevance |  
| **English Only** | Filters non-English content |  
| **Exclude Docs** | Removes documentaries |  
| **Popularity Bias** | Slider (0=pure similarity ↔ 1=popularity) |  
| **Min Rating** | Quality threshold (0-10 scale) |  

### 🧠 Under the Hood  
- **Embeddings**: `all-mpnet-base-v2` sentence transformer  
- **Dataset**: Top 100K TMDB movies  
- **Framework**: Gradio for UI  


## 📊 Performance Notes  
- Typical latency: <500ms on CPU  
- Handles ~100K movie corpus efficiently  
- Quality improves with detailed input descriptions  

## 📜 License  
[MIT](LICENSE)  

---

**Pro Tip**: For action movies, try keywords like ["explosion", "car chase", "martial arts"] 🔥  
