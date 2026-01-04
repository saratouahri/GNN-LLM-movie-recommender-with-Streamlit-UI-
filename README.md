# Hybrid GNN + LLM Movie Recommender

An **explainable, knowledge-aware** recommendation system that fuses:
- **Graph Neural Networks (GNN)** over a user-movie-genre knowledge graph
- **Semantic embeddings** from a pretrained language model (Sentence-BERT)

✅ Achieves **26% higher Precision@10** than GNN or LLM baselines  
✅ Generates **natural language explanations**  
✅ Features **interactive Streamlit UI** with **knowledge graph visualization**

## 📸 Demo



## 🚀 Run the App

```bash
pip install -r requirements.txt
streamlit run app.py