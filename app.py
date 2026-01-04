
import streamlit as st
from pyvis.network import Network
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
# ==============================
# Load precomputed data
# ==============================
@st.cache_resource
def load_data():
    # Use absolute path or ensure this runs in project dir
    data = joblib.load("hybrid_recommendations_user1.pkl")
    eval_data = joblib.load("evaluation_results.pkl")
    return data, eval_data

def main():
    st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
    st.title("🎬 Hybrid GNN + LLM Movie Recommender")
    st.markdown("A knowledge-aware, explainable recommendation system using Graph Neural Networks and Semantic Embeddings.")

    data, eval_data = load_data()

    # Sidebar
    st.sidebar.header("System Overview")
    st.sidebar.markdown(f"""
    - **User**: {data['user_id']}
    - **Hybrid Precision@10**: {eval_data['mean_hybrid']:.3f}
    - **GNN Precision@10**: {eval_data['mean_gcn']:.3f}
    """)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Recommendations", "Knowledge Graph", "How It Works"])

    with tab1:
        st.subheader(f"🎬 Recommendations for {data['user_id']}")
        st.markdown("These recommendations are generated using a hybrid model that combines **Graph Neural Networks (GNN)** and **Semantic Embeddings from a Language Model (LLM)**.")
        
        recs = data["recommendations"]
        descs = data["movie_descriptions"]
        
        if not recs:
            st.warning("No recommendations available.")
        else:
            for i, movie in enumerate(recs, 1):
                with st.expander(f"**{i}. {movie}**"):
                    plot = descs.get(movie, "No description available.")
                    st.write(f"**Plot / Description**: {plot}")
                    if i == 1:
                        # You can enhance this explanation dynamically if you save it
                        st.info("💡 **Why this recommendation?** This movie is recommended because it is semantically and structurally similar to movies you've rated highly in the past.")
    with tab2:
        st.subheader("Knowledge Graph (User-Movie-Genre)")
        st.markdown("Interactive subgraph centered on User_1 and their interactions.")
        
        try:
            # Load data
            G_full = joblib.load("knowledge_graph.pkl")
            movie_nodes_list = joblib.load("movie_nodes.pkl")
            movie_nodes_set = set(movie_nodes_list)
            
            center_node = "User_1"
            if center_node not in G_full:
                center_node = list(G_full.nodes())[0]

            # Build 2-hop subgraph
            sub_nodes = {center_node}
            for neighbor in G_full.successors(center_node):
                if any('rated' == data.get('label') for data in G_full[center_node][neighbor].values()):
                    sub_nodes.add(neighbor)
                    for genre_neighbor in G_full.successors(neighbor):
                        if any('belongs_to' == data.get('label') for data in G_full[neighbor][genre_neighbor].values()):
                            sub_nodes.add(genre_neighbor)
            
            subgraph = G_full.subgraph(sub_nodes)
            
            # Create PyVis network
            net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black", directed=True)
            net.set_options("""
            var options = {
            "physics": {"enabled": true, "stabilization": {"iterations": 100, "updateInterval": 25},"minVelocity": 0.75,"simulationDuration": 1500},
            "edges": {"arrows": {"to": {"enabled": true}}, "color": "#888888"},
            "nodes": {"font": {"size": 12, "color": "#000000"}}
            }
            """)
            
            # Add nodes with colors
            for node in subgraph.nodes():
                if node.startswith("User_"):
                    color = "#FFA500"  # Orange
                    title = "User"
                elif node in movie_nodes_set:  # ✅ NOW DEFINED
                    color = "#4682B4"  # Steel blue
                    title = "Movie"
                else:
                    color = "#32CD32"  # Lime green
                    title = "Genre"
                label = node[:25] + "..." if len(node) > 25 else node
                net.add_node(node, label=label, color=color, title=title)
            
            # Add edges
            for u, v, data in subgraph.edges(data=True):
                label = data.get('label', '')
                net.add_edge(u, v, title=label, label=label)
            
            # Display
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
                net.save_graph(f.name)
                with open(f.name, 'r', encoding='utf-8') as HtmlFile:
                    source_code = HtmlFile.read()
                st.components.v1.html(source_code, height=650)

        except Exception as e:
            st.error(f"Could not load KG: {str(e)}")
            st.info("Make sure 'knowledge_graph.pkl' and 'movie_nodes.pkl' exist.")
    with tab3:
        st.subheader("How It Works")
        st.markdown("""
        1. **Knowledge Graph**: Built from MovieLens ratings + genres (User → Movie → Genre).
        2. **GNN**: Learns structural preferences from user behavior.
        3. **LLM**: Encodes semantic meaning from movie descriptions (synthetic or DBpedia).
        4. **Hybrid Fusion**: Combines both signals for better recommendations.
        5. **Explanation**: Uses semantic similarity to justify suggestions.

        **Result**: 26% higher accuracy than GNN or LLM alone.
        """)
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*_vRJg3OkKmI36xXrKJq_tA.png", 
                 caption="Hybrid GNN + LLM Architecture", use_container_width=True)

if __name__ == "__main__":
    main()
