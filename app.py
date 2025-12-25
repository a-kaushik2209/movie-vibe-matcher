import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Cinematch AI", page_icon="🎬", layout="wide")

st.title("CineMatch: The 'Vibe' Search")
st.markdown("Don't search by keywords. **Search by plot.**")
st.markdown("*Example: 'A corporate guy gets forced into a life of crime'*")

@st.cache_resource
def load_resources():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    data_parts = []
    part_num = 0
    while True:
        try:
            with open(f'movie_db_part_{part_num}.pkl', 'rb') as f:
                data_parts.append(f.read())
            part_num += 1
        except FileNotFoundError:
            break
            
    if not data_parts:
        return None, None, None
        
    full_data_bytes = b"".join(data_parts)
    data = pickle.loads(full_data_bytes)
        
    return model, data['embeddings'], data['metadata']

try:
    model, embeddings, metadata = load_resources()
    if model is None:
        st.error("Error: Could not find movie_db_part_0.pkl. Did you upload the files?")
        st.stop()
    else:
        st.success("System Loaded!")
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

query = st.text_input("Describe the story you want to see:")

if query:
    query_vec = model.encode([query])
    
    sim_scores = cosine_similarity(query_vec, embeddings)[0]
    
    top_indices = sim_scores.argsort()[-15:][::-1]
    
    st.divider()
    st.subheader("Results")
    
    seen_movies = set()
    count = 0
    
    for idx in top_indices:
        meta = metadata[idx]
        title = meta['Title']
        score = sim_scores[idx]
        
        if title in seen_movies: continue
        seen_movies.add(title)
        count += 1
        
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.metric(label="Match", value=f"{int(score*100)}%")
            with col2:
                st.subheader(f"{title} ({meta['Year']})")
                st.caption(f"Genre: {meta['Genre']}")
                with st.expander("See why it matched (Spoiler)"):
                    st.write(f"...{meta['Text']}...")
            st.divider()
            
        if count >= 5: break