import streamlit as st
import pickle
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMatch AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    /* Style the Text Input */
    .stTextInput > div > div > input {
        padding: 12px 20px;
        border-radius: 25px;
        border: 1px solid #444;
        background-color: #1E1E1E;
        color: #fff;
    }
    .stTextInput > div > div > input:focus {
        border-color: #00ADB5;
        box-shadow: 0 0 5px rgba(0, 173, 181, 0.5);
    }

    /* Style the Buttons */
    .stButton > button {
        border-radius: 20px;
        background-color: #00ADB5;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #007F85;
        transform: scale(1.02);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }

    /* Card Styling */
    div[data-testid="stVerticalBlock"] > div[style*="background-color"] {
        border-radius: 15px;
        padding: 20px;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if 'limit' not in st.session_state:
    st.session_state.limit = 6
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

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
            
    if not data_parts: return None, None, None
        
    full_data = pickle.loads(b"".join(data_parts))
    return model, full_data['embeddings'], full_data['metadata']

try:
    with st.spinner("Initializing AI Engine..."):
        model, embeddings, metadata = load_resources()
    if model is None:
        st.error("System Error: Database files missing.")
        st.stop()
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: #00ADB5;'>CineMatch AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; margin-top: -15px;'>Semantic Plot Search Engine</p>", unsafe_allow_html=True)

query = st.text_input("", placeholder="Describe the plot... (e.g. 'A detective hunting a serial killer in the rain')", label_visibility="collapsed")

if query != st.session_state.last_query:
    st.session_state.limit = 6
    st.session_state.last_query = query

if query:
    query_vec = model.encode([query])
    sim_scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = sim_scores.argsort()[-50:][::-1]
    
    st.markdown("---")
    st.markdown(f"<h5 style='color: #666; margin-bottom: 20px;'>Top Matches</h5>", unsafe_allow_html=True)

    grid_cols = st.columns(2)
    
    seen_movies = set()
    count_displayed = 0
    
    for idx in top_indices:
        if count_displayed >= st.session_state.limit:
            break
            
        meta = metadata[idx]
        title = meta['Title']
        
        if title in seen_movies: continue
        seen_movies.add(title)

        col_idx = count_displayed % 2
        
        with grid_cols[col_idx]:
            with st.container():
                st.markdown(f"""
                <div style="
                    background-color: #262730;
                    padding: 20px;
                    border-radius: 12px;
                    margin-bottom: 20px;
                    border: 1px solid #333;
                ">
                    <h3 style="margin: 0; color: #EEE;">{title}</h3>
                    <p style="color: #00ADB5; font-size: 0.9em; margin-top: 5px;">{meta['Year']} • {meta['Genre']}</p>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("View Plot Match"):
                    st.write(f"...{meta['Text']}...")
        
        count_displayed += 1
        
    if count_displayed >= st.session_state.limit and st.session_state.limit < 40:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("Load More Results", use_container_width=True):
                st.session_state.limit += 6
                st.rerun()

    if count_displayed == 0:
        st.warning("No matches found. Try a different description.")
