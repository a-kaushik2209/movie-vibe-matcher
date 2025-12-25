import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* IMPORT GOOGLE FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* RESET & BASE STYLES */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        color: #0F172A; /* Slate 900 */
        background-color: #F8FAFC; /* Slate 50 */
    }

    /* HIDE STREAMLIT BRANDING */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* CUSTOM HERO SECTION */
    .hero-container {
        text-align: center;
        padding: 4rem 0 2rem 0;
        animation: fadeIn 1s ease-in-out;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(45deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #64748B; /* Slate 500 */
        font-weight: 300;
    }

    /* SEARCH BAR STYLING */
    .stTextInput > div > div > input {
        padding: 15px 20px;
        font-size: 16px;
        border-radius: 12px;
        border: 2px solid #E2E8F0;
        background-color: #FFFFFF;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        color: #334155;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3B82F6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
    }

    /* CARD CONTAINER STYLING */
    /* We target the container of the cards */
    div[data-testid="stVerticalBlock"] > div[style*="background-color"] {
        background: white !important;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.025);
        border: 1px solid #F1F5F9;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    /* Hover effect for cards (Trickier in Streamlit, but possible via global selection) */
    div[data-testid="stVerticalBlock"] > div[style*="background-color"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }

    /* CARD TYPOGRAPHY */
    .movie-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 5px;
        line-height: 1.2;
    }
    .movie-meta {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
        margin-bottom: 15px;
    }
    .match-tag {
        background-color: #EFF6FF;
        color: #3B82F6;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 10px;
    }

    /* BUTTON STYLING */
    .stButton > button {
        background: linear-gradient(90deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 50px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s;
        width: auto;
        margin-top: 20px;
    }
    .stButton > button:hover {
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4);
        transform: scale(1.02);
    }

    /* ANIMATIONS */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* CUSTOM EXPANDER */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #475569;
        background-color: transparent;
    }
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
    model, embeddings, metadata = load_resources()
    if model is None:
        st.error("System Error: Database files missing.")
        st.stop()
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

st.markdown("""
<div class="hero-container">
    <div class="hero-title">CineMatch AI</div>
    <div class="hero-subtitle">Don't search by keywords. Search by the <b>vibe</b>.</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    query = st.text_input("", placeholder="Try: 'A corporate guy gets forced into a life of crime'...", label_visibility="collapsed")

if query != st.session_state.last_query:
    st.session_state.limit = 6
    st.session_state.last_query = query

if query:
    query_vec = model.encode([query])
    sim_scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = sim_scores.argsort()[-50:][::-1]
    
    st.markdown("<br>", unsafe_allow_html=True)

    cols = st.columns(3)
    
    seen_movies = set()
    count_displayed = 0
    
    for idx in top_indices:
        if count_displayed >= st.session_state.limit:
            break
            
        meta = metadata[idx]
        title = meta['Title']
        
        if title in seen_movies: continue
        seen_movies.add(title)

        col_idx = count_displayed % 3
        
        with cols[col_idx]:
            with st.container():
                st.markdown(f"""
                <div class="match-tag">Match: {int(sim_scores[idx]*100)}%</div>
                <div class="movie-title">{title}</div>
                <div class="movie-meta">{meta['Year']} • {meta['Genre']}</div>
                """, unsafe_allow_html=True)

                with st.expander("Why this matched"):
                    st.write(f"...{meta['Text']}...")
                    
        count_displayed += 1

    if count_displayed >= st.session_state.limit and st.session_state.limit < 40:
        st.markdown("<br><br>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns([1, 1, 1])
        with b2:
            if st.button("Load More Results", use_container_width=True):
                st.session_state.limit += 6
                st.rerun()

    if count_displayed == 0:
        st.markdown("<div style='text-align: center; color: #64748B;'>No movies found. Try a different description.</div>", unsafe_allow_html=True)

st.markdown("<br><br><br>", unsafe_allow_html=True)
