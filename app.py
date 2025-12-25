import streamlit as st
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMatch Database",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* BASE SETTINGS */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', Helvetica, Arial, sans-serif;
        background-color: #F8F8F8; /* Light grey background */
        color: #000000;
    }
    
    /* HIDE STREAMLIT CHROME */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* HEADER STYLES */
    .imdb-header {
        background-color: #121212; /* IMDb Black */
        padding: 15px 30px;
        border-radius: 8px;
        margin-bottom: 30px;
        display: flex;
        align-items: center;
        border-bottom: 3px solid #F5C518; /* IMDb Yellow */
    }
    .header-logo {
        background-color: #F5C518;
        color: #000;
        font-weight: 900;
        padding: 5px 12px;
        border-radius: 4px;
        font-size: 24px;
        margin-right: 15px;
        letter-spacing: -1px;
    }
    .header-text {
        color: #FFF;
        font-size: 20px;
        font-weight: 500;
    }

    /* SEARCH BAR (IMDb style) */
    .stTextInput > div > div > input {
        border-radius: 4px;
        border: 1px solid #ccc;
        padding: 12px;
        font-size: 16px;
    }
    .stTextInput > div > div > input:focus {
        border-color: #F5C518; /* Yellow focus */
        box-shadow: 0 0 0 1px #F5C518;
    }

    /* MOVIE CARD CONTAINER */
    /* This targets the container of each movie result */
    div[data-testid="stVerticalBlock"] > div[style*="background-color"] {
        background: white !important;
        border: 1px solid #ddd;
        border-radius: 4px; /* Sharper corners */
        padding: 0px !important; /* Reset padding for poster look */
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: box-shadow 0.2s ease;
        height: 100%;
        overflow: hidden;
    }
    div[data-testid="stVerticalBlock"] > div[style*="background-color"]:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* FAKE POSTER GENERATOR */
    .poster-placeholder {
        height: 180px;
        background: linear-gradient(135deg, #e0e0e0 0%, #f5f5f5 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 48px;
        font-weight: 900;
        color: #ccc;
        border-bottom: 1px solid #eee;
        position: relative;
    }
    
    /* RATING BADGE (Floating on poster) */
    .rating-badge {
        position: absolute;
        bottom: 10px;
        left: 10px;
        background-color: #F5C518; /* IMDb Yellow */
        color: #000;
        font-weight: 700;
        padding: 4px 8px;
        border-radius: 2px;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    /* CARD CONTENT */
    .card-content {
        padding: 15px;
    }
    .movie-title {
        font-size: 18px;
        font-weight: 700;
        color: #121212;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .movie-meta {
        font-size: 14px;
        color: #666;
        margin-bottom: 12px;
    }
    
    /* BUTTON STYLES */
    .stButton > button {
        background-color: #F5C518; /* Yellow */
        color: #000;
        font-weight: 700;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        text-transform: uppercase;
    }
    .stButton > button:hover {
        background-color: #E2B616;
        color: #000;
    }
    
    /* EXPANDER CUSTOMIZATION */
    .streamlit-expanderHeader {
        font-size: 13px;
        color: #0055AA; /* Link Blue */
    }
</style>
""", unsafe_allow_html=True)

if 'limit' not in st.session_state:
    st.session_state.limit = 8
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
        st.error("Database Error.")
        st.stop()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

st.markdown("""
<div class="imdb-header">
    <div class="header-logo">IMDb</div>
    <div class="header-text">Plot Matcher</div>
</div>
""", unsafe_allow_html=True)

query = st.text_input("", placeholder="Search for a plot line... (e.g., 'A corporate guy gets forced into crime')", label_visibility="collapsed")

if query != st.session_state.last_query:
    st.session_state.limit = 8
    st.session_state.last_query = query

if query:
    query_vec = model.encode([query])
    sim_scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = sim_scores.argsort()[-50:][::-1]
    
    st.markdown("<h4 style='padding-left: 5px; margin-top: 20px;'>Top Results</h4>", unsafe_allow_html=True)

    cols = st.columns(4)
    
    seen_movies = set()
    count_displayed = 0
    
    for idx in top_indices:
        if count_displayed >= st.session_state.limit:
            break
            
        meta = metadata[idx]
        title = meta['Title']
        
        if title in seen_movies: continue
        seen_movies.add(title)
        
        match_score = sim_scores[idx] * 10
        star_rating = f"{match_score:.1f}"

        initials = "".join([x[0] for x in title.split()[:2]]).upper()
        
        col_idx = count_displayed % 4
        
        with cols[col_idx]:
            with st.container():
                st.markdown(f"""
                <div class="poster-placeholder">
                    {initials}
                    <div class="rating-badge">★ {star_rating}</div>
                </div>
                <div class="card-content">
                    <div class="movie-title" title="{title}">{title}</div>
                    <div class="movie-meta">{meta['Year']} • {meta['Genre']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Native Expander for plot
                with st.expander("Read Plot Match"):
                    st.caption(f"...{meta['Text']}...")

        count_displayed += 1

    # Load More Button
    if count_displayed >= st.session_state.limit and st.session_state.limit < 40:
        st.markdown("<br>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([1, 1, 1])
        with btn_col:
            if st.button("Load More"):
                st.session_state.limit += 8
                st.rerun()

    if count_displayed == 0:
        st.warning("No matches found.")

# Footer spacing
st.markdown("<br><br>", unsafe_allow_html=True)
