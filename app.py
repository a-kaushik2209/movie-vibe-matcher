import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMatch AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background-color: #0b0c10; /* Ultra Dark Navy */
        color: #c5c6c7;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    .stTextInput > div > div > input {
        padding: 18px 30px;
        font-size: 1.15rem;
        border-radius: 50px;
        border: 2px solid #1f2833;
        background: rgba(31, 40, 51, 0.8);
        color: #66fcf1; /* Cyan Text */
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #45a29e;
        box-shadow: 0 0 25px rgba(102, 252, 241, 0.3);
        background: rgba(31, 40, 51, 1);
    }
    
    div[data-testid="stVerticalBlock"] > div[style*="background-color"] {
        background: transparent !important;
    }
    
    .movie-card {
        background: linear-gradient(145deg, #1f2833, #0b0c10);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #1f2833;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        transition: transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1), box-shadow 0.3s ease, border-color 0.3s ease;
        margin-bottom: 25px;
        position: relative;
        overflow: hidden;
    }
    
    .movie-card:hover {
        transform: translateY(-8px) scale(1.01);
        box-shadow: 0 20px 40px rgba(0,0,0,0.7);
        border-color: #66fcf1; /* Glow on hover */
    }

    .movie-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 10px 0 5px 0;
        background: -webkit-linear-gradient(#fff, #a5a5a5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .movie-meta {
        color: #45a29e;
        font-size: 0.95rem;
        font-weight: 400;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .badge {
        font-size: 0.75rem;
        padding: 5px 12px;
        border-radius: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: inline-block;
    }
    .badge-gold { 
        background: linear-gradient(45deg, #FFD700, #FDB931); 
        color: #000; 
        box-shadow: 0 4px 10px rgba(255, 215, 0, 0.2); 
    }
    .badge-teal { 
        background: rgba(69, 162, 158, 0.2); 
        color: #66fcf1; 
        border: 1px solid #45a29e; 
    }

    .stButton > button {
        width: 100%;
        border-radius: 15px;
        background: linear-gradient(90deg, #45a29e 0%, #66fcf1 100%);
        color: #0b0c10;
        border: none;
        padding: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 252, 241, 0.3);
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(102, 252, 241, 0.5);
        color: #000;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if 'search_results' not in st.session_state:
    st.session_state.search_results = []
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
    with st.spinner("Lighting up the Cinema..."):
        model, embeddings, metadata = load_resources()
    if model is None:
        st.error("System Error: Database files missing.")
        st.stop()
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

def normalize_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

def search_movies(query, model, embeddings, metadata):
    results = []
    seen_titles = set()
    norm_query = normalize_text(query)
    query_words = norm_query.split()

    title_matches = []
    for idx, meta in enumerate(metadata):
        norm_title = normalize_text(meta['Title'])
        score = 0
        if norm_query == norm_title: score = 3
        elif norm_title.startswith(norm_query): score = 2
        else:
            if all(re.search(r'\b' + re.escape(w) + r'\b', norm_title) for w in query_words):
                score = 1
        
        if score > 0:
            title_matches.append({
                'meta': meta,
                'year': meta['Year'],
                'type': 'Title Match',
                'vector': embeddings[idx],
                'match_score': score,
                'score': 100 + score
            })
    
    title_matches.sort(key=lambda x: (x['match_score'], x['year']), reverse=True)
    
    target_movie_vector = None
    if title_matches:
        target_movie_vector = title_matches[0]['vector']
    
    for item in title_matches:
        if item['meta']['Title'] not in seen_titles:
            results.append(item)
            seen_titles.add(item['meta']['Title'])

    if target_movie_vector is not None:
        query_vec = target_movie_vector.reshape(1, -1)
        search_type_label = "Similar Plot"
    else:
        query_vec = model.encode([query])
        search_type_label = "Plot Match"

    sim_scores = cosine_similarity(query_vec, embeddings)[0]
    top_indices = sim_scores.argsort()[-80:][::-1]
    
    plot_candidates = []
    for idx in top_indices:
        meta = metadata[idx]
        title = meta['Title']
        if title in seen_titles: continue
        
        plot_candidates.append({
            'meta': meta,
            'year': meta['Year'],
            'type': search_type_label,
            'score': sim_scores[idx]
        })
        seen_titles.add(title)

    results.extend(plot_candidates)
    return results

with st.sidebar:
    st.markdown("## **Studio Settings**")
    st.markdown("---")
    st.write("Sorting Preference:")
    sort_option = st.radio("", ["Newest Releases", "Best Match"], label_visibility="collapsed")
    st.markdown("---")
    st.info("**Pro Tip:**\nTry abstract searches like:\n*\"Cyberpunk detective story\"*\n*\"Space opera with war\"*")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: #66fcf1; font-size: 3.5rem; font-weight: 800; letter-spacing: -2px; text-shadow: 0 0 30px rgba(102, 252, 241, 0.4);'>CINEMATCH</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; letter-spacing: 4px; text-transform: uppercase; font-size: 0.8rem; margin-top: -20px;'>AI Powered Discovery</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

query = st.text_input("", placeholder="🔍 Search movies, plots, or vibes...", label_visibility="collapsed")

if query and (query != st.session_state.last_query or not st.session_state.search_results):
    st.session_state.limit = 6
    st.session_state.last_query = query
    st.session_state.search_results = search_movies(query, model, embeddings, metadata)

if st.session_state.search_results:
    titles = [r for r in st.session_state.search_results if r['type'] == 'Title Match']
    plots = [r for r in st.session_state.search_results if r['type'] != 'Title Match']
    
    if sort_option == "Newest Releases":
        plots.sort(key=lambda x: x['year'], reverse=True)
    else:
        plots.sort(key=lambda x: x['score'], reverse=True)
    
    sorted_display_list = titles + plots
    visible_results = sorted_display_list[:st.session_state.limit]

    st.markdown("<br>", unsafe_allow_html=True)
    grid_cols = st.columns(2)
    
    for i, item in enumerate(visible_results):
        meta = item['meta']
        match_type = item['type']
        
        if match_type == "Title Match":
            badge_html = '<span class="badge badge-gold">★ DIRECT HIT</span>'
            border_glow = "border: 1px solid #FFD700;" 
        else:
            match_pct = int(item["score"]*100)
            badge_html = f'<span class="badge badge-teal">MATCH SCORE: {match_pct}%</span>'
            border_glow = ""

        col_idx = i % 2
        with grid_cols[col_idx]:
            with st.container():
                st.markdown(f"""
                <div class="movie-card" style="{border_glow}">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div style="width: 85%;">
                            {badge_html}
                            <div class="movie-title">{meta['Title']}</div>
                            <div class="movie-meta">{meta['Year']} • {meta['Genre']}</div>
                        </div>
                        <div style="font-size: 2.5rem; opacity: 0.2;">🎬</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                with st.expander(f"Read Plot Synopsis"):
                    st.write(meta['Text'])

    if st.session_state.limit < len(sorted_display_list):
        st.markdown("<br><br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("Load More Results"):
                st.session_state.limit += 6
                st.rerun()

elif query:
    st.markdown(f"<br><h3 style='text-align: center; color: #444;'>No signals found for '{query}'...</h3>", unsafe_allow_html=True)
