import streamlit as st
import pickle
import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="CineMatch AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

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

    div[data-testid="stVerticalBlock"] > div[style*="background-color"] {
        border-radius: 15px;
        padding: 20px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .match-badge {
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 8px;
        display: inline-block;
        font-weight: bold;
    }
    .badge-title { background-color: #FFD700; color: black; }
    .badge-plot { background-color: #00ADB5; color: white; }
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

        if norm_query == norm_title:
            score = 3

        elif norm_title.startswith(norm_query):
            score = 2

        else:
            all_words_found = True
            for word in query_words:
                if not re.search(r'\b' + re.escape(word) + r'\b', norm_title):
                    all_words_found = False
                    break
            
            if all_words_found:
                score = 1

        if score > 0:
            title_matches.append({
                'meta': meta,
                'year': meta['Year'],
                'type': 'Title Match',
                'vector': embeddings[idx],
                'match_score': score
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

    top_indices = sim_scores.argsort()[-60:][::-1]
    
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

    plot_candidates.sort(key=lambda x: x['year'], reverse=True)

    results.extend(plot_candidates)
    
    return results

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: #00ADB5;'>CineMatch AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; margin-top: -15px;'>Universal Search: Titles & Plots</p>", unsafe_allow_html=True)

query = st.text_input("", placeholder="Type a movie name (e.g., 'Tron') or a plot (e.g., 'Future robots')...", label_visibility="collapsed")

if query != st.session_state.last_query:
    st.session_state.limit = 6
    st.session_state.last_query = query

if query:
    search_results = search_movies(query, model, embeddings, metadata)
    
    st.markdown("---")
    
    grid_cols = st.columns(2)
    count_displayed = 0

    for item in search_results:
        if count_displayed >= st.session_state.limit:
            break
            
        meta = item['meta']
        match_type = item['type']

        badge_class = "badge-title" if match_type == "Title Match" else "badge-plot"
        
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
                    position: relative;
                ">
                    <span class="match-badge {badge_class}">{match_type}</span>
                    <h3 style="margin: 10px 0 0 0; color: #EEE;">{meta['Title']}</h3>
                    <p style="color: #00ADB5; font-size: 0.9em; margin-top: 5px;">{meta['Year']} • {meta['Genre']}</p>
                </div>
                """, unsafe_allow_html=True)

                with st.expander(f"View Plot ({meta['Year']})"):
                    st.write(f"...{meta['Text']}...")
        
        count_displayed += 1

    if count_displayed >= st.session_state.limit and count_displayed < len(search_results):
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("Load More Results", use_container_width=True):
                st.session_state.limit += 6
                st.rerun()

    if count_displayed == 0:
        st.warning(f"No results found for '{query}'.")
