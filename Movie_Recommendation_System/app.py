import html, os

import pandas as pd
import streamlit as st

from recommender import Recommender, load_movielens

st.set_page_config(page_title="Matinee: movie recommendations", page_icon="🎞️", layout="wide")

SYNTH = os.environ.get("MOVIELENS_SYNTHETIC") == "1"
GENRE_COLORS = {"Action": "#D9694A", "Adventure": "#D99A4A", "Animation": "#4AA8D9", "Children": "#6FC3A0",
                "Comedy": "#E3C04D", "Crime": "#8C6BB1", "Documentary": "#7FA37A", "Drama": "#B5547A",
                "Fantasy": "#9B7BD6", "Film-Noir": "#7A8394", "Horror": "#B8453F", "Musical": "#D67BB5",
                "Mystery": "#5C7FB5", "Romance": "#E07A8F", "Sci-Fi": "#4FB3C9", "Thriller": "#6E8B9E",
                "War": "#8B7B5C", "Western": "#C08A5A", "IMAX": "#5CA3D6"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=DM+Sans:wght@400;500;600&display=swap');
:root{--bg:#121418;--panel:#1A1D23;--line:#2A2E37;--text:#EDEAE3;--muted:#9A9FA9;--amber:#E9B44C;}
#MainMenu,footer,header[data-testid="stHeader"]{visibility:hidden;height:0}
.stApp{font-family:'DM Sans',sans-serif}
.block-container{padding-top:2.2rem;max-width:1180px}
section[data-testid="stSidebar"]{border-right:1px solid var(--line)}
.brand{font-family:'Fraunces',serif;font-weight:700;font-size:2.6rem;letter-spacing:-.02em;line-height:1}
.tag{color:var(--muted);margin:.35rem 0 .2rem;font-size:1rem}
.facts{display:flex;gap:1.6rem;color:var(--muted);font-size:.88rem;margin-bottom:1.2rem}
.facts b{color:var(--text);font-weight:600}
h2.sec{font-family:'Fraunces',serif;font-weight:500;font-size:1.55rem;margin:.6rem 0 .1rem}
.sub{color:var(--muted);font-size:.92rem;margin-bottom:1rem;max-width:62ch}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-bottom:1.6rem}
.ticket{display:grid;grid-template-columns:1fr 96px;background:var(--panel);border:1px solid var(--line);
 border-left:4px solid var(--g);border-radius:10px;min-height:112px}
.ticket .main{padding:14px 16px;min-width:0}
.ticket .top{display:flex;gap:10px;align-items:baseline}
.rk{color:var(--muted);font-size:.8rem;min-width:1.2rem}
.ticket h3{font-family:'Fraunces',serif;font-weight:500;font-size:1.12rem;line-height:1.25;margin:0;padding:0}
.meta{margin:.5rem 0 0 1.9rem;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.yr{color:var(--muted);font-size:.82rem;margin-right:4px}
.chip{border:1px solid var(--line);border-radius:999px;padding:1px 9px;font-size:.74rem;color:var(--muted)}
.note{margin:.55rem 0 0 1.9rem;font-size:.8rem;color:var(--muted)}
.note b{color:var(--text);font-weight:500}
.stub{position:relative;border-left:2px dashed var(--line);display:flex;flex-direction:column;align-items:center;justify-content:center}
.stub:before,.stub:after{content:"";position:absolute;left:-7px;width:12px;height:12px;border-radius:50%;background:var(--bg);border:1px solid var(--line)}
.stub:before{top:-7px}.stub:after{bottom:-7px}
.pct{font-family:'Fraunces',serif;font-weight:700;font-size:1.7rem;color:var(--amber);line-height:1}
.pct small{font-size:.9rem}
.lbl{color:var(--muted);font-size:.72rem;margin-top:4px}
.bars{display:grid;gap:9px;margin-bottom:1.4rem}
.bar{display:grid;grid-template-columns:96px 1fr;gap:10px;align-items:center;font-size:.86rem}
.bar i{display:block;height:8px;border-radius:4px;background:var(--g)}
.row{display:flex;justify-content:space-between;gap:12px;padding:9px 0;border-bottom:1px solid var(--line);font-size:.92rem}
.row span{color:var(--muted)}.row em{color:var(--amber);font-style:normal;font-weight:600}
.demo{border:1px solid var(--amber);color:var(--amber);border-radius:8px;padding:8px 12px;font-size:.85rem;margin-bottom:1rem}
button[data-baseweb="tab"]{font-size:.95rem}
@media (max-width:900px){.grid{grid-template-columns:1fr}.facts{flex-wrap:wrap;gap:.8rem 1.4rem}.brand{font-size:2.1rem}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading MovieLens and building models. First run takes a minute.")
def get_model():
    return Recommender(*load_movielens(synthetic=SYNTH))


@st.cache_resource(show_spinner="Evaluating all models on the held-out ratings.")
def get_results():
    return get_model().evaluate()


def ticket(rank, r, label, suffix="%"):
    color = GENRE_COLORS.get(r.genres[0], "#7A8394")
    chips = "".join(f'<span class="chip">{html.escape(g)}</span>' for g in r.genres[:3])
    year = f'<span class="yr">{r.year}</span>' if r.year else ""
    note = (f'<div class="note">Because you rated <b>{html.escape(r.because)}</b> highly</div>'
            if r.because else "")
    return (f'<div class="ticket" style="--g:{color}"><div class="main"><div class="top">'
            f'<span class="rk">{rank}</span><h3>{html.escape(r.name)}</h3></div>'
            f'<div class="meta">{year}{chips}</div>{note}</div>'
            f'<div class="stub"><div class="pct">{r.match}<small>{suffix}</small></div>'
            f'<div class="lbl">{label}</div></div></div>')


def grid(df, label):
    st.markdown('<div class="grid">' + "".join(ticket(i, r, label) for i, r in enumerate(df.itertuples(), 1))
                + "</div>", unsafe_allow_html=True)


try:
    model = get_model()
except Exception as exc:
    st.error("Could not load the MovieLens data. Check your internet connection and reload.")
    st.code(str(exc))
    st.stop()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="brand" style="font-size:1.7rem">Matinee</div>', unsafe_allow_html=True)
    st.caption("Choose a viewer and a method.")
    user = st.selectbox("Viewer", model.user_ids, format_func=lambda u: f"Viewer {u}")
    method = st.radio("Method", ["Hybrid", "Collaborative", "Content-based"],
                      help="Collaborative uses what similar viewers rated. Content-based uses genres and tags.")
    alpha = st.slider("Collaborative weight", 0.0, 1.0, 0.5, 0.05) if method == "Hybrid" else 0.5
    n = st.slider("Number of picks", 4, 20, 8, 2)
key = {"Hybrid": "hybrid", "Collaborative": "cf", "Content-based": "content"}[method]

# ----------------------------------------------------------------- header
n_ratings = int(model.seen.sum())
st.markdown(f'<div class="brand">Matinee</div><div class="tag">Movie recommendations built from '
            f'real viewing history.</div><div class="facts"><span><b>{len(model.user_ids):,}</b> viewers</span>'
            f'<span><b>{len(model.movies):,}</b> movies</span><span><b>{n_ratings:,}</b> ratings</span></div>',
            unsafe_allow_html=True)
if SYNTH:
    st.markdown('<div class="demo">Demo mode: generated data, not MovieLens.</div>', unsafe_allow_html=True)

tab_rec, tab_sim, tab_res, tab_about = st.tabs(["For you", "Similar titles", "Model results", "About"])

# ----------------------------------------------------------------- for you
with tab_rec:
    desc = {"hybrid": f"Blends collaborative ({alpha:.0%}) and content-based ({1 - alpha:.0%}) scores.",
            "cf": "Finds movies rated like the ones this viewer liked, using other viewers' ratings.",
            "content": "Finds movies that share genres and tags with this viewer's favourites."}[key]
    st.markdown(f'<h2 class="sec">Picks for Viewer {user}</h2><div class="sub">{desc} '
                f'Match shows how this title ranks among every movie the viewer has not rated.</div>',
                unsafe_allow_html=True)
    grid(model.recommend(user, n, key, alpha), "match")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<h2 class="sec">Taste profile</h2><div class="sub">Genres across movies this viewer rated 4 or higher.</div>',
                    unsafe_allow_html=True)
        prof = model.taste_profile(user)
        st.markdown('<div class="bars">' + "".join(
            f'<div class="bar" style="--g:{GENRE_COLORS.get(g, "#7A8394")}"><span>{html.escape(g)}</span>'
            f'<i style="width:{v * 100:.0f}%"></i></div>' for g, v in prof.items()) + "</div>",
            unsafe_allow_html=True)
    with right:
        st.markdown('<h2 class="sec">Highest rated by this viewer</h2>', unsafe_allow_html=True)
        st.markdown("".join(
            f'<div class="row"><div>{html.escape(r.name)} <span>{r.year or ""}</span></div><em>★ {r.rating:g}</em></div>'
            for r in model.history(user, 6).itertuples()), unsafe_allow_html=True)

# ----------------------------------------------------------------- similar
with tab_sim:
    st.markdown('<h2 class="sec">Find similar titles</h2><div class="sub">Pick a movie to see its closest matches.</div>',
                unsafe_allow_html=True)
    lookup = dict(zip(model.movies.movieId, model.movies.name + model.movies.year.map(lambda y: f" ({y})" if y else "")))
    c1, c2 = st.columns([3, 2])
    movie = c1.selectbox("Movie", list(lookup), format_func=lookup.get, index=0)
    basis = c2.radio("Compare by", ["Viewer ratings", "Genres and tags"], horizontal=True)
    sim = model.similar(movie, 10, "cf" if basis == "Viewer ratings" else "content")
    if sim.empty:
        st.info("No close matches found. Try the other comparison.")
    else:
        grid(sim, "similar")

# ----------------------------------------------------------------- results
with tab_res:
    res = get_results()
    best = res.loc[res["Precision@10"].idxmax()]
    st.markdown(f'<h2 class="sec">How well do the models work?</h2><div class="sub">Each viewer\'s ratings were split 80/20. '
                f'Models learn from the 80% and are scored on the hidden 20%. Best ranking: <b>{best.Model}</b> '
                f'(Precision@10 {best["Precision@10"]:.3f}).</div>', unsafe_allow_html=True)
    st.dataframe(res.drop(columns="alpha"), hide_index=True, width="stretch", column_config={
        "RMSE": st.column_config.NumberColumn(help="Rating error. Lower is better.", format="%.3f"),
        "MAE": st.column_config.NumberColumn(help="Average rating error. Lower is better.", format="%.3f"),
        "Precision@10": st.column_config.NumberColumn(help="Share of the top 10 the viewer liked. Higher is better.", format="%.3f"),
        "Recall@10": st.column_config.NumberColumn(help="Share of liked movies found in the top 10. Higher is better.", format="%.3f")})
    a, b = st.columns(2, gap="large")
    a.markdown("**Ranking quality** (Precision@10, higher is better)")
    a.bar_chart(res.dropna(subset=["Precision@10"]).set_index("Model")["Precision@10"], color="#E9B44C")
    b.markdown("**Rating error** (RMSE, lower is better)")
    b.bar_chart(res.set_index("Model")["RMSE"], color="#4FB3C9")
    sweep = res.dropna(subset=["alpha"]).sort_values("alpha").set_index("alpha")[["Precision@10", "Recall@10"]]
    st.markdown("**Effect of the collaborative weight** (0 is content only, 1 is collaborative only)")
    st.line_chart(sweep, color=["#E9B44C", "#4FB3C9"])

# ------------------------------------------------------------------- about
with tab_about:
    st.markdown("""
<h2 class="sec">How it works</h2>

**Collaborative filtering (item-based).** Every movie is compared with every other using cosine similarity on
mean-centred ratings. A movie's score for a viewer is the similarity-weighted average of that viewer's ratings of its
50 nearest neighbours.

**Content-based.** Each movie becomes a TF-IDF vector from its genres and user tags. Movies with similar vectors are
neighbours, and scoring works the same way.

**Hybrid.** `alpha * collaborative + (1 - alpha) * content`. Ranking also adds a small popularity prior, so obscure titles
with weak evidence do not crowd out well-known ones.

**Baselines.** Most popular, and each viewer's average rating. A model is only useful if it beats these.

Data: [MovieLens latest-small](https://grouplens.org/datasets/movielens/latest/), GroupLens Research.
""", unsafe_allow_html=True)
