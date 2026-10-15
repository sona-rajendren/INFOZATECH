"""Movie recommender on MovieLens: item-based CF, content-based (genres + tags),
a blended hybrid, two baselines, and an evaluation harness.

Run `python recommender.py` for the results table (saved to results.csv).
Add --synthetic to test the code offline with generated data.
"""
import argparse, io, re, urllib.request, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path(__file__).parent / "data"
NEIGHBOURS_K = 50      # neighbours kept per movie
MIN_RATINGS = 3        # drop movies with fewer ratings (too little signal)
SHRINK = 1.0           # damps predictions that rest on weak similarity
LIKE = 4.0             # rating counted as "relevant" for ranking metrics
POP_WEIGHT = 1.0       # popularity prior added when RANKING (not for rating error)
ALPHAS = [0.25, 0.5, 0.75]
TOP_N = 10
SEED = 42


# ----------------------------------------------------------------- data
def synthetic_data(n_users=250, n_movies=500, seed=SEED):
    """Random ratings with real taste structure, for offline code checks only."""
    rng = np.random.default_rng(seed)
    genres = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Drama",
              "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"]
    mg = [rng.choice(genres, size=rng.integers(1, 4), replace=False) for _ in range(n_movies)]
    G = np.array([[g in gs for g in genres] for gs in mg], float)
    taste = rng.normal(0, 1, (n_users, len(genres)))
    quality = rng.normal(0, 0.5, n_movies)
    pop = rng.pareto(1.2, n_movies) + 1
    pop /= pop.sum()
    rows = []
    for u in range(n_users):
        items = rng.choice(n_movies, size=rng.integers(25, 80), replace=False, p=pop)
        raw = 3.4 + 0.5 * (G[items] @ taste[u]) + quality[items] + rng.normal(0, 0.5, len(items))
        rows += [(u + 1, i + 1, r) for i, r in zip(items, np.clip(np.round(raw * 2) / 2, 0.5, 5))]
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])
    years = rng.integers(1960, 2020, n_movies)
    movies = pd.DataFrame({"movieId": np.arange(1, n_movies + 1),
                           "title": [f"Sample Film {i + 1} ({y})" for i, y in enumerate(years)],
                           "genres": ["|".join(g) for g in mg]})
    return ratings, movies, None


def load_movielens(synthetic=False):
    if synthetic:
        return synthetic_data()
    folder = DATA_DIR / "ml-latest-small"
    if not (folder / "ratings.csv").exists():
        DATA_DIR.mkdir(exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA_DIR)
    tags = folder / "tags.csv"
    return (pd.read_csv(folder / "ratings.csv"), pd.read_csv(folder / "movies.csv"),
            pd.read_csv(tags) if tags.exists() else None)


def split_title(raw):
    """'Usual Suspects, The (1995)' -> ('The Usual Suspects', 1995)."""
    m = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", raw)
    title, year = (m.group(1), int(m.group(2))) if m else (raw, 0)
    m = re.match(r"^(.*), (The|A|An)$", title)
    return (f"{m.group(2)} {m.group(1)}" if m else title), year


def _topk(S, k):
    """Keep each row's k strongest positive similarities, zero the rest."""
    S = S.astype(np.float32).copy()
    np.fill_diagonal(S, 0)
    drop = np.argpartition(S, -k, axis=1)[:, :-k]
    np.put_along_axis(S, drop, 0, axis=1)
    return np.clip(S, 0, None)


# ---------------------------------------------------------------- model
class Recommender:
    def __init__(self, ratings, movies, tags=None, k=NEIGHBOURS_K, use_tags=True, seed=SEED):
        counts = ratings.movieId.value_counts()
        keep = counts[counts >= MIN_RATINGS].index
        ratings = ratings[ratings.movieId.isin(keep)].copy()
        movies = movies[movies.movieId.isin(keep)].sort_values("movieId").reset_index(drop=True)
        parsed = movies.title.map(split_title)
        movies["name"] = [p[0] for p in parsed]
        movies["year"] = [p[1] for p in parsed]
        movies["genre_list"] = movies.genres.map(
            lambda g: [x for x in g.split("|") if "no genres" not in x] or ["Unclassified"])
        self.movies = movies
        self.user_ids = np.sort(ratings.userId.unique())
        self.uidx = {u: i for i, u in enumerate(self.user_ids)}
        self.midx = {m: i for i, m in enumerate(movies.movieId)}
        ratings["u"] = ratings.userId.map(self.uidx)
        ratings["i"] = ratings.movieId.map(self.midx)

        # hold out 20% of every user's ratings for testing
        shuf = ratings.sample(frac=1, random_state=seed)
        pos = shuf.groupby("u").cumcount()
        size = shuf.groupby("u")["rating"].transform("size")
        is_test = pos < np.maximum(1, (size * 0.2).round())
        self.train, self.test = shuf[~is_test], shuf[is_test]

        shape = (len(self.user_ids), len(movies))
        self.F = self._matrix(ratings, shape)        # every known rating
        self.R = self._matrix(self.train, shape)     # training ratings only
        self.mask, self.seen = self.R > 0, self.F > 0
        n_user = np.maximum(self.mask.sum(1), 1)
        self.user_mean = (self.R.sum(1) / n_user).astype(np.float32)
        self.global_mean = float(self.train.rating.mean())
        n_item = self.mask.sum(0)
        self.item_count = n_item
        self.item_mean = np.where(n_item > 0, self.R.sum(0) / np.maximum(n_item, 1),
                                  self.global_mean).astype(np.float32)
        self.pop_prior = (np.log1p(n_item) / np.log1p(n_item.max())).astype(np.float32)
        self._fit(tags if use_tags else None, k)

    @staticmethod
    def _matrix(df, shape):
        M = np.zeros(shape, np.float32)
        M[df.u.values, df.i.values] = df.rating.values
        return M

    def _fit(self, tags, k):
        # item-based CF: cosine on ratings centred by each user's mean
        C = (self.R - self.user_mean[:, None]) * self.mask
        Cn = C / (np.linalg.norm(C, axis=0) + 1e-9)
        self.S_cf = _topk(Cn.T @ Cn, k)
        # content-based: TF-IDF on genres (+ user tags when available)
        docs = (self.movies.genres.str.replace("(no genres listed)", "", regex=False)
                .str.replace("-", "", regex=False).str.replace("|", " ", regex=False).str.lower())
        if tags is not None:
            t = tags.assign(tag=tags.tag.astype(str).str.lower()).groupby("movieId").tag.apply(" ".join)
            docs = docs + " " + self.movies.movieId.map(t).fillna("")
        X = TfidfVectorizer(token_pattern=r"[^\s]+", sublinear_tf=True,
                            min_df=2 if tags is not None else 1).fit_transform(docs)
        self.S_ct = _topk((X @ X.T).toarray(), k)
        # weighted average of the user's centred ratings over each movie's neighbours
        m = self.mask.astype(np.float32)
        self.dev_cf = (C @ self.S_cf.T) / (m @ self.S_cf.T + SHRINK)
        self.dev_ct = (C @ self.S_ct.T) / (m @ self.S_ct.T + SHRINK)

    # ------------------------------------------------------- scoring
    def deviation(self, model="hybrid", alpha=0.5):
        if model == "cf":
            return self.dev_cf
        if model == "content":
            return self.dev_ct
        return alpha * self.dev_cf + (1 - alpha) * self.dev_ct

    def rank_score(self, dev):
        """Ranking score: predicted deviation plus a small popularity prior."""
        return dev + POP_WEIGHT * self.pop_prior

    # ------------------------------------------------------ evaluation
    def _metrics(self, pred, rank, n=TOP_N):
        tu, ti, tr = self.test.u.values, self.test.i.values, self.test.rating.values
        err = np.clip(pred[tu, ti], 0.5, 5) - tr
        out = {"RMSE": float(np.sqrt((err ** 2).mean())), "MAE": float(np.abs(err).mean())}
        if rank is None:
            return {**out, f"Precision@{n}": np.nan, f"Recall@{n}": np.nan}
        rel = np.zeros_like(self.F, bool)
        rel[tu, ti] = tr >= LIKE
        s = np.where(self.mask, -np.inf, rank)
        top = np.argpartition(-s, n, axis=1)[:, :n]
        hits = rel[np.arange(len(rel))[:, None], top].sum(1)
        n_rel = rel.sum(1)
        ok = n_rel > 0
        return {**out, f"Precision@{n}": float((hits / n)[ok].mean()),
                f"Recall@{n}": float((hits / np.maximum(n_rel, 1))[ok].mean())}

    def evaluate(self):
        um = self.user_mean[:, None]
        pop_pred = np.broadcast_to(self.item_mean, self.F.shape)
        rows = [("Most popular", np.nan, pop_pred, np.broadcast_to(self.item_count, self.F.shape).astype(float)),
                ("User average", np.nan, np.broadcast_to(um, self.F.shape), None),
                ("Item-based CF", 1.0, um + self.dev_cf, self.rank_score(self.dev_cf)),
                ("Content-based", 0.0, um + self.dev_ct, self.rank_score(self.dev_ct))]
        for a in ALPHAS:
            d = self.deviation("hybrid", a)
            rows.append((f"Hybrid (alpha={a})", a, um + d, self.rank_score(d)))
        return pd.DataFrame([{"Model": n, "alpha": a, **self._metrics(p, r)} for n, a, p, r in rows])

    # -------------------------------------------------------- queries
    def _cards(self, idx, extra):
        df = self.movies.iloc[idx]
        out = pd.DataFrame({"movieId": df.movieId.values, "name": df.name.values,
                            "year": df.year.values, "genres": df.genre_list.values})
        for k, v in extra.items():
            out[k] = v
        return out

    def recommend(self, user_id, n=10, model="hybrid", alpha=0.5):
        u = self.uidx[user_id]
        s = self.rank_score(self.deviation(model, alpha)[u])
        cand = np.where(~self.seen[u])[0]
        cs = s[cand]
        pct = cs.argsort().argsort() / max(len(cs) - 1, 1) * 100   # rank among unseen titles
        order = np.argsort(-cs)[:n]
        liked = np.where(self.F[u] >= LIKE)[0]
        because = []
        for i in cand[order]:
            sim = self.S_cf[i, liked] + self.S_ct[i, liked] if len(liked) else np.zeros(1)
            because.append(self.movies.name[liked[sim.argmax()]] if sim.max() > 0 else "")
        return self._cards(cand[order], {"match": np.round(pct[order]).astype(int), "because": because})

    def similar(self, movie_id, n=10, basis="cf"):
        S = self.S_cf if basis == "cf" else self.S_ct
        row = S[self.midx[movie_id]]
        idx = np.argsort(-row)[:n]
        idx = idx[row[idx] > 0]
        return self._cards(idx, {"match": np.round(row[idx] * 100).astype(int), "because": [""] * len(idx)})

    def history(self, user_id, n=8):
        u = self.uidx[user_id]
        idx = np.argsort(-self.F[u], kind="stable")[:n]
        return self._cards(idx, {"rating": self.F[u, idx]})

    def taste_profile(self, user_id, top=6):
        u = self.uidx[user_id]
        liked = np.where(self.F[u] >= LIKE)[0]
        counts = pd.Series([g for i in liked for g in self.movies.genre_list[i]]).value_counts()
        return (counts / counts.max()).head(top) if len(counts) else counts


# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="offline test on generated data")
    args = ap.parse_args()
    model = Recommender(*load_movielens(args.synthetic))
    table = model.evaluate()
    table.to_csv("results.csv", index=False)
    print(table.round(4).to_string(index=False))
    uid = int(model.user_ids[0])
    print(f"\nTop picks for user {uid}:")
    print(model.recommend(uid, 5)[["name", "year", "match", "because"]].to_string(index=False))
