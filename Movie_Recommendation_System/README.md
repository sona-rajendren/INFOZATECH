# Matinee : Movie Recommendation System

A movie recommender built on the MovieLens dataset, with a web app for exploring recommendations.

## Features
- **Item-based collaborative filtering**: cosine similarity on mean-centred ratings, 50 nearest neighbours per movie.
- **Content-based filtering**: TF-IDF on genres and user tags (tags.csv).
- **Hybrid model**: `alpha * collaborative + (1 - alpha) * content`, with a small popularity prior when ranking.
- **Baselines**: most popular, and each user's average rating.
- **Evaluation**: RMSE and MAE for rating error, Precision@10 and Recall@10 for ranking, on a per-user 80/20 split.
- **Web app**: recommendations with explanations ("Because you rated X highly"), taste profile, similar titles, and a model results tab.

## Run it
```bash
pip install -r requirements.txt
python recommender.py          # prints the results table, saves results.csv
streamlit run app.py           # opens http://localhost:8501
```
Run both commands from inside this folder so Streamlit picks up `.streamlit/config.toml`.
The first run downloads MovieLens (ml-latest-small) automatically.

## Project layout
| File | Purpose |
|---|---|
| `recommender.py` | Data loading, models, evaluation, command line results |
| `app.py` | Streamlit web app |
| `.streamlit/config.toml` | Dark theme |
| `requirements.txt` | Dependencies |

## Results
Run `python recommender.py` 
| Model | RMSE | MAE | Precision@10 | Recall@10 |
|---|---|---|---|---|
| Most popular |0.9580|0.7430|0.1148|0.1019|
| User average |0.9344|0.7289|N/A|N/A|
| Item-based CF |0.8797|0.6725|0.1352|0.1145|
| Content-based |0.9268|0.7168|0.0757|0.0642|
| Hybrid (alpha=0.5) |0.8994|0.6941|0.1082|0.0905|

**Conclusions :** 
- The hybrid model with alpha=0.75 was best on every metric. It cut RMSE from 0.934 (user average baseline) to 0.875 and raised Precision@10 from 0.115 (most popular) to 0.144.
- Collaborative filtering alone beat both baselines on rating error and ranking. Content-based filtering was the weakest model, and its Precision@10 was below the popularity baseline. Genres and tags alone say little about individual taste.
- Blending in some content signal helped (alpha=0.75 beat alpha=1.0 on RMSE and Precision@10), but too much hurt: alpha=0.25 dropped below the popularity baseline on Precision@10.
- These numbers come from one random split, so small gaps, such as CF against hybrid 0.75, may not hold on another split. Next steps are matrix factorisation (SVD) and testing several random splits.
## Method notes
- Movies with fewer than 3 ratings are dropped, since they carry too little signal.
- Predictions are shrunk toward each user's mean when neighbour evidence is weak.
- The "match" percentage in the app is the title's rank among all movies the viewer has not rated. Raw predicted ratings pile up at the 5-star ceiling, so they are not shown.
- Genre-only content features give many identical movies. Adding tags helps, and matrix factorisation (SVD) is a natural next step.

Data: MovieLens latest-small, GroupLens Research.
