import uvicorn
from fastapi import FastAPI, Request, Form
import numpy as np
import pandas as pd
import tomotopy as tp
from scipy import spatial
import numba
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path


app = FastAPI()


BASE_PATH = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_PATH / "htmldirectory"))
model = tp.LDAModel.load('recommendation.bin')
# Routes
train_df = pd.read_pickle('keywords(total_final_30)(latest).pkl')
embeddings = np.zeros((len(train_df['keywords']), model.k))
for i, doc in enumerate(model.docs):
    embeddings[i, :] = doc.get_topic_dist()
embeddings_series = pd.Series(embeddings.tolist(), index=train_df.url)


@numba.guvectorize(["void(float64[:], float64[:], float64[:])"], "(n),(n)->()", target='parallel')
async def fast_cosine_gufunc(u, v, result):
    m = u.shape[0]
    udotv = 0
    u_norm = 0
    v_norm = 0
    for i in range(m):
        if (np.isnan(u[i])) or (np.isnan(v[i])):
            continue

        udotv += u[i] * v[i]
        u_norm += u[i] * u[i]
        v_norm += v[i] * v[i]

    u_norm = np.sqrt(u_norm)
    v_norm = np.sqrt(v_norm)

    if (u_norm == 0) or (v_norm == 0):
        ratio = 1.0
    else:
        ratio = udotv / (u_norm * v_norm)
    result[:] = ratio


SIMILARITY_METRIC = {
    "fast_cosine_gufunc": fast_cosine_gufunc
}


class Recommender():
    def __init__(self, data, embeddings, user_id, num_recommendations=5, metric="fast_cosine_gufunc"):
        self._data = data
        self._user_id = user_id
        self._embeddings = embeddings
        self._num_recommendations = num_recommendations

        self.user_history = self.get_user_history()

        self.last_url = self.user_history["url"].values[0][-1]

        self.last_url_embed = embeddings[embeddings.index == self.last_url]

    def get_user_history(self):
        all_user_history = self._data.groupby(
            "user")["url"].apply(list).reset_index()
        return all_user_history[all_user_history.user == self._user_id]

    def get_similarities(self):
        sims = []
        for target_embed in self._embeddings:
            sims.append(fast_cosine_gufunc(
                self.last_url_embed.values[0], target_embed))
        return sims

    def get_recommendations(self):
        sims = self.get_similarities()
        recommendation_ids = list(
            self._embeddings.iloc[np.argsort(sims)][-self._num_recommendations:].index)
        return self.last_url, recommendation_ids


@app.get("/predict/")
async def index(request: Request):
    return templates.TemplateResponse("home.html", context={"request": request})


@app.post("/predict/")
async def fetch_predict(request: Request, user: int = Form(...)):
    recommender = Recommender(
        train_df, embeddings_series, user, num_recommendations=4, metric="fast_cosine_gufunc")
    x, recommendation_ids = recommender.get_recommendations()
    past_urls = recommender.user_history["url"].values
    urls = []
    for i in recommendation_ids:
        label_uni = train_df.loc[train_df['url'] == i].label.unique()
        urls.append(label_uni.tolist())
    return templates.TemplateResponse("abc.html", context={"request": request, "recommendation_ids": urls, "past_urls": past_urls.tolist()})


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
