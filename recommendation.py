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
train_df = train_df.drop_duplicates(subset=['url', 'user'], keep="last")
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


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


SIMILARITY_METRIC = {
    "fast_cosine_gufunc": fast_cosine_gufunc
}


class Recommender():
    def __init__(self, data, embeddings, user_id, url_id, num_recommendations=2, metric="fast_cosine_gufunc"):
        self._data = data
        self._user_id = user_id
        self.url_id = url_id
        self._embeddings = embeddings
        self.metric = SIMILARITY_METRIC[metric]
        self._num_recommendations = num_recommendations

        self.user_history = self.get_user_history()
        self.last_url = self.user_history["url"].values
        self.last_url[0].append(url_id)
        countero = len(self.last_url[0])
        self.stack = []
        for i in range(countero):
            globals()["last_url_embed" + str(i)
                      ] = embeddings[embeddings.index == self.last_url[0][i]]
            self.stack.append(globals()["last_url_embed" + str(i)])
        self.final_df = pd.concat(self.stack, axis=0)

    def get_user_history(self):
        all_user_history = self._data.groupby(
            "user")["url"].apply(list).reset_index()
        return all_user_history[all_user_history.user == self._user_id]

    def get_similarities(self):
        sims = []
        for target_embed in self._embeddings:
            for i, j in enumerate(self.final_df.index.unique()):
                z = self.final_df.iloc[self.final_df.index == j].values[0]
                globals()["sims" + str(i)
                          ] = fast_cosine_gufunc(z, target_embed)
                sims.append(globals()["sims" + str(i)])
        return sims

    def get_recommendations(self):
        sims = self.get_similarities()
        sims = list(split(sims, len(self.stack)))
        recommendation_ids_stack = []
        for i in sims:
            globals()["recommendation_ids" + str(i)] = list(
                self._embeddings.iloc[np.argsort(i)][-self._num_recommendations:].index)
            recommendation_ids_stack.append(
                globals()["recommendation_ids" + str(i)])
        return self.last_url, recommendation_ids_stack


@app.get("/predict/")
async def index(request: Request):
    return templates.TemplateResponse("home.html", context={"request": request})


@app.post("/predict/")
async def fetch_predict(request: Request, user: int = Form(...), url_id: int = Form(...)):
    recommender = Recommender(
        train_df, embeddings_series, user, url_id, num_recommendations=2, metric="fast_cosine_gufunc")
    x, recommendation_id = recommender.get_recommendations()
    past_urls = recommender.last_url[0]
    urls = []
    # for i in recommendation_id:
    #     label_uni = train_df.loc[train_df['url'] == i].label.unique()
    #     urls.append(label_uni.tolist())
    return templates.TemplateResponse("abc.html", {"request": request, "recommendation_ids": recommendation_id, "past_urls": past_urls})


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
