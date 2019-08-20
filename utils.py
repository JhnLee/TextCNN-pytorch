import numpy as np
import json
import torch
from pathlib import Path
from gensim.models import Word2Vec


def get_preweight(w2v_path, tokenizer, rand_init=False):
    ''' gensim Word2Vec으로부터 embedding matrix 생성 '''
    w2v_model = Word2Vec.load(w2v_path)

    embedding_size = w2v_model.wv.vectors.shape[1]
    embedding = np.zeros((len(tokenizer.word2idx), embedding_size))
    for w, i in tokenizer.word2idx.items():
        try:
            embedding[i] += w2v_model.wv.vectors[w2v_model.wv.index2word.index(w)]
        except:
            if rand_init:
                embedding[i] += np.random.normal(scale=0.6, size=(embedding_size,))

    return embedding


def acc(yhat, y):
    with torch.no_grad():
        yhat = yhat.max(dim=1)[1]
        acc = (yhat == y).float().mean()
    return acc


class CheckpointManager:
    def __init__(self, model_dir):
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)
        self._model_dir = model_dir

    def save_checkpoint(self, state, filename):
        torch.save(state, self._model_dir / filename)

    def load_checkpoint(self, filename):
        state = torch.load(self._model_dir / filename)
        return state


class SummaryManager:
    def __init__(self, model_dir):
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)
        self._model_dir = model_dir
        self._summary = {}

    def save(self, filename):
        with open(self._model_dir / filename, mode='w') as io:
            json.dump(self._summary, io, indent=4)

    def load(self, filename):
        with open(self._model_dir / filename, mode='r') as io:
            metric = json.loads(io.read())
        self.update(metric)

    def update(self, summary):
        self._summary.update(summary)

    def reset(self):
        self._summary = {}

    @property
    def summary(self):
        return self._summary

