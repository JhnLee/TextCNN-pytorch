from gensim.models import Word2Vec
from multiprocessing import cpu_count
from tokenizer import SentencePieceTokenizer
import pandas as pd


class Pretrain_wv:
    def __init__(self, data_path,
                 tokenizer_path,
                 save_path,
                 min_count=1,
                 window_size=10,
                 wordvec_dim=150,
                 num_iter=50,
                 negative=5,
                 sample=1e-5,
                 sg=True):

        data = pd.read_csv(data_path)
        s = SentencePieceTokenizer(tokenizer_path)

        self.corpus = [s.tokenize(tt) for tt in data.Sentence]
        self.num_iter = num_iter
        self.save_path = save_path
        self.model = Word2Vec(self.corpus,
                              min_count=min_count,
                              window=window_size,
                              size=wordvec_dim,
                              sg=sg,
                              negative=negative,
                              sample=sample,
                              workers=cpu_count() - 1)


    def train(self):
        total_words = len(self.model.wv.index2word)
        self.model.train(self.corpus, total_words=total_words, epochs=self.num_iter)
        self.model.save(self.save_path)


def main():
    data_path = './datasets/korean_single_turn_utterance_text.csv'
    c = Pretrain_wv(data_path, tokenizer_path='m32k.model', save_path='w2v_pretrained', num_iter=100)
    c.train()

if __name__ == '__main__':
    main()
