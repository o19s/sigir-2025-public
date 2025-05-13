import os
from abc import ABC, abstractmethod
from typing import List, Tuple

import Stemmer
import bm25s
import torch
from usearch.compiled import MetricKind
from usearch.index import Index

from corpus import Corpus
from custom_types import DocId, Score
from embeddings import EmbeddingModel


class Retriever(ABC):

    @abstractmethod
    def retrieve(self, keywords: List[str], size: int) -> List[List[Tuple[DocId, Score]]]:
        pass


class BM25SRetriever(Retriever):
    """A retriever using the bm25s library, note: creating the full Fineweb index using that library requires
    a huge amount of resources, it barely finished with 128GB RAM"""

    def __init__(self, corpus: Corpus, text_field: str, index_dir: str):
        self.stemmer = Stemmer.Stemmer("english")
        self.stopword_language = "en"
        if not os.path.exists(index_dir):
            print(f"Building new BM25s index in: {index_dir}")
            self.bm25index = self.create_index(corpus, text_field, index_dir)
        else:
            print(f"Reusing BM25s index from: {index_dir}")
            self.bm25index = bm25s.BM25.load(index_dir)

    def retrieve(self, keywords: List[str], size: int = 40) -> List[List[Tuple[DocId, Score]]]:
        keywords_tokenized = bm25s.tokenize(keywords, stopwords=self.stopword_language, stemmer=self.stemmer)
        result = self.bm25index.retrieve(query_tokens=keywords_tokenized, k=size)
        documents, scores = result
        # transpose to our format
        return [[(doc.item(), score.item()) for doc, score in zip(doc_list, score_list)] for doc_list, score_list in
                zip(documents, scores)]

    def score_single(self, keywords: str, doc_id: DocId) -> Score:
        print(f"Scoring for {doc_id}: {keywords}")
        keywords_tokenized = \
            bm25s.tokenize(keywords, stopwords=self.stopword_language, stemmer=self.stemmer, return_ids=False)[0]
        scores = self.bm25index.get_scores(keywords_tokenized)[doc_id].item()
        return scores

    def create_index(self, corpus: Corpus, text_field: str, index_dir: str):

        corpus_tokens = bm25s.tokenize([doc[text_field] for id, doc in corpus.iterator()],
                                       stopwords=self.stopword_language, stemmer=self.stemmer)
        retriever = bm25s.BM25(backend="numba")
        retriever.index(corpus_tokens)
        retriever.save(index_dir)
        return retriever

class UsearchRetriever(Retriever):
    """A retriever using an _existing_ Usearch index"""

    def __init__(self, index_path: str, n_dim: int, embedding_model: EmbeddingModel):
        print(f"Loading Usearch index from {index_path}")
        self.index = Index(ndim=n_dim, metric=MetricKind.IP, expansion_search=1024)
        self.index.load(index_path)
        self.embedding_model = embedding_model

    def retrieve(self, keywords: List[str], size: int = 40) -> List[List[Tuple[DocId, Score]]]:
        query_embeddings = self.embedding_model.tokenize_and_embed(keywords, is_query=True).float()
        matches = self.index.search(query_embeddings.numpy(force=True), count=size)
        if len(keywords) == 1:
            # special case where usearch already does the unwrapping
            return [[(doc_id, doc_score) for doc_score, doc_id in zip(matches.distances, matches.keys)]]
        else:
            return [[(doc_id, doc_score) for doc_score, doc_id in zip(query_scores, query_indices)]
                    for query_scores, query_indices in zip(matches.distances, matches.keys)]

def cos_sim(a: torch.Tensor, b: torch.Tensor):
    # assumes tensors are normalized already
    return torch.mm(a, b.transpose(0, 1))
