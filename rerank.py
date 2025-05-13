from abc import ABC, abstractmethod
from typing import List, Tuple

from rerankers import Reranker, Document

from corpus import Corpus
from custom_types import DocId, Score


class ReRanker(ABC):

    @abstractmethod
    def rerank(self, corpus: Corpus, queries: List[str], results: List[List[Tuple[DocId, Score]]], n: int):
        """
        Rerank the first n results for the given corpus and queries, the results list must contain the query-specific
        results in the same order as the queries list.
        """
        pass


class RerankersLibraryReRanker(ReRanker):
    """A reranker that delegates the task to the rerankers library
    """

    def __init__(self, **kwargs):
        self.ranker = Reranker(**kwargs)
        # self.chunker = ChunkNorris()

    def rerank(self, corpus: Corpus, queries: List[str], results: List[List[Tuple[DocId, Score]]], n: int):
        from tqdm import tqdm

        reranked = []
        for query, query_result in tqdm(zip(queries, results)):
            # Create sentence-level splits
            # colbert_documents = self.generate_document_sentence_split(corpus, [doc_id for doc_id, _ in query_result[:n]])

            # char length split
            passage_docs = [doc
                            for doc_id, score in query_result[:n]
                            for doc in self.generate_documents_basic_char_length_split(corpus, int(doc_id))]

            print(f"Re-ranking {len(passage_docs)} documents")
            ranked = [(ranked_result.document.metadata['corpus_id'], ranked_result.score)
                      for ranked_result in self.ranker.rank(query=query, docs=passage_docs).results]
            # if one document is present with multiple chunks in the result, we only take the first one
            seen = set()
            ranked_deduplicated = []
            for id, score in ranked:
                if id not in seen:
                    ranked_deduplicated.append((id, score))
                    seen.add(id)
            reranked.append(ranked_deduplicated)
        return reranked

    def generate_document_sentence_split(self, corpus, corpus_ids: List[DocId]):
        texts = [corpus.get(int(corpus_id))['text'] for corpus_id in corpus_ids]
        docs_chunks = self.chunker.split_sentence_level(texts)
        colbert_documents = []
        for corpus_id, doc_chunks in zip(corpus_ids, docs_chunks):
            for idx, doc_chunk in enumerate(doc_chunks):
                colbert_document = Document(
                    doc_id=f"{corpus_id}_{idx}",
                    text=doc_chunk,
                    metadata={'corpus_id': corpus_id}
                )
                colbert_documents.append(colbert_document)
        return colbert_documents

    def generate_documents_basic_char_length_split(self, corpus, corpus_id):
        text = corpus.get(corpus_id)['text']
        chunks = self.basic_split_string_with_overlap(text)
        return [
            Document(
                doc_id=f"{corpus_id}_{idx}",
                text=chunk,
                metadata={
                    "corpus_id": corpus_id,
                }
            )
            for idx, chunk in enumerate(chunks)
        ]


    def basic_split_string_with_overlap(self, text, chunk_size=1500, overlap=250):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

