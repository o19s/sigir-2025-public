from abc import abstractmethod, ABC
from typing import Tuple, List, Callable, Optional, Dict

from custom_types import Score, DocId


class RankCombiner(ABC):

    @abstractmethod
    def combine(self, results: List[List[List[Tuple[DocId, Score]]]], n: int) -> List[List[Tuple[DocId, Score]]]:
        """
        Combines the results for multiple queries by multiple rankers. Returns the top n result sorted by score
        descending.
        """
        pass


class LinearScoreRankCombiner(RankCombiner):

    def __init__(self,
                 score_normalizer: List[Optional[Callable[[Tuple[DocId, Score]], Tuple[DocId, Score]]]],
                 weights: List[float]):
        """
        :param score_normalizer: optional functions for retriever-specific score normalization, in order of the retrievers
                                 pass None for a retriever where no score normalization should be applied
        :param weights: relative weights for each retriever, in order of the retrievers
        """
        self.score_normalizers = score_normalizer
        self.weights = weights

    def combine(self, results: List[List[List[Tuple[DocId, Score]]]], n: int) -> List[List[Tuple[DocId, Score]]]:
        num_retrievers = len(results)
        if num_retrievers == 0:
            return []
        if num_retrievers != len(self.score_normalizers):
            raise ValueError(f'Number of retrievers does not match number of score normalizers')
        if num_retrievers != len(self.weights):
            raise ValueError(f'Number of retrievers does not match number of relative weights')

        combined_query_results = []
        # transpose the list of retriever -> query -> [(docid, score)] to a list query -> retriever -> [(docid, score)]
        transposed = list(zip(*results))
        for query_results in transposed:
            retriever_results_processed = []
            for retriever_num, retriever_result in enumerate(query_results):
                # apply score normalization, if any
                if self.score_normalizers[retriever_num] is not None:
                    retriever_result = self.score_normalizers[retriever_num](retriever_result)
                # apply weight for this retriever
                retriever_result = LinearScoreRankCombiner.multiply_scores(retriever_result, self.weights[retriever_num])
                retriever_results_processed.append(retriever_result)
            combined_query_result = LinearScoreRankCombiner.rank_by_score_sum(retriever_results_processed)
            combined_query_results.append(combined_query_result[:n])
        return combined_query_results


    @staticmethod
    def min_max_normalization(result: List[Tuple[DocId, Score]]) -> List[Tuple[DocId, Score]]:
        scores = [score for _, score in result]
        min_score, max_score = min(scores), max(scores)
        if min_score == max_score:
            return result
        normalized_scores = []
        for doc_id, score in result:
            normalized_score = (score - min_score) / (max_score - min_score)
            normalized_scores.append((doc_id, normalized_score))
        return normalized_scores

    @staticmethod
    def rrf_score_normalization(k):

        def rrf(result: List[Tuple[DocId, Score]]) -> List[Tuple[DocId, Score]]:
            # assume result list is already sorted by score descending, so we can use index as ranks
            return list([(doc_id, (1 / (k + idx + 1))) for idx, (doc_id, score) in enumerate(result)])

        return rrf

    @staticmethod
    def multiply_scores(result: List[Tuple[DocId, Score]], weight: float) -> List[Tuple[DocId, Score]]:
        return [(doc_id, score * weight) for doc_id, score in result]

    @staticmethod
    def rank_by_score_sum(result: List[List[Tuple[DocId, Score]]]) -> List[Tuple[DocId, Score]]:
        combined_scores: Dict[DocId, Score] = {}
        for doc_id_scores in result:
            for doc_id, score in doc_id_scores:
                combined_scores[doc_id] = combined_scores.get(doc_id, 0) + score
        return sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
