import argparse
import json
import os
from typing import List, Tuple

import pandas as pd
from dotenv import load_dotenv

from corpus import FineWeb10BTCorpus
from embeddings import EmbeddingModel
from query_generator import generate_paragraphs
from retriever import BM25SRetriever, UsearchRetriever


def unzip(zipped):
    return list(map(list, zip(*zipped)))

def read_questions(filename) -> Tuple[List[str], List[int]]:
    if filename.endswith('.csv'):
        questions = pd.read_csv(filename)['question'].tolist()
        # no id in input, just use a 1-based index
        question_ids = list(range(1, len(questions) + 1))
        return questions, question_ids
    elif filename.endswith('.jsonl'):
        questions: List[str] = []
        question_ids: List[int] = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                questions.append(data['question'])
                question_ids.append(data['id'])
        return questions, question_ids
    else:
        raise ValueError('Unsupported file type')


def parse_args():
    # read defaults from environment
    env_file = os.getenv("ENV_FILE")
    if env_file:
        load_dotenv(dotenv_path=env_file)
    bm25s_default_index_path = os.getenv("BM25S_INDEX_PATH", default="./fineweb-bm25s")
    usearch_default_indexes_path = os.getenv("USEARCH_INDEXES_PATH", default="./usearch-indexes")

    parser = argparse.ArgumentParser(description="Retrieval runner")
    parser.add_argument("-qf", "--questions_file", help=f"The list of questions, must be a CSV with a 'question' column.",
                        required=False, default='questions_answers.csv')
    parser.add_argument("-rf", "--results_file", help="The results file.", required=False, default="retrieval_results.parquet")
    parser.add_argument("-bm25s", "--bm25s", help="The BM25S index directory (when using bm25s for BM25)",
                        required=False, default=bm25s_default_index_path)
    parser.add_argument("-usearch", "--usearch", help="The directory for usearch indices (when using usearch for KNN)",
                        required=False, default=usearch_default_indexes_path)
    parser.add_argument("-n", "--limit", help="The top N number of documents to retrieve. (default=1000)",
                        required=False, type=int, default=1000)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # model names
    arctic_embed_model = "Snowflake/snowflake-arctic-embed-l-v2.0"

    print("Reading questions...")
    questions, question_ids = read_questions(args.questions_file)

    print("Generating alternative queries...")
    paragraphs = generate_paragraphs(questions)

    print("Loading Corpus...")
    corpus = FineWeb10BTCorpus()

    retrievers = {
        "bm25-bm25s":
            lambda: BM25SRetriever(corpus, "text",  args.bm25s),
        "usearch-arctic-embed-l":
            lambda: UsearchRetriever(index_path=f"{args.usearch}/arctic-embed-l.usearch", n_dim=1024,
                                     embedding_model=EmbeddingModel(arctic_embed_model))
    }

    bm25_retriever = "bm25-bm25s"
    knn_retrievers = ["usearch-arctic-embed-l"]

    # evaluate these retrievers individually, do the two different bm25 ones to compare their results
    retrievers_to_run = knn_retrievers + [bm25_retriever]

    retriever_results = {}

    for retriever_name in retrievers_to_run:
        retriever = retrievers[retriever_name]()
        retriever_results[retriever_name] = retriever.retrieve(questions, size=args.limit)
        # run against alternative queries
        retriever_results[f"{retriever_name}_hyde"] = retriever.retrieve(paragraphs, size=args.limit)
        # explicitly delete it, might help GC
        del retriever

    all_results = {
        "id": question_ids,
        "question": questions,
    }

    for retriever_name, results in retriever_results.items():
        # split result ids off scores
        col_result_ids = f"{retriever_name}_ids"
        col_result_scores = f"{retriever_name}_scores"
        all_results[col_result_ids], all_results[col_result_scores] = unzip([unzip(idx_with_score) for idx_with_score in results])

    pd.DataFrame(all_results).to_parquet(args.results_file, index=False)
