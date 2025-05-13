import argparse
import pandas as pd

from corpus import FineWeb10BTCorpus
from rerank import RerankersLibraryReRanker
from torch_device import torch_device


def unzip(zipped):
    return list(map(list, zip(*zipped)))


def parse_args():
    parser = argparse.ArgumentParser(description="Reranking")
    parser.add_argument("-i", "--input_file", help="The input file, must be a Pandas readable Parquet"
                                                   " file with at least the columns 'question' and one int[] column"
                                                   " representing the docid of the corpus documents that should be reranked ", required=True)
    parser.add_argument("-o", "--output_file", help="The output file.", required=True)
    parser.add_argument('-d', "--docid_column", help='The (input) passage or doc ID column')
    parser.add_argument('-s', "--score_column", help='The (input) passage or doc score column')
    parser.add_argument("-k", "--k", help="The number of results to consider when reranking", required=False, type=int, default=100)
    parser.add_argument("-n", "--limit", help="The top N results to export", required=False, type=int, default=40)
    return parser.parse_args()

RERANKERS = {
    "inranker-base":    lambda: RerankersLibraryReRanker(model_name='unicamp-dl/InRanker-base', device=torch_device(), dtype="bfloat16"),
}

if __name__ == '__main__':
    # take retrieved passages and rerank TOP n to increase precision
    # input: question, doc_ids, scores
    args = parse_args()

    print("Loading Corpus...")
    corpus = FineWeb10BTCorpus()

    print("Reading results file...")
    df = pd.read_parquet(args.input_file)

    output_dict = {
        "id": df['id'],
        "question": df['question']
    }

    results = [list(zip(ids, scores))
               for ids, scores in zip(df[args.docid_column], df[args.score_column])]

    for reranker_name, reranker_ctr in RERANKERS.items():
        reranked_results = reranker_ctr().rerank(corpus, df['question'], results, n=args.k)
        output_dict[f"{reranker_name}_ids"], output_dict[f"{reranker_name}_scores"] = unzip(
            [unzip(idx_with_score) for idx_with_score in reranked_results]
        )

    pd.DataFrame(output_dict).to_parquet(args.output_file, index=False)

