import argparse

import numpy as np
import pandas as pd

from rank_combiner import LinearScoreRankCombiner


def unzip(zipped):
    return list(map(list, zip(*zipped)))


def parse_args():
    parser = argparse.ArgumentParser(description="Result fusion")
    parser.add_argument("-i", "--input_file", help="The input file, must be a Pandas readable Parquet"
                                                   " file with at least the columns 'question' and one int[] column"
                                                   " representing the docid for each retriever you want to fuse ", required=True)
    parser.add_argument("-o", "--output_file", help="The output file.", required=True)
    parser.add_argument('-r', "--retrievers", action='append', help='The retriever names to fuse'
                                                                    ' (use multiple -r parameters to provide their list),'
                                                                    ' there must be a <retriever>_ids column for each retriever'
                                                                    ' in the Parquet input file.')
    parser.add_argument("-k", "--rrf_k", action="append", help="The k parameters for the RRF formula, we will create"
                                                               " the columns rrf_{k}_ids and rrf_{k}_scores for each k, respectively",
                        required=False, type=int, default=range(1,81))
    parser.add_argument("-n", "--limit", help="The top N results to export", required=False, type=int, default=100)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    print("Reading results file...")
    df = pd.read_parquet(args.input_file)

    # unify into one id type, strangely we got uint64 and int64 in the input
    id_column_names = [f"{retriever}_ids" for retriever in args.retrievers]
    for id_column in id_column_names:
        df[id_column] = df[id_column].apply(lambda value: value.astype(np.uint64))

    # pass through the question, add the fused results (and the RRF scores) for each question
    output_dict = {
        "id": df['id'],
        "question": df['question']
    }

    for k in args.rrf_k:
        print(f"Fusion top {args.limit} results of {args.retrievers} with k={k}")
        score_normalizers = [LinearScoreRankCombiner.rrf_score_normalization(k) for _ in args.retrievers]
        weights = [1 for _ in args.retrievers]
        combiner = LinearScoreRankCombiner(score_normalizers, weights)

        # re-zip the List[List[List[Tuple[DocId, Score]]]] required for the combiner
        results_for_retrievers = [
            [
                list(zip(question_ids, question_scores))
                for question_ids, question_scores in zip(df[f"{retriever}_ids"], df[f"{retriever}_scores"])
            ]
            for retriever in args.retrievers
        ]

        fused_results = combiner.combine(results_for_retrievers, n=args.limit)
        output_dict[f"rrf_k{k}_ids"], output_dict[f"rrf_k{k}_scores"] = unzip([unzip(idx_with_score) for idx_with_score in fused_results])

    # pass in the type of the ids explicitly
    pd.DataFrame(output_dict).to_parquet(args.output_file, index=False)
