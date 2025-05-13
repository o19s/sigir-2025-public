import argparse

import numpy as np
from tqdm import tqdm
from usearch.compiled import MetricKind
from usearch.index import Index

from array_io import batch_embeddings_loader
from file_io import Fs, LocalFs

def create_usearch_index(index_file: str, dimensions: int, fs: Fs, file_path_pattern: str,
                         m: int = 32, ef_construction: int = 384, ef_search: int = 1024):
    index = Index(ndim=dimensions, metric=MetricKind.IP, connectivity=m, expansion_add=ef_construction,
                  expansion_search=ef_search)
    progress_bar = tqdm(unit=' vectors indexed')
    for start_idx, vectors in batch_embeddings_loader(fs, file_path_pattern, 1024):
        ids = np.arange(start_idx, start_idx + len(vectors))
        index.add(ids, vectors)
        progress_bar.update(len(vectors))
    index.save(index_file)

def parse_args():
    parser = argparse.ArgumentParser(description="Usearch Indexer")
    parser.add_argument("-f", "--file", help=f"The index file to create", required=True)
    parser.add_argument("-d", "--dim", help=f"The number of dimensions for vectors in this collection (e.g. 512)", required=True, type=int)
    parser.add_argument("-p", "--path", help="The path pattern to read from, needs to contain the {i} placeholder for numerical iteration, e.g. /embeddings/bge-base-fineweb_{i}_*", required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    create_usearch_index(index_file=args.file, dimensions=args.dim, fs=LocalFs(), file_path_pattern=args.path)
