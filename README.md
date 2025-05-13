# SIGIR 2025 LiveRAG Challenge

Some code to process the SIGIR 2025 LiveRAG challenge. It has been tested on M1 Mac and Windows (with CUDA).

## Setup

### Setup virtual environment

#### Mac
```
python -m venv .venv
source .venv/bin/activate
# system-specific pytorch not part of the requirements.txt
pip install torch==2.7.0
pip install -r requirements.txt
```

#### Windows Powershell
```
python -m venv .venv
.venv\Scripts\Activate.ps1
# pytorch for CUDA 12.8 according to https://pytorch.org/get-started/locally/
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Environment Variables

Copy `.env-template` to `.env` and replace the respective values.

### Download data

#### Index Snapshots

We use bm25s for BM25 retrieval and Snowflake/arctic-embed-l embeddings  in a usearch kNN index for retrieval.

You don't need to re-create these embeddings/indices. We have a prebuilt version for download.
The download is about 60GB, so will take some time.

#### Mac
```
./01_download.sh
```

#### Windows
```
./01_download.ps1
```

## Run Processing

The challenge is run in 4 steps:

1. Retrieval: BM25 and kNN results of the original question and a Falcon-generated HyDE passage
2. Result fusion: RRF of the 4 previously retrieved result sets
3. Reranking: Re-ranking the fused results using a reranker model
4. Answer generation

Run the script to execute them all sequentially.

#### Mac
```
./02_run.sh
```

#### Windows
```
./02_run.ps1
```

Note that the reranking step is using unicamp-dl/InRanker-base which is slow  when running on a non-CUDA platform.

The resulting file will be liverag_step4.jsonl. There are .parquet files for the intermediate results.
