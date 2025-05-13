#!/bin/bash
set -x

S3_BUCKET_URL='https://sigir-2025-public.s3.eu-central-1.amazonaws.com'
S3_BUCKET_BM25="${S3_BUCKET_URL}/snapshots/bm25s"
S3_BUCKET_USEARCH="${S3_BUCKET_URL}/snapshots/usearch"

SNAPSHOTS_DIR='./snapshots'
BM25_DIR="${SNAPSHOTS_DIR}/bm25s"
USEARCH_DIR="${SNAPSHOTS_DIR}/usearch"

echo "Downloading BM25s index, this will take quite a while..."

mkdir -p "${BM25_DIR}"
curl -o "${BM25_DIR}/data.csc.index.npy" "${S3_BUCKET_BM25}/data.csc.index.npy"
curl -o "${BM25_DIR}/indices.csc.index.npy" "${S3_BUCKET_BM25}/indices.csc.index.npy"
curl -o "${BM25_DIR}/indptr.csc.index.npy" "${S3_BUCKET_BM25}/indptr.csc.index.npy"
curl -o "${BM25_DIR}/params.index.json" "${S3_BUCKET_BM25}/params.index.json"
curl -o "${BM25_DIR}/vocab.index.json" "${S3_BUCKET_BM25}/vocab.index.json"

echo "Downloading KNN index, this will take another while..."
mkdir -p "${USEARCH_DIR}"
curl -o "${USEARCH_DIR}/arctic-embed-l.usearch" "${S3_BUCKET_USEARCH}/arctic-embed-l.usearch"

echo "Downloading data file"
curl -o ./LiveRAG_LCD_Session2_Question_file.jsonl "${S3_BUCKET_URL}/LiveRAG_LCD_Session2_Question_file.jsonl"
