# Define variables
$S3_BUCKET_URL = 'https://sigir-2025-public.s3.eu-central-1.amazonaws.com'
$S3_BUCKET_BM25 = "$S3_BUCKET_URL/snapshots/bm25s"
$S3_BUCKET_USEARCH = "$S3_BUCKET_URL/snapshots/usearch"

$SNAPSHOTS_DIR = './snapshots'
$BM25_DIR = "$SNAPSHOTS_DIR/bm25s"
$USEARCH_DIR = "$SNAPSHOTS_DIR/usearch"

Write-Host "Downloading BM25s index, this will take quite a while..."

# Create directories if they don't exist
if (-not (Test-Path -Path $BM25_DIR -PathType Container)) {
    New-Item -Path $BM25_DIR -ItemType Directory -Force
}

# Download BM25 index files
Invoke-WebRequest -Uri "$S3_BUCKET_BM25/data.csc.index.npy" -OutFile "$BM25_DIR/data.csc.index.npy"
Invoke-WebRequest -Uri "$S3_BUCKET_BM25/indices.csc.index.npy" -OutFile "$BM25_DIR/indices.csc.index.npy"
Invoke-WebRequest -Uri "$S3_BUCKET_BM25/indptr.csc.index.npy" -OutFile "$BM25_DIR/indptr.csc.index.npy"
Invoke-WebRequest -Uri "$S3_BUCKET_BM25/params.index.json" -OutFile "$BM25_DIR/params.index.json"
Invoke-WebRequest -Uri "$S3_BUCKET_BM25/vocab.index.json" -OutFile "$BM25_DIR/vocab.index.json"

Write-Host "Downloading KNN index, this will take another while..."

# Create directory if it doesn't exist
if (-not (Test-Path -Path $USEARCH_DIR -PathType Container)) {
    New-Item -Path $USEARCH_DIR -ItemType Directory -Force
}

# Download USEARCH index file
Invoke-WebRequest -Uri "$S3_BUCKET_USEARCH/arctic-embed-l.usearch" -OutFile "$USEARCH_DIR/arctic-embed-l.usearch"

Write-Host "Downloading data file"

# Download the JSONL data file
Invoke-WebRequest -Uri "$S3_BUCKET_URL/LiveRAG_LCD_Session2_Question_file.jsonl" -OutFile "./LiveRAG_LCD_Session2_Question_file.jsonl"