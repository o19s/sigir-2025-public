# Set environment variable (PowerShell doesn't directly use 'export' in the same way)
$env:PYTHONUNBUFFERED = 1

# Run Python scripts
python runstep_1_retrieval.py -qf LiveRAG_LCD_Session2_Question_file.jsonl -rf liverag_step1.parquet -bm25s "./snapshots/bm25s" -usearch "./snapshots/usearch"
python runstep_2_result_fusion.py -i liverag_step1.parquet -r bm25-bm25s -r bm25-bm25s_hyde -r usearch-arctic-embed-l -r usearch-arctic-embed-l_hyde -o liverag_step2.parquet
python runstep_3_reranking.py -i liverag_step2.parquet -d rrf_k79_ids -s rrf_k79_scores -k 100 -n 40 -o liverag_step3.parquet
python runstep_4_answer_generation.py -i liverag_step3.parquet -o liverag_step4.parquet --r inranker-base