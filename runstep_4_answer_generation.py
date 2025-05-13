import argparse
import os
from collections import defaultdict
from typing import Tuple

import pandas as pd
from ai71 import AI71
from dotenv import load_dotenv
from tqdm import tqdm

from corpus import FineWeb10BTCorpus
from llm import FalconLlm

SYSTEM_PROMPT = """
You are an expert. You answer questions truthfully based on provided documents.You are talkative and provide lots of specific details from the context.
For each document check whether it is related to the question.
Only use documents that are related to the question to answer it.
Ignore documents that are not related to the question.
If the answer exists in several documents, summarize them.
Only answer based on the documents provided. Don't make things up.
Always use references in the form [NUMBER OF DOCUMENT] when using information from a document. e.g. [3], for Document[3].
The reference must only refer to the number that comes in square brackets before the passage.
Otherwise, do not use brackets in your answer and reference ONLY the number of the passage without mentioning the word passage.
If the documents can't answer the question or you are unsure say: 'The answer can't be found in the text'.
"""

USER_PROMPT = """
These are the documents:
===
{{passages}}
===
Question: {{question}}
"""


def generate_answer(llm, question, passages) -> Tuple[str, str]:
    passages_in_prompt = "\n---\n".join([f"Document [{i+1}]:\n{passage}\n" for i, passage in enumerate(passages)])
    prompt = USER_PROMPT.replace("{{passages}}", passages_in_prompt).replace("{{question}}", question)
    answer, token_usage = llm.generate(prompt=prompt)
    return prompt, answer

def get_passages_for_doc_ids(doc_ids, first_max_chars: int = 4096):
    passages_with_doc_urns = []
    for doc_id in doc_ids:
        corpus_doc = corpus.get(int(doc_id))
        urn = corpus_doc['id']
        passage = corpus_doc['text'][:first_max_chars]
        passages_with_doc_urns.append((urn, passage))
    return passages_with_doc_urns

def parse_args():
    # read defaults from environment, always load .env (not checked-in, i.e. with secrets)
    load_dotenv()
    env_file = os.getenv("ENV_FILE")
    if env_file:
        load_dotenv(dotenv_path=env_file)
    parser = argparse.ArgumentParser(description="Answer Generation")
    parser.add_argument("-i", "--input_file", help="The input file, must be a Pandas readable Parquet"
                                                   " file with at least the columns 'question' and one int[] column"
                                                   " representing the docid of the corpus documents that should be used for generation ", required=True)
    parser.add_argument("-o", "--output_file", help="The output file.", required=True)
    parser.add_argument('-r', "--result", help='The result to use. There must be a <result>_ids column in the input file.', required=True)
    parser.add_argument("-k", "--k", help="The number of results to consider when generating", required=False, type=int, default=10)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    llm = FalconLlm(ai71_client=AI71(api_key=os.getenv("AI71_API_KEY"), base_url=os.getenv("AI71_BASE_URL")),
                    system_prompt=SYSTEM_PROMPT)

    print("Loading Corpus...")
    corpus = FineWeb10BTCorpus()

    print("Reading results file...")
    df = pd.read_parquet(args.input_file)

    results = defaultdict(list)
    all_passages = []
    all_prompts = []
    all_answers = []
    print("Generating answers...")
    for id, question, doc_ids in tqdm(list(zip(df['id'], df['question'], df[f"{args.result}_ids"]))):
        for num_docs, num_chars in [(5, 8192), (10, 4096), (20, 2048), (4, 10240), (30, 1536)]:
            urns_and_passages = get_passages_for_doc_ids(doc_ids[:num_docs], num_chars)
            prompt, answer = generate_answer(llm, question, [passage for _, passage in urns_and_passages])
            if not "The answer can't be found" in answer:
                break

        results['id'].append(id)
        results['question'].append(question)
        results['passages'].append([{'passage': passage, 'doc_IDs': [urn]} for urn, passage in urns_and_passages])
        results['final_prompt'].append(SYSTEM_PROMPT + " " + prompt)
        results['answer'].append(answer)

    df = pd.DataFrame.from_dict(results)
    df.to_parquet(args.output_file, index=False)

    # also output to a JSONL file
    jsonl_file_name = args.output_file.rsplit('.', 1)[0] + ".jsonl"
    df.to_json(jsonl_file_name, orient='records', lines=True, force_ascii=False)
