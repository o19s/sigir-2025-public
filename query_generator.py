import json
import os
from typing import List

from ai71 import AI71
from dotenv import load_dotenv
from tqdm import tqdm

from llm import FalconLlm

SYSTEM_PROMPT = """
You are an expert in many topics.
"""

USER_PROMPT = """
You are given a topic. Produce a creative sample paragraph from a website loosely related to that topic. Imagine it
to be a paragraph from an online discussion forum or an online news source. Evaluate your generated paragraph and
include your reasoning in your response. You will be rated for creativity.

Please respond with the following JSON format:
```
{
  "reasoning": "<your-reasoning>",
  "paragraph": "<your-paragraph>"
}
```

This is the topic:
{{question}}
"""

def generate_prompt(question: str) -> str:
    return USER_PROMPT.replace('{{question}}', question)

def generate_paragraphs_for_questions(llm_ctr, questions: List[str]) -> List[str]:
    llm = llm_ctr()
    generated = []
    responses = llm.generate_multiple(prompts=[generate_prompt(question) for question in questions], max_parallel=6)
    for idx, (response, _) in enumerate(responses):
        try:
            js = json.loads(response)
            generated.append(js["paragraph"] if "paragraph" in js else '')
        except Exception as e:
            print("Could not parse response:", e)
            print(response)
            # use the original question in the error case for consistency
            generated.append(questions[idx])
    return generated

def generate_paragraphs(questions: List[str]) -> List[str]:
    load_dotenv()
    llm_ctr = lambda: FalconLlm(ai71_client=AI71(api_key=os.getenv("AI71_API_KEY"), base_url=os.getenv("AI71_BASE_URL")),
                                system_prompt=SYSTEM_PROMPT)
    return generate_paragraphs_for_questions(llm_ctr, questions)


