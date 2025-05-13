import time
from abc import abstractmethod, ABC
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Tuple, Optional, List

from ai71 import AI71
from tqdm import tqdm


@dataclass
class TokenUsage:
    input_tokens: int
    input_cost: float
    output_tokens: int
    output_cost: float

    @staticmethod
    def empty():
        return TokenUsage(input_tokens=0, input_cost=0, output_tokens=0, output_cost=0)

def unwrap(response_content, leading, trailing):
    return (response_content[len(leading):-len(trailing)]
            if response_content.startswith(leading) and response_content.endswith(trailing)
            else response_content)

def unwrap_backticks(response_content):
    # unwrap this LLM quirk in code-generating LLM responses
    s = response_content
    s = unwrap(s, "```json", "```")
    s = unwrap(s, "```", "```")
    s = s[3:] if s.startswith("```") else s
    s = s[:-3] if s.endswith("```") else s
    return s

def remove_assistant_token(response_content):
    return response_content[len("<|assistant|>"):].strip() if response_content.startswith("<|assistant|>") else response_content

class Llm(ABC):

    @abstractmethod
    def generate(self, prompt: str) -> Tuple[str, TokenUsage]:
        pass

    def generate_multiple(self, prompts: List[str], max_parallel = 10) -> List[Tuple[str, TokenUsage]]:
        if not prompts:
            return []
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            return list(tqdm(executor.map(self.generate, prompts), total=len(prompts)))

    def close(self):
        pass

class FalconLlm(Llm):

    def __init__(self, ai71_client: AI71, system_prompt: Optional[str] = None,
                 model: str = "tiiuae/falcon3-10b-instruct", max_tokens: int = 1024, n_retries: int = 5):
        self.ai71_client = ai71_client
        self.system_prompt = system_prompt
        self.model = model
        self.max_tokens = max_tokens
        self.n_retries = n_retries

    def generate(self, prompt: str) -> Tuple[str, TokenUsage]:
        retries = 0
        while True:
            try:
                system_prompt_messages = [] if self.system_prompt is None else [
                    {
                        "role": "system",
                        "content": self.system_prompt
                     }
                ]
                user_messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
                response = self.ai71_client.chat.completions.create(
                    model=self.model,
                    messages=system_prompt_messages + user_messages,
                    max_tokens=self.max_tokens)
                response_content = response.choices[0].message.content
                response_content = remove_assistant_token(response_content)
                response_content = unwrap_backticks(response_content)
                return response_content, TokenUsage.empty()
            except Exception as e:
                retries += 1
                if retries > self.n_retries:
                    raise e
                print(f"Retrying for the {retries} time(s)... (error: {e})")
                time.sleep(retries * 10)
