import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from torch_device import torch_device

# the model-specific max. context length, default: use the transformers library default which is reading it from the model metadata
MODEL_MAX_LENGTH = {
    "Snowflake/snowflake-arctic-embed-l-v2.0": 8192,
}

# the model-specific formatting of a query for a passage (query is used verbatim if not given)
MODEL_QUESTION_FORMATTER = {
    "Snowflake/snowflake-arctic-embed-l-v2.0":  lambda s: f"query: {s}",
}

MODEL_PASSAGE_FORMATTER = {
}

MODEL_KWARGS = {
    "Snowflake/snowflake-arctic-embed-l-v2.0": {
        "torch_dtype": torch.bfloat16,
        "add_pooling_layer": False
    }
}

def default_string_formatter(s):
    return s

class EmbeddingModel:

    def __init__(self, model_name: str, device: str = torch_device(), local_models_path: Optional[str] = None):
        self.question_formatter = MODEL_QUESTION_FORMATTER.get(model_name, default_string_formatter)
        self.passage_formatter = MODEL_PASSAGE_FORMATTER.get(model_name, default_string_formatter)
        self.device = device
        self.local_models_path = local_models_path if local_models_path else os.getenv('LOCAL_MODELS_PATH')
        self.max_length = MODEL_MAX_LENGTH.get(model_name, None)
        model_kwargs = MODEL_KWARGS.get(model_name, {})
        self.torch_dtype = model_kwargs.get("torch_dtype", None)

        print(f'Using {self.device} for {model_name}, overrides: dtype={self.torch_dtype}, max_length={self.max_length}')

        if model_name.startswith("s3://"):
            # HF transformers does not support loading models directly from S3, we download an S3 model to a local path
            model_name = self.mirror_model_from_s3(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, **model_kwargs).to(self.device)

    def tokenize_and_embed(self, strings: List[str], is_query: bool = False) -> torch.Tensor:
        # apply a model-specific question/passage formatter
        if is_query:
            strings = [self.question_formatter(s) for s in strings]
        else:
            strings = [self.passage_formatter(s) for s in strings]

        tokenizer_kwargs = {'max_length': self.max_length} if self.max_length is not None else {}
        encoded_input = self.tokenizer(strings, padding=True, truncation=True, return_tensors='pt', **tokenizer_kwargs).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        dtype = self.torch_dtype if self.torch_dtype is not None else torch.float32
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(dtype)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def mirror_model_from_s3(self, model_path) -> str:
        from file_io import mirror_s3_to_local
        return mirror_s3_to_local(model_path, self.local_models_path)