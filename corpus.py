from abc import ABC, abstractmethod
from typing import Generator, Tuple, Optional, Dict, List

from datasets import load_dataset

from custom_types import FieldName, DocId


class Corpus(ABC):

    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def get(self, idx: int) -> Dict[FieldName, any]:
        pass

    @abstractmethod
    def iterator(self) -> Generator[Tuple[DocId, Dict[FieldName, any]], None, None]:
        pass

class HuggingFaceCorpus(Corpus):
    """simple wrapper around HuggingFace dataset, mainly adding the iterator that stops
       at a specified sample size"""

    def __init__(self, name: str,  config_name: str, split_name: str, fields: List[str],
                 sample_size: Optional[int] = None):
        self.ds = load_dataset(path=name, name=config_name, split=split_name, streaming=False)
        self.fields = fields
        ds_length = len(self.ds)
        self.num_items = ds_length if sample_size is None else min(sample_size, ds_length)

    def size(self) -> int:
        return self.num_items

    def get(self, idx: int) -> Dict[FieldName, any]:
        return self.ds[idx]

    def iterator(self, skip=0) -> Generator[Tuple[DocId, Dict[FieldName, any]], None, None]:
        """Corpus iterator yields tuples of identifiers and documents, optionally skips the first "skip" elements"""
        total = 0
        for row in self.ds:
            if total == self.num_items:
                break
            if total >= skip:
                # just use the dataset index as id
                yield total, {field: row[field] for field in self.fields}
            total += 1

def FineWeb10BTCorpus(sample_size: Optional[int]=None) -> HuggingFaceCorpus:
    return HuggingFaceCorpus(name="HuggingFaceFW/fineweb", config_name="sample-10BT", split_name="train",
                             fields=['id', 'text', 'date', 'url'], sample_size=sample_size)

def WikipediaEnCorpus(sample_size: Optional[int]=None) -> HuggingFaceCorpus:
    return HuggingFaceCorpus(name="wikimedia/wikipedia", config_name="20231101.en", split_name="train",
                             fields=['text'], sample_size=sample_size)
