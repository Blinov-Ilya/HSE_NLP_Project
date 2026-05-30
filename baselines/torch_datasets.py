import re
from collections import Counter
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import Dataset


TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())


def build_vocab(
    texts: Iterable[str],
    max_size: int = 50_000,
    min_freq: int = 2,
) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in counter.most_common(max_size - len(vocab)):
        if count < min_freq:
            continue
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    unk_id = vocab[UNK_TOKEN]
    ids = [vocab.get(token, unk_id) for token in tokenize(text)[:max_len]]
    if len(ids) < max_len:
        ids.extend([vocab[PAD_TOKEN]] * (max_len - len(ids)))
    return ids


class TextSequenceDataset(Dataset):
    def __init__(self, texts, labels: np.ndarray, vocab: Dict[str, int], max_len: int):
        self.texts = list(texts)
        self.labels = labels.astype(np.float32)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        input_ids = encode_text(self.texts[index], self.vocab, self.max_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class TransformerMovieDataset(Dataset):
    def __init__(self, texts, labels: np.ndarray, tokenizer, max_len: int):
        self.texts = list(texts)
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.float32)
        return item

