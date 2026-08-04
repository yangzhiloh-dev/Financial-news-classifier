from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from financial_news.config import LABEL_TO_ID
from financial_news.text.padding import pad_and_truncate
from financial_news.text.vocabulary import Vocabulary

class FinancialNewsDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[str], vocabulary: Vocabulary, max_length: int =  64):
        """
        Converts text and label into tensors pytorch can train on
        """

        if not len(texts) == len(labels):
            raise ValueError("Length of texts and labels must be the same.")

        if not all(isinstance(text, str) for text in texts):
            raise TypeError("Every texts must be a string.")

        if not isinstance(vocabulary, Vocabulary):
            raise TypeError("vocabulary must be Vocabulary object")

        if not isinstance(max_length, int) or max_length < 1:
            raise ValueError("max_length must be a positive integer.")

        unknown_labels = set(labels) - set(LABEL_TO_ID)
        if unknown_labels:
            raise ValueError(f"Unknown labels found: {unknown_labels}")

        self.texts = list(texts)
        self.labels = list(labels)
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        text = self.texts[index]
        label = self.labels[index]

        token_ids = self.vocabulary.encode_text(text)

        truncated, attention_mask = pad_and_truncate(token_ids, self.max_length)
        label_id = LABEL_TO_ID[label]

        return {
            "input_ids": torch.tensor(truncated, dtype = torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype = torch.long),
            "label": torch.tensor(label_id, dtype = torch.long)
        }

            

        

