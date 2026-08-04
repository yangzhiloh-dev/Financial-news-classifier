from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from financial_news.config import NUM_CLASSES
from financial_news.text.padding import pad_and_truncate
from financial_news.text.vocabulary import Vocabulary

class FinancialNewsDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[str], vocabulary: Vocabulary):
        
