import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from financial_news.config import PROCESSED_DIR
from financial_news.text.vocabulary import PAD_ID, Vocabulary
from financial_news.config import NUM_CLASSES
from financial_news.models.embedding_mlp import EmbeddingAverageMLP
from financial_news.training.dataset import FinancialNewsDataset

def main() -> None:
    train_df = pd.read_csv( PROCESSED_DIR / "sentiment_data_train.csv" )

    vocabulary = Vocabulary.build(train_df["text"], min_freq=2, max_size=None)

    train_dataset = FinancialNewsDataset(
        texts=train_df["text"],
        labels=train_df["sentiment"],
        vocabulary=vocabulary,
        max_length=64,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    train_batch = next(iter(train_loader))

    EmbeddingAverageMLP(vocabulary_size=len(vocabulary), )

