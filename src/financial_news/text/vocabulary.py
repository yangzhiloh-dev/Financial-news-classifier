from collections import Counter
from collections.abc import Iterable

from financial_news.text.tokenizer import tokenize

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1


class Vocabulary:
    def __init__(self, token_to_id: dict[str, int], token_freq: dict[str, int]):
        self.token_to_id = token_to_id
        self.token_freq = token_freq
        self.id_to_token = {token_id: token
                            for token, token_id in token_to_id.items()
        }
        

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 2, max_size: int | None = None) -> "Vocabulary":
        """
        Builds a vocabulary from the given texts, filtering tokens based on minimum frequency and maximum size.
        """
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Iterable):
            raise TypeError("texts must be an iterable of strings, not one string.")

        if min_freq < 1:
            raise ValueError("Minimum frequency must be at least 1.")

        if max_size is not None and max_size < 2:
            raise ValueError("Max size must be at least 2 to accomodate PAD and UNK tokens.")

        token_counts: Counter[str] = Counter()
        for text in texts:
            tokens = tokenize(text)
            token_counts.update(tokens)

        # Filter tokens based on minimum frequency
        eligible_tokens = [(token, freq) for token, freq in token_counts.items()
                           if freq >= min_freq and token not in {PAD_TOKEN, UNK_TOKEN}]

        # Sort tokens by frequency (descending) and then alphabetically (ascending)
        eligible_tokens.sort(key=lambda x: (-x[1], x[0]))

        # Limit the vocabulary size if max_size is specified
        if max_size is not None:
            eligible_tokens = eligible_tokens[:max_size - 2]  # Reserve space for PAD and UNK

        # Create token to ID mapping
        token_to_id = {PAD_TOKEN: PAD_ID, 
                       UNK_TOKEN: UNK_ID
                       }
        for token, _ in eligible_tokens:
            token_to_id[token] = len(token_to_id)

        return cls(token_to_id=token_to_id, token_freq=dict(token_counts))

    def __len__(self) -> int:
        return len(self.token_to_id)

    def lookup_token(self, token: str) -> int:
        return self.token_to_id.get(token, UNK_ID)

    def encode_tokens(self, tokens: list[str]) -> list[int]:
        return [self.lookup_token(token) for token in tokens]

    def encode_text(self, text: str) -> list[int]:
        return self.encode_tokens(tokenize(text))

    def decode_ids(self, token_ids: list[int]) -> list[str]:
        return [self.id_to_token.get(token_id, UNK_TOKEN) for token_id in token_ids]
        