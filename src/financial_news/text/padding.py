from financial_news.text.vocabulary import PAD_ID

def pad_and_truncate(token_ids: list[int], max_length: int, pad_id: int = PAD_ID) -> tuple[list[int], list[int]]:
    """
    Truncate or right-pad token ids and create attention mask
    """

    if not isinstance(token_ids, list):
        raise TypeError("token_ids must be a list.")

    if not all(isinstance(token_id, int) for token_id in token_ids):
        raise TypeError("Every token_id must be a int")

    if not isinstance(max_length, int):
        raise TypeError("Max length must be an integer")

    if max_length < 1:
        raise ValueError("Max length must be more than one")

    if not isinstance(pad_id, int):
        raise TypeError("Pad id must be an integer")

    if pad_id < 0:
        raise ValueError("Pad id cannot be negative")

    num_of_real_tokens = len(token_ids)
    truncated = token_ids[:max_length]

    if num_of_real_tokens < max_length:
        num_of_padded_tokens = max_length - num_of_real_tokens
        for i in range(num_of_padded_tokens):
            truncated.append(pad_id)
        attention_mask = [1]*num_of_real_tokens + [0]*num_of_padded_tokens
        return (truncated, attention_mask)

    attention_mask = [1]*len(truncated)

    return (truncated, attention_mask)