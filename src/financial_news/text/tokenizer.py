import re

TOKEN_PATTERN = re.compile(
    r"[a-z]+(?:[-'][a-z]+)*"
    r"|[+-]?\d+(?:[.,]\d+)*%?"
    r"|[$€£¥]"
    r"|[^\w\s]",
    flags=re.IGNORECASE,
)

def tokenize(text: str) -> list[str]:
    """
    Tokenizes the financial text into a list of tokens based on the defined TOKEN_PATTERN.
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    normalized_text = re.sub(r"\s+", " ", text.lower().strip())

    return TOKEN_PATTERN.findall(normalized_text)