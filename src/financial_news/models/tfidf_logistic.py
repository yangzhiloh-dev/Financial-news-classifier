from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def build_tfidf_logistic_model() -> Pipeline:
    """
    Build a TF-IDF + Logistic Regression model pipeline for text classification.
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    logistic_regression = LogisticRegression(max_iter=1000, random_state=42)

    pipeline = Pipeline([('tfidf', tfidf_vectorizer), ('logistic', logistic_regression)])

    return pipeline