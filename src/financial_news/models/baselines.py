from sklearn.dummy import DummyClassifier

def build_baseline_model():
    """Builds a majority class baseline model using DummyClassifier."""

    return DummyClassifier(strategy='most_frequent')
    
