from financial_news.data.load_phrasebank import load_phrasebank
from financial_news.models.baselines import build_baseline_model
from financial_news.data.splitting import split_x_y
from financial_news.evaluation.metrics import calculate_classification_metrics
from financial_news.config import PROCESSED_DIR

def train_baseline_model():
    """
    Train a baseline model using train data and evaluate it on validation data.
    """
    train_df = load_phrasebank(PROCESSED_DIR / "sentiment_data_train.csv")
    val_df = load_phrasebank(PROCESSED_DIR / "sentiment_data_val.csv")
    X_train, y_train = split_x_y(train_df)
    X_val, y_val = split_x_y(val_df)

    model = build_baseline_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    metrics = calculate_classification_metrics(y_val, predictions)
    accuracy = metrics["accuracy"]
    macro_precision = metrics["macro_precision"]
    macro_f1_score = metrics["macro_f1_score"]

    print(f"Accuracy: {accuracy:.4f}, \nMacro Precision: {macro_precision:.4f}, \nMacro F1 Score: {macro_f1_score:.4f}")

    

if __name__ == "__main__":
    train_baseline_model()
    
