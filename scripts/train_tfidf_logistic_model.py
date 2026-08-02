from financial_news.data.load_phrasebank import load_phrasebank
from financial_news.data.splitting import split_x_y
from financial_news.evaluation.metrics import calculate_classification_metrics
from financial_news.config import PROCESSED_DIR
from financial_news.config import ID_TO_LABEL
from financial_news.models.tfidf_logistic import build_tfidf_logistic_model

def train_tfidf_logistic_model():
    """
    Train a TF-IDF + Logistic Regression model using train data and evaluate it on validation data.
    """
    train_df = load_phrasebank(PROCESSED_DIR / "sentiment_data_train.csv")
    val_df = load_phrasebank(PROCESSED_DIR / "sentiment_data_val.csv")
    X_train, y_train = split_x_y(train_df)
    X_val, y_val = split_x_y(val_df)

    model = build_tfidf_logistic_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    metrics = calculate_classification_metrics(y_val, predictions)
    accuracy = metrics["accuracy"]
    macro_precision = metrics["macro_precision"]
    macro_f1_score = metrics["macro_f1_score"]

    print(f"Accuracy: {accuracy:.4f}, \nMacro Precision: {macro_precision:.4f}, \nMacro F1 Score: {macro_f1_score:.4f}")

    tfidf = model.named_steps['tfidf']
    logistic = model.named_steps['logistic']
    feature_names = tfidf.get_feature_names_out()
    class_ids = logistic.classes_

    print(f"Number of vocabulary terms: {feature_names.size}")
    for i, class_id in enumerate(class_ids):
        class_label = ID_TO_LABEL[class_id]
        top_features = sorted(zip(logistic.coef_[i], feature_names), reverse=True)[:5]
        print(f"\nTop features for class '{class_label}':")
        for coef, feature in top_features:
            print(f"{feature}: {coef:.4f}")

if __name__ == "__main__":
    train_tfidf_logistic_model()