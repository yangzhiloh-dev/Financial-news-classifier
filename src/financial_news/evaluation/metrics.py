from sklearn.metrics import accuracy_score, precision_score, f1_score

def calculate_classification_metrics(y_true, y_pred):
    """
    Calculate classification metrics for model evaluation.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must not be empty.")

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1_score = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return {"accuracy": float(accuracy), 
            "macro_precision": float(macro_precision), 
            "macro_f1_score": float(macro_f1_score)
            }

