# Sentiments
negative → 0
neutral  → 1
positive → 2

# Market target
next_return = next_close / current_close - 1
next_direction = 1 if next_return > 0 else 0

# Hypothesis
H1: The PyTorch model performs better than a majority-class baseline.

H2: The PyTorch model performs comparably to or better than TF-IDF Logistic Regression.

H3: Daily sentiment scores show some relationship with the following trading session’s return.

H4: Transaction costs and neutral signals reduce apparent backtest performance.