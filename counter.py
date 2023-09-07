from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class NumericCounter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, msgs):
        msgs_series = pd.Series(msgs)  # Convert list to pandas Series
        return pd.DataFrame(msgs_series.apply(lambda s: sum(c.isdigit() for c in s)))