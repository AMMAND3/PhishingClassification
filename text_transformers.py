from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, msgs):
        if isinstance(msgs, list):
            return pd.DataFrame([len(msg) for msg in msgs])
        else:
            return pd.DataFrame(msgs.apply(len))
