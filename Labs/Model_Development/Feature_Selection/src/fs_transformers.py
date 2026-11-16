"""Custom scikit-learn transformers used by Feature_Selection_Extended.ipynb.

They live in a module (not the notebook) so the pickled serving model in models/ can be
loaded from any script: pickle stores the class by import path.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


class CorrelationFilter(BaseEstimator, TransformerMixin):
    '''Unsupervised: drop the later feature of every pair with |Pearson r| above `threshold`.'''
    def __init__(self, threshold=0.9):
        self.threshold = threshold
    def fit(self, X, y=None):
        corr = np.abs(np.corrcoef(np.asarray(X, dtype=float), rowvar=False))
        upper = np.triu(corr, k=1)
        self.support_ = ~(upper > self.threshold).any(axis=0)
        return self
    def get_support(self):
        return self.support_
    def transform(self, X):
        return X.iloc[:, self.support_] if hasattr(X, "iloc") else np.asarray(X)[:, self.support_]


class PermutationImportanceSelector(BaseEstimator, TransformerMixin):
    '''Embedded selection for models without coef_/feature_importances_ (like HistGradientBoosting):
    fit on 75% of the training fold, permute each feature on the other 25%, keep features whose
    mean AUC drop is > `min_drop`.'''
    def __init__(self, estimator, min_drop=0.0005, n_repeats=5, random_state=42):
        self.estimator, self.min_drop, self.n_repeats, self.random_state = estimator, min_drop, n_repeats, random_state
    def fit(self, X, y):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, stratify=y, random_state=self.random_state)
        est = clone(self.estimator).fit(Xa, ya)
        imp = permutation_importance(est, Xb, yb, scoring="roc_auc", n_repeats=self.n_repeats,
                                     random_state=self.random_state, n_jobs=-1)
        self.importances_ = imp.importances_mean
        self.support_ = self.importances_ > self.min_drop
        return self
    def get_support(self):
        return self.support_
    def transform(self, X):
        return X.iloc[:, self.support_] if hasattr(X, "iloc") else np.asarray(X)[:, self.support_]


class ColumnSubset(BaseEstimator, TransformerMixin):
    '''Serving-time selector: just picks the named columns (no scores, no fitted selector model).'''
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.columns]
