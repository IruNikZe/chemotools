import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class NonNegative(BaseEstimator, TransformerMixin):
    def __init__(self, mode: str = "zero"):
        self.mode = mode

    def fit(self, X: np.ndarray, y=None) -> "NonNegative":
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        # Check that the estimator is fitted
        check_is_fitted(self, "_is_fitted")

        # Check that X is a 2D array and has only finite values
        X = check_input(X)
        X_ = X.copy()

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features but got {X_.shape[1]}")

        # Calculate non-negative values
        for i, x in enumerate(X_):
            if self.mode == "zero":
                X_[i] = np.clip(x, a_min=0, a_max=np.inf)
            
            if self.mode == "abs":
                X_[i] = np.abs(x)

        return X_.reshape(-1, 1) if X_.ndim == 1 else X_