import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.utils.validation import check_is_fitted

from chemotools.utils.check_inputs import check_input


class IndexScaler(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """
    A transformer that scales the input data by the value at a given index.

    Parameters
    ----------
    index : int, optional
        The index to scale the data by.

    Attributes
    ----------
    n_features_in_ : int
        The number of features in the input data.

    _is_fitted : bool
        Whether the transformer has been fitted to data.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer to the input data.

    transform(X, y=0, copy=True)
        Transform the input data by scaling by the value at a given index.
    """
    def __init__(self, index: int = 0):
        self.index = index


    def fit(self, X: np.ndarray, y=None) -> "IndexScaler":
        """
        Fit the transformer to the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data to fit the transformer to.

        y : None
            Ignored.

        Returns
        -------
        self : IndexScaler
            The fitted transformer.
        """
        # Check that X is a 2D array and has only finite values
        X = check_input(X)

        # Set the number of features
        self.n_features_in_ = X.shape[1]

        # Set the fitted attribute to True
        self._is_fitted = True

        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Transform the input data by scaling by the value at a given index.

        Parameters
        ----------
        X : np.ndarray
            The input data to transform.

        y : None
            Ignored.

        Returns
        -------
        X_ : np.ndarray
            The transformed data.
        """
        # Check that the estimator is fitted
        check_is_fitted(self, "_is_fitted")

        # Check that X is a 2D array and has only finite values
        X = check_input(X)
        X_ = X.copy()

        # Check that the number of features is the same as the fitted data
        if X_.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features but got {X_.shape[1]}")

        # Scale the data by index
        for i, x in enumerate(X_):
            X_[i] = x / x[self.index]
        
        return X_.reshape(-1, 1) if X_.ndim == 1 else X_