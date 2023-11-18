from ml_from_scratch.base import BaseModel

import numpy as np


class LinearDiscriminantAnalysis(BaseModel):
    def __init__(self, priors=None) -> None:
        self.priors = priors

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = self._validate_data(X, y)

        n_samples, n_features = X.shape
        if n_samples <= n_features + 1:
            raise ValueError(
                f"The number of samples ({n_samples}) is too low compared to the number of features ({n_features})"
            )

        # Classes
        self.classes_, _counts = np.unique(y, return_counts=True)

        # Priors
        if self.priors is None:
            self.priors_ = np.array([cnt / n_samples for cnt in _counts])
        else:
            self.priors_ = self.priors

        # Means and pooled covariance matrices
        _means = []
        _pooled_covs = []
        for k in self.classes_:
            Xk = X[y == k, :]
            _means.append(Xk.mean(axis=0))
            _pooled_covs.append(
                np.cov(Xk, rowvar=False, bias=True) * Xk.shape[0]
            )  # mutiply by Xk sample size (Xk.shape[0]) for pooled cov calculation

        self.means_ = np.array(_means)

        # Inverse covariance matrix (pooled)
        self.inverse_covariance_ = np.linalg.inv(
            sum(_pooled_covs) / (n_samples - len(self.classes_))
        )
        self._fitted = True

    def predict(self, X: np.ndarray):
        return np.apply_along_axis(
            lambda x: self.classes_[np.argmax(x, axis=0)],
            axis=1,
            arr=self.predict_proba(X),
        )

    def predict_proba(self, X: np.ndarray):
        X, _ = self._validate_data(X)
        return np.apply_along_axis(self._predict_proba_sample, axis=1, arr=X)

    def _predict_proba_sample(self, x: np.ndarray):
        # Denominator
        denominator = 0
        for k in range(len(self.classes_)):
            md = self._compute_mahalanobis_distance(
                x, self.means_[k], self.inverse_covariance_
            )
            denominator += self.priors_[k] * np.exp(-md / 2)

        # Proba
        probas = []
        for k in range(len(self.classes_)):
            md = self._compute_mahalanobis_distance(
                x, self.means_[k], self.inverse_covariance_
            )
            probas.append((self.priors_[k] * np.exp(-md / 2)) / denominator)

        return np.array(probas)

    @staticmethod
    def _compute_mahalanobis_distance(x, mean, inverse_covariance):
        return (x - mean).dot(inverse_covariance).dot(x - mean)
