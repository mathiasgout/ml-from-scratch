import numpy as np


class BaseModel:
    def _validate_data(self, X, y=None):
        X_validated = np.array(X)

        if y is None:
            return X_validated

        y_validated = np.array(y)

        if len(X) != len(y):
            raise ValueError(f"X and y shapes does not match ({X.shape} and {y.shape})")

        return X_validated, y_validated
