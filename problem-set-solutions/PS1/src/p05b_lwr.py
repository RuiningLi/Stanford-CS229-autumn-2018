import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)

    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    m_valid = x_valid.shape[0]
    print("The MSE on valid of Locally Weighted Linear Regression:", np.linalg.norm(y_pred - y_valid) ** 2 / m_valid)

    # Plot the data
    # plt.scatter(x_train[:, 1], y_train, c='b', marker='X')
    # plt.scatter(x_valid[:, 1], y_pred, c='r', marker='o')
    # plt.show()


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        self.x = x
        self.y = y

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        y_pred = []
        for d in x:
            W = np.diag(np.exp(-np.linalg.norm(self.x - d, axis=1) ** 2 / 2 / self.tau / self.tau))
            theta = np.linalg.inv(self.x.T @ W @ self.x) @ self.x.T @ W @ self.y
            y_pred.append(np.dot(d, theta))
        return np.array(y_pred)
