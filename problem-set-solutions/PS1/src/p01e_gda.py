import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    clf = GDA()
    clf.fit(x_train, y_train)

    # Plot the data and the decision boundary.
    # n = x_train.shape[1]
    # theta = np.zeros(n + 1)
    # theta[1:] = np.linalg.inv(clf.cov) @ (clf.mu1 - clf.mu0)
    # theta[0] = (clf.mu0.T @ np.linalg.inv(clf.cov) @ clf.mu0 - clf.mu1.T @ np.linalg.inv(clf.cov) @ clf.mu1) / 2 - np.log((1 - clf.phi) / clf.phi)
    # util.plot(x_train, y_train, theta)
    # plt.show()

    y_pred = clf.predict(x_valid)
    m_train = x_train.shape[0]
    m_valid = x_valid.shape[0]
    print("The accuracy on train of GDA:", np.sum(clf.predict(x_train) == y_train) / m_train)
    print("The accuracy on valid of GDA:", np.sum(clf.predict(x_valid) == y_valid) / m_valid)
    pred_file = open(pred_path, 'w+')
    np.savetxt(pred_path, y_pred)


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        m, n = x.shape
        sum_y = np.sum(y)
        self.phi = sum_y / m
        self.mu0 = np.sum(x[y == 0], axis=0) / (m - sum_y)
        self.mu1 = np.sum(x[y == 1], axis=0) / sum_y
        x0 = x[y == 0] - self.mu0
        x1 = x[y == 1] - self.mu1
        self.cov = (x0.T @ x0 + x1.T @ x1) / m


    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        x0 = x - self.mu0
        x1 = x - self.mu1
        inv_cov = np.linalg.inv(self.cov)
        posterior_0 = np.exp(-np.sum((x0 @ inv_cov) * x0, axis=1) / 2) * (1 - self.phi)
        posterior_1 = np.exp(-np.sum((x1 @ inv_cov) * x1, axis=1) / 2) * self.phi
        return posterior_0 < posterior_1
