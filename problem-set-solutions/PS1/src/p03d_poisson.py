import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    model = PoissonRegression(step_size=2e-7, max_iter=500)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    m_train = x_train.shape[0]
    m_valid = x_valid.shape[0]
    print("The RMSE on train of Poisson Regression:", np.sqrt(np.linalg.norm(y_train - model.predict(x_train)) ** 2 / m_train))
    print("The RMSE on valid of Poisson Regression:", np.sqrt(np.linalg.norm(y_valid - model.predict(x_valid)) ** 2 / m_valid))
    pred_file = open(pred_path, 'w+')
    np.savetxt(pred_path, y_pred)


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        m, n = x.shape
        self.theta = np.zeros(n)
        current_iter: int = 0
        while current_iter < self.max_iter:
            current_iter += 1
            delta = self.step_size * x.T @ (y - np.exp(x @ self.theta)) / m
            self.theta += delta
            if np.linalg.norm(delta, 1) < self.eps:
                break

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        return np.exp(x @ self.theta)
