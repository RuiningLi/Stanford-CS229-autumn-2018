import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # Plot the data and the decision boundary.
    # util.plot(x_train, y_train, clf.theta)
    # plt.show()

    y_pred = clf.predict(x_valid)
    m_train = x_train.shape[0]
    m_valid = x_valid.shape[0]
    print("The accuracy on train of Logistic Regression:", np.sum(clf.predict(x_train) == y_train) / m_train)
    print("The accuracy on valid of Logistic Regression:", np.sum(clf.predict(x_valid) == y_valid) / m_valid)
    pred_file = open(pred_path, 'w+')
    np.savetxt(pred_path, y_pred)


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        m, n = x.shape

        def h(x):
            def sigmoid(z):
                return 1 / (1 + np.exp(-z))
            return sigmoid(x @ self.theta)

        def gradient(x, y):
            return -x.T @ (y - h(x)) / m

        def hessian(x, y):
            h_x = h(x)
            return np.matmul((h_x * (1 - h_x)) * x.T, x) / m
        
        # Initialize the parameters.
        self.theta = np.zeros(n)
        # Update the parameters using Newton's method
        should_terminate: bool = False
        current_iter: int = 0
        while not should_terminate:
            theta_0 = self.theta
            self.theta = theta_0 - np.linalg.inv(hessian(x, y)) @ gradient(x, y)
            if np.linalg.norm(theta_0 - self.theta, 1) < self.eps:
                should_terminate = True
            current_iter += 1
            if current_iter == self.max_iter:
                break


    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        return x @ self.theta >= 0
    