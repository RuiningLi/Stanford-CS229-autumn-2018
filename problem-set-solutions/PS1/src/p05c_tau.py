import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    m_valid = x_valid.shape[0]
    m_test = x_test.shape[0]

    best_tau, best_mse = None, None
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_valid)
        mse = np.linalg.norm(y_pred - y_valid) ** 2 / m_valid
        if best_mse is None or best_mse > mse:
            best_tau, best_mse = tau, mse
        
        # Plot the data
        plt.scatter(x_train[:, 1], y_train, c='b', marker='X')
        plt.scatter(x_valid[:, 1], y_pred, c='r', marker='o')
        plt.title("tau =" + str(tau))
        plt.show()
    
    model = LocallyWeightedLinearRegression(best_tau)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("The MSE on test of Locally Weighted Linear Regression of best tau:", np.linalg.norm(y_pred - y_test) ** 2 / m_test)
