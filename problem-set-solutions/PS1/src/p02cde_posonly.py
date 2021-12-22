import matplotlib.pyplot as plt
import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # Part (c)
    x_train, y_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, label_col='t', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    m_train, m_valid, m_test = x_train.shape[0], x_valid.shape[0], x_test.shape[0]

    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # Plot the data and the decision boundary.
    # util.plot(x_test, y_test, clf.theta)
    # plt.show()

    y_train_pred = clf.predict(x_train)
    print("Using the true t-labels accuracy on train:", np.sum(y_train_pred == y_train) / m_train)
    y_valid_pred = clf.predict(x_valid)
    print("Using the true t-labels accuracy on valid:", np.sum(y_valid_pred == y_valid) / m_valid)
    y_test_pred = clf.predict(x_test)
    print("Using the true t-labels accuracy on test:", np.sum(y_test_pred == y_test) / m_test)
    pred_file_c = open(pred_path_c, 'w+')
    np.savetxt(pred_path_c, y_test_pred)

    # Part (d)
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    m_train, m_valid, m_test = x_train.shape[0], x_valid.shape[0], x_test.shape[0]

    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # Plot the data and the decision boundary.
    # util.plot(x_test, y_test, clf.theta)
    # plt.show()

    y_valid_pred = clf.predict(x_valid)
    print("Using the y-labels accuracy on valid:", np.sum(y_valid_pred == y_valid) / m_valid)
    y_test_pred = clf.predict(x_test)
    print("Using the y-labels accuracy on test:", np.sum(y_test_pred == y_test) / m_test)
    pred_file_d = open(pred_path_d, 'w+')
    np.savetxt(pred_path_d, y_test_pred)

    # Part (e)
    def h(x):
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))
        return sigmoid(x @ clf.theta)
    alpha = np.sum(h(x_valid[y_valid == 1])) / np.sum(y_valid == 1)
    clf.theta[0] += np.log(2 / alpha - 1)

    # Plot the data and the decision boundary.
    # util.plot(x_test, y_test, clf.theta)
    # plt.show()
    
    y_valid_pred = clf.predict(x_valid)
    print("Using the y-labels (rescaled) accuracy on valid:", np.sum(y_valid_pred == y_valid) / m_valid)
    y_test_pred = clf.predict(x_test)
    print("Using the y-labels (rescaled) accuracy on test:", np.sum(y_test_pred == y_test) / m_test)
    pred_file_e = open(pred_path_e, 'w+')
    np.savetxt(pred_path_e, y_test_pred)
