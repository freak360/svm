import numpy as np

# Defining the SVM Class for Implementing Algorithm
class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * np.dot(x_i, self.w) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)

                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt

    # Getting the dataset ready
    X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40)
    y = np.where(y==0, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Fitting the Algorithm and making predictions
    clf = SVM()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true==y_pred) / len(y_true)
        return accuracy
    
    print("SVM Classification Accuracy:", accuracy(y_test, predictions))


    # Defining the Visualization method to see the plots of our Algorithm
    def visualize_svm():
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]
    
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--", label='Decision Boundary')
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k", label='Negative Margin')
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k", label='Positive Margin')

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim(x1_min - 3, x1_max + 3)
        plt.legend()
        plt.show()

# Finally calling the Visualize method to see plots
visualize_svm()
