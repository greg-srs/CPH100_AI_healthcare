import numpy as np
import tqdm

class LogisticRegression():
    """
        A logistic regression model trained with stochastic gradient descent.
    """

    def __init__(self, num_epochs=100, learning_rate=1e-4, batch_size=16, regularization_lambda=0,  verbose=False):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.regularization_lambda = regularization_lambda

    def fit(self, X, Y):
        """
            Train the logistic regression model using stochastic gradient descent.
        """
        self.theta = np.zeros(X.shape[1])
        self.b = 0

        for epoch in range(self.num_epochs):
            order = np.random.permutation(len(Y))
            for i in range(0, len(Y), self.batch_size):
                batch_idx = order[i : i + self.batch_size]
                X_batch = X[batch_idx]
                Y_batch = Y[batch_idx]
                grad_theta, grad_b = self.gradient(X_batch,Y_batch)
                self.theta -= self.learning_rate * grad_theta
                self.b -= self.learning_rate * grad_b

        return self.theta, self.b

    def gradient(self, X, Y):
        """
            Compute the gradient of the loss with respect to theta and bias with L2 Regularization.
            Hint: Pay special attention to the numerical stability of your implementation.
        """
        P = self.predict_proba(X)
        for i in range(len(P)):
            p = P[i]
            if p < 0.001:
                p=0.001
            elif p>0.999:
                p=0.999
            P[i] = p
        grad_theta = (X.T @ (P - Y)) / len(Y) + self.regularization_lambda * self.theta
        grad_b = np.mean(P-Y)
        return grad_theta, grad_b
        
    def compute_loss(self, X, Y):
        """
            Binary cross entropy loss with L2 regularization, as defined in the pset writeup.
        """
        P = self.predict_proba(X)
        for i in range(len(P)):
            p = P[i]
            if p < 0.001:
                p=0.001
            elif p>0.999:
                p=0.999
            P[i] = p        
        bce = -np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P))
        l2 = (self.regularization_lambda / 2) * np.sum(self.theta ** 2)
        return bce + l2

    def predict_proba(self, X):
        """
            Predict the probability of lung cancer for each sample in X.
        """
        z = X @ self.theta + self.b
        p = 1/(1 + np.exp(-z))
        return p

    def predict(self, X, threshold=0.5):
        """
            Predict the if patient will develop lung cancer for each sample in X.
        """
        probas = self.predict_proba(X)
        prediction = (probas >= threshold).astype(int)
        return prediction
        