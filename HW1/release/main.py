import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term for intercept
        beta_hat = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        self.intercept = beta_hat[0]
        self.weights = beta_hat[1:]

    def predict(self, X):
        return X @ self.weights + self.intercept


class LinearRegressionGradientdescent(LinearRegressionBase):
    def normalize_data(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return (X - mean) / std, mean, std

    def fit(self, X, y, learning_rate=0.01, epochs=50):
        X, mean, std = self.normalize_data(X)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        theta = np.random.randn(X_b.shape[1]) * 0.01  # Small random initialization
        m = len(y)
        y = y.flatten()  # Ensure y is 1D
        losses = []

        for epoch in range(epochs):
            predictions = X_b @ theta
            gradient = (1 / m) * (X_b.T @ (predictions - y))  # Gradient computation
            theta -= learning_rate * gradient  # Update theta

            # Calculate Mean Squared Error (MSE) for the current epoch
            loss = (1 / (2 * m)) * np.sum(np.square(predictions - y))
            losses.append(loss)

            if epoch % 10000 == 0:
                logger.info(f'Epoch {epoch}, Loss: {loss:.4f}')

        # Set weights and intercept after training
        self.intercept = theta[0] - np.sum(theta[1:] * mean / std)
        self.weights = theta[1:] / std
        return losses

    def predict(self, X):
        return X @ self.weights + self.intercept

    def plot_learning_curve(self, losses):
        plt.plot(losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend(['Training Loss'])
        plt.show()


def compute_mse(prediction, ground_truth):
    return np.mean(np.power(prediction - ground_truth, 2))


def main():
    # Load training data
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()

    # Closed-form solution
    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'Closed-form weights: {LR_CF.weights}, intercept: {LR_CF.intercept:.4f}')

    # Gradient Descent solution
    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=0.1, epochs=100)
    LR_GD.plot_learning_curve(losses)
    logger.info(f'Gradient Descent weights: {LR_GD.weights}, intercept: {LR_GD.intercept:.4f}')

    # Load testing data
    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    # Predictions
    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)

    # MSE comparison
    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100

    # Log MSE and difference
    logger.info(f'MSE (Closed-form): {mse_cf:.4f}, MSE (GD): {mse_gd:.4f}, Difference: {diff:.2f}%')


if __name__ == '__main__':
    main()
