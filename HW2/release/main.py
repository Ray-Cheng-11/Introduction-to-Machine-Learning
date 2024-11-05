import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[np.float64],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        num_samples, num_features = inputs.shape
        self.weights = np.zeros(num_features)
        self.intercept = 0

        for i in range(self.num_iterations):
            linear_model = np.dot(inputs, self.weights) + self.intercept
            predictions = self.sigmoid(linear_model)
            error = predictions - targets
            self.weights -= (self.learning_rate / num_samples) * np.dot(inputs.T, error)
            self.intercept -= (self.learning_rate / num_samples) * np.sum(error)

            if i % 100 == 0:
                loss = self.cross_entropy_loss(predictions, targets)
                logger.info(f'Iteration {i}: Loss={loss:.4f}')

    def cross_entropy_loss(self, predictions, targets):
        m = len(targets)
        loss = -(1 / m) * np.sum(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        return loss

    def predict(
        self,
        inputs: npt.NDArray[np.float64],
    ) -> t.Tuple[t.Sequence[np.float64], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        linear_model = np.dot(inputs, self.weights) + self.intercept
        probabilities = self.sigmoid(linear_model)
        predictions = np.where(probabilities >= 0.5, 1, 0)
        return probabilities, predictions

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))


class FLD:
    """Implement FLD
    You can add arguments as you need,
    but don't modify those already exist variables.
    """
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[np.float64],
        targets: t.Sequence[int],
    ) -> None:
        X0 = inputs[targets == 0]
        X1 = inputs[targets == 1]

        self.m0 = np.mean(X0, axis=0)
        self.m1 = np.mean(X1, axis=0)
        self.sw = np.dot((X0 - self.m0).T, (X0 - self.m0)) + np.dot((X1 - self.m1).T, (X1 - self.m1))
        self.sb = np.dot((self.m0 - self.m1).reshape(-1, 1), (self.m0 - self.m1).reshape(1, -1))
        self.w = np.linalg.inv(self.sw) @ (self.m1 - self.m0)
        self.slope = self.w[1] / self.w[0] if self.w[0] != 0 else np.inf

    def predict(
        self,
        inputs: npt.NDArray[np.float64],
    ) -> t.Sequence[t.Union[int, bool]]:
        projected = np.dot(inputs, self.w)
        projected_m0 = np.dot(self.m0, self.w)
        projected_m1 = np.dot(self.m1, self.w)

        return np.where(np.abs(projected - projected_m0) < np.abs(projected - projected_m1), 0, 1)

    def plot_projection(self, inputs: npt.NDArray[np.float64], title_suffix: str = '') -> None:
        plt.figure(figsize=(10, 8))
        predictions = self.predict(inputs)
        w_norm = self.w / np.linalg.norm(self.w)
        # Calculate the center point of the data
        center = np.mean(inputs, axis=0)
        # Calculate points for the projection line
        line_length = np.max(np.linalg.norm(inputs - center, axis=1)) * 2
        line_direction = np.array([w_norm[0], w_norm[1]])
        line_points = np.vstack([
            center + line_length * line_direction,
            center - line_length * line_direction
        ])
        slope = w_norm[1] / w_norm[0] if w_norm[0] != 0 else np.inf
        if slope != np.inf:
            intercept = line_points[0, 1] - slope * line_points[0, 0]
        else:
            intercept = line_points[0, 0]
        # Plot the projection line
        plt.plot(line_points[:, 0], line_points[:, 1], 'k--', label='Projection Line')

        # Plot data points colored by predictions
        class0_mask = predictions == 0
        class1_mask = predictions == 1

        plt.scatter(inputs[class0_mask][:, 0],
                    inputs[class0_mask][:, 1],
                    c='blue', label='Predicted Class 0')
        plt.scatter(inputs[class1_mask][:, 0],
                    inputs[class1_mask][:, 1],
                    c='red', label='Predicted Class 1')

        # Calculate and plot projections for each point
        for i in range(len(inputs)):
            point = inputs[i]
            # Calculate projection point using vector projection formula
            proj_scalar = np.dot(point - center, w_norm)
            proj_point = center + proj_scalar * w_norm

            # Set color based on prediction
            point_color = 'blue' if predictions[i] == 0 else 'red'

            # Plot line connecting point to its projection
            plt.plot([point[0], proj_point[0]],
                     [point[1], proj_point[1]],
                     color='green', alpha=0.2)

            plt.plot(proj_point[0], proj_point[1], 'o',
                     color=point_color, alpha=0.5, markersize=4)

        if slope == np.inf:
            equation_text = f"b = {intercept:.4f}"
        else:
            equation_text = f"w = {slope:.4f}, b= {intercept:.4f}"

        if hasattr(self, 'm0') and hasattr(self, 'm1'):
            plt.scatter(self.m0[0], self.m0[1], c='cyan', s=100, marker='*', label='Mean Class 0')
            plt.scatter(self.m1[0], self.m1[1], c='magenta', s=100, marker='*', label='Mean Class 1')

        plt.title(f'FLD Projection ({equation_text}) {title_suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


def compute_auc(y_trues, y_preds):
    return roc_auc_score(y_trues, y_preds)


def accuracy_score(y_trues, y_preds):
    return np.mean(np.array(y_trues) == np.array(y_preds))


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print(y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=0.03,  # You can modify the parameters as you want
        num_iterations=1000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['10', '20']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    """
    (TODO): Implement your code to
    1) Fit the FLD model
    2) Make prediction
    3) Compute the evaluation metrics

    Please also take care of the variables you used.
    """

    FLD_.fit(x_train, y_train)
    fld_preds = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, fld_preds)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    """
    (TODO): Implement your code below to plot the projection
    """
    FLD_.plot_projection(x_train, " (Train Set)")
    FLD_.plot_projection(x_test, " (Test Set)")


if __name__ == '__main__':
    main()
