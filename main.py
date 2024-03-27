import numpy as np
import pandas as pd
import argparse


class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=10, n_features=None):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = np.zeros(n_features + 1)  # Including the bias weight

    def predict(self, X):
        summation = np.dot(X, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, X, y):
        for epoch in range(self.n_epochs):
            for inputs, label in zip(X, y):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
            accuracy = self.evaluate(X, y)
            print(f"Epoch {epoch + 1}/{self.n_epochs}, Accuracy: {accuracy * 100:.2f}%")

    def evaluate(self, X, y):
        predictions = np.array([self.predict(xi) for xi in X])
        accuracy = np.mean(predictions == y)
        return accuracy


def load_dataset(path):
    dataset = pd.read_csv(path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].apply(lambda x: 0 if x == dataset.iloc[:, -1].unique()[0] else 1).values
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a Perceptron model.")
    parser.add_argument('train_data', type=str, help='Path to the training data CSV file')
    parser.add_argument('test_data', type=str, help='Path to the test data CSV file')
    parser.add_argument('learning_rate', type=float, help='Learning rate for training')
    parser.add_argument('n_epochs', type=int, help='Number of epochs for training')

    args = parser.parse_args()

    # Load and prepare data
    X_train, y_train = load_dataset(args.train_data)
    X_test, y_test = load_dataset(args.test_data)

    # Initialize and train the Perceptron
    perceptron = Perceptron(learning_rate=args.learning_rate, n_epochs=args.n_epochs, n_features=X_train.shape[1])
    perceptron.train(X_train, y_train)

    # Evaluate on the test set
    test_accuracy = perceptron.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")

    # Interactive prediction
    while True:
        user_input = input("\nEnter new observation (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        try:
            observation = np.array([float(num) for num in user_input.split(',')], ndmin=2)
            prediction = perceptron.predict(observation[0])
            print("Predicted class:", prediction)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
