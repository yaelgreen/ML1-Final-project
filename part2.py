from typing import Dict
import numpy as np

from part1 import random_indices, subset_of_data_set

training_set_size = 1000
num_of_labels = 10
training_set_shape = 3072  # 1024 * 3


class Model():
    wight_vectors = None
    biases = None

    def hinge_loss(self, y_hat, y):
        return np.maximum(0, 1 - y_hat * y)

    def dot_product(self, x_i):
        dot_product = np.dot(self.wight_vectors, x_i)
        print(f"dot_product: {dot_product}")
        return dot_product

    def gradient(self, training_set):
        m = len(training_set[b'labels'])
        loss = np.zeros(10)
        for idx in range(m):
            print(f"training_set[b'data'][idx].shape: {training_set[b'data'][idx].shape}")
            print(f"training_set[b'data'][idx]: {training_set[b'data'][idx]}")
            print(f"training_set[b'labels'][idx]: {training_set[b'labels'][idx]}")
            loss += self.hinge_loss(self.dot_product(training_set[b'data'][idx]),
                                           training_set[b'labels'][idx])
        return (1 / m) * loss

    def verify_training_input(self, learning_rate: float,
                              num_of_iterations: int, batch_size: int,
                              momentum_coefficient: float,
                              l2_regularization_coefficient: float,
                              standard_deviation: float):
        valid = True
        if batch_size < 1 or batch_size > training_set_size:
            print(f"Batch size {batch_size} must be an integer between 1 and"
                  f"training set size {training_set_size}")
            valid = False
        if learning_rate < 0:
            print(f"Learning rate {learning_rate} must be an positive float")
            valid = False
        if momentum_coefficient <= 0 or momentum_coefficient >= 1:
            print(f"Momentum Coefficient {momentum_coefficient} must be a"
                  f"float between 1 and 0")
            valid = False
        if l2_regularization_coefficient < 0:
            print(f"L2 regularization coefficient "
                  f"{l2_regularization_coefficient} must be a non negative"
                  f"float")
            valid = False
        if standard_deviation < 0:
            print(f"Standard Deviation {standard_deviation} must be a non"
                  f"negative float")
            valid = False
        if num_of_iterations <= 0:
            print(f"Number of iterations {num_of_iterations} must be a"
                  f"positive int")
            valid = False
        return valid

    def sgd(self, training_set, num_of_iterations, batch_size, learning_rate):
        # init wight_vectors
        self.wight_vectors = np.random.uniform(low=0.0, high=0.1, size=(1, training_set_shape))
        for t in range(num_of_iterations):
            # Random batch
            i_t = random_indices(batch_size, training_set_size)
            batch = subset_of_data_set(training_set, i_t)
            self.wight_vectors = self.wight_vectors - learning_rate * self.gradient(batch)

    def training(self, training_set: Dict,
                 learning_rate: float,
                 num_of_iterations: int,
                 batch_size: int,
                 momentum_coefficient: float,
                 l2_regularization_coefficient: float,
                 standard_deviation: float) -> None:
        valid = self.verify_training_input(learning_rate, num_of_iterations,
                                           batch_size, momentum_coefficient,
                                           l2_regularization_coefficient,
                                           standard_deviation)
        if not valid:
            return

        self.sgd(training_set, num_of_iterations, batch_size, learning_rate)


    def inference(self, set_of_instances: np.ndarray) -> np.ndarray:
        if not self.wight_vectors or not self.biases:
            print("You need to train the model before you can run inference")
        pass

# model = Model()
# model.training(self)