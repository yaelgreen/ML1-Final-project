from typing import Dict
import numpy as np
from part1 import random_indices, subset_of_data_set

training_set_size = 1000
num_of_labels = 10
training_set_shape = 3072  # 1024 * 3


class Model():

    def __init__(self, training_set: Dict,
                 learning_rate: float,
                 num_of_iterations: int,
                 batch_size: int,
                 momentum_coefficient: float,
                 l2_regularization_coefficient: float,
                 standard_deviation: float):
        valid = self.verify_training_input(learning_rate, num_of_iterations,
                                           batch_size, momentum_coefficient,
                                           l2_regularization_coefficient,
                                           standard_deviation)
        self.training_set = training_set
        self.learning_rate = learning_rate
        self.num_of_iterations = num_of_iterations
        self.batch_size = batch_size
        self.momentum_coefficient = momentum_coefficient
        self.l2_regularization_coefficient = l2_regularization_coefficient
        self.standard_deviation = standard_deviation
        if not valid:
            print("One or more of the parametres to the model is invalid")
        self.wight_vectors = None
        self.biases = None

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

    def sgd(self):
        # init wight_vectors
        self.wight_vectors = np.random.normal(loc=0, scale=self.standard_deviation,
                                              size=(num_of_labels + 1) * training_set_shape)
        for iteration_number in range(num_of_iterations):
            # Random batch
            training_indices = random_indices(batch_size, training_set_size)
            # batch = subset_of_data_set(training_set, i_t)
            for training_index in training_indices:
                x = self.training_set[b'data'][training_index]
                for i in range(num_of_labels):
                    theta_i, b_i = self.get_theta_b_i(i)
                    self.wight_vectors = self.wight_vectors - self.learning_rate * self.calc_derivative(theta_i, b_i, x)
            self.wight_vectors = self.wight_vectors - learning_rate * self.gradient(batch)

    def training(self) -> None:
        self.sgd()

    def inference(self, set_of_instances: np.ndarray) -> np.ndarray:
        if not self.wight_vectors or not self.biases:
            print("You need to train the model before you can run inference")
        pass

    def get_theta_b_i(self, i):
        theta_i = self.wight_vectors[i:i + training_set_shape]
        b_i = self.wight_vectors[i + training_set_shape]
        return theta_i, b_i

    def calc_exp(self, theta_i, b_i, x):
        return np.exp(np.inner(theta_i, x) + b_i)

    def calc_exp_sum(self, x):
        sum = 0
        for j in range(num_of_labels):
            theta_j, b_j = self.get_theta_b_i(j)
            sum += self.calc_exp(theta_j, b_j, x)

    def calc_derivative(self, theta_i, b_i, x):
        self.calc_exp(theta_i, b_i, x) / self.calc_exp_sum(x)


# model = Model()
# model.training(self)
