import math
from pprint import _safe_key
from typing import Dict
import numpy as np
from part1 import random_indices, subset_of_data_set
import warnings
warnings.filterwarnings("error")
from sklearn.linear_model import LinearRegression

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
                                              size=num_of_labels * (training_set_shape + 1))
        for iteration_number in range(self.num_of_iterations):
            print(f'started {iteration_number} iteration')
            # Random batch
            training_indices = random_indices(self.batch_size, training_set_size)
            v_t = np.zeros(len(self.wight_vectors))
            # batch = subset_of_data_set(training_set, i_t)
            for training_index in training_indices:
                x = self.training_set[b'data'][training_index]
                y = self.training_set[b'labels'][training_index]
                v_t = self.momentum_coefficient * v_t - (1 - self.momentum_coefficient) * self.learning_rate * self.calc_derivative(x, y)
                self.wight_vectors += v_t
        return self.wight_vectors

    def training(self) -> None:
        theta = self.sgd()
        with open('result.txt', 'w') as f:
            as_str = ' '.join([str(v) for v in self.wight_vectors])
            f.write(as_str)
        return theta

    def inference(self, set_of_instances: np.ndarray) -> np.ndarray:
        # if not self.wight_vectors or not self.biases:
        #     print("You need to train the model before you can run inference")
        print('Inference')
        successes = 0
        for i in range(9999):
            print(f'image {i}')
            x = self.training_set[b'data'][i]
            y = self.training_set[b'labels'][i]
            max_result = None
            index = 0
            for j in range(num_of_labels):
                theta, b = self.get_theta_b_i(j)
                result = np.inner(theta, x) + b
                max_result = result if not max_result else max(result, max_result)
                index = j if result == max_result else index
            if y == index:
               successes += 1
        print(successes)

    def get_theta_b_i(self, i):
        theta_i = self.wight_vectors[i:i + training_set_shape]
        b_i = self.wight_vectors[i + training_set_shape]
        return theta_i, b_i

    def calc_exp(self, theta_i, b_i, x):
        try:
            return np.exp(np.inner(theta_i, x) + b_i)
        except RuntimeWarning:
            print(f"{theta_i=}")
            print(f"{x=}")
            print(f"{b_i=}")

    def calc_exp_sum(self, x):
        sum = 0
        for j in range(num_of_labels):
            theta_j, b_j = self.get_theta_b_i(j)
            sum += self.calc_exp(theta_j, b_j, x)
        return sum

    def calc_derivative(self, x, y):
        vector = []
        r = [0] * num_of_labels
        r[y] = 1

        for i in range(num_of_labels):
            theta_i, b_i = self.get_theta_b_i(i)
            r_i = r[i]
            for j in range(training_set_shape):
                x_j = x[j]
                try:
                    derivative = x_j * self.calc_exp(theta_i, b_i, x) / (self.calc_exp_sum(x) - r_i) + 2 * self.l2_regularization_coefficient * theta_i[j]
                except RuntimeWarning:
                    print(f"{theta_i=}")
                    print(f"{x=}")
                    print(f"{b_i=}")
                    print(f"{r_i=}")
                    print(self.calc_exp(theta_i, b_i, x))
                    print(self.calc_exp_sum(x))
                vector.append(derivative)
            derivative_b = self.calc_exp(theta_i, b_i, x) / (self.calc_exp_sum(x) - r_i)
            vector.append(derivative_b)
        return np.array(vector)


# model = Model()
# model.training(self)
