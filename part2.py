from textwrap import wrap
from typing import Dict
import numpy as np
from part1 import random_indices, create_training_and_validation_sets, \
    convert_pixel_intensity, plot_random_images
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("error")

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
                 standard_deviation: float,
                 model_name: str):
        valid = self.verify_training_input(learning_rate, num_of_iterations,
                                           batch_size, momentum_coefficient,
                                           l2_regularization_coefficient,
                                           standard_deviation)
        if not valid:
            print("One or more of the parameters to the model is invalid")
        self.training_set = training_set
        self.learning_rate = learning_rate
        self.num_of_iterations = num_of_iterations
        self.batch_size = batch_size
        self.momentum_coefficient = momentum_coefficient
        self.l2_regularization_coefficient = l2_regularization_coefficient
        self.standard_deviation = standard_deviation
        self.model_name = model_name
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
        self.cross_entropy_losses = {}
        self.hing_losses = {}
        for iteration_number in range(self.num_of_iterations):
            print(f'started {iteration_number} iteration')
            # Random batch
            training_indices = random_indices(self.batch_size, training_set_size)
            v_t = np.zeros(len(self.wight_vectors))
            for training_index in training_indices:
                x = self.training_set[b'data'][training_index]
                y = self.training_set[b'labels'][training_index]
                v_t = self.momentum_coefficient * v_t - (1 - self.momentum_coefficient) * self.learning_rate * self.calc_derivative(x, y)
                self.wight_vectors += v_t
            self.cross_entropy_losses[iteration_number] = self.calc_cross_entropy_loss(self.training_set[b'data'], self.training_set[b'labels'])
            print(f"iteration: {iteration_number} cross entropy losses: {self.cross_entropy_losses[iteration_number]}")
            self.hing_losses[iteration_number] = self.calc_hinge_loss(self.training_set[b'data'], self.training_set[b'labels'])
            print(
                f"iteration: {iteration_number} hing losses: {self.hing_losses[iteration_number]}")
        return self.wight_vectors

    def plot_losses(self):
        fig, axs = plt.subplots(2)
        plt_title = f'cross_entropy_loss_and_hing_loss_for_{self.model_name}'
        fig.suptitle("\n".join(wrap(plt_title, 60)))
        axs[0].plot(list(self.cross_entropy_losses.values()))
        axs[1].plot(list(self.hing_losses.values()))
        plt.show()
        # fig.savefig(f"{plt_title}.png")
        plt.close()

    def training(self) -> None:
        theta = self.sgd()
        with open('result.txt', 'w') as f:
            as_str = ' '.join([str(v) for v in self.wight_vectors])
            f.write(as_str)
        return theta

    def inference_training_set(self):
        # if not self.wight_vectors or not self.biases:
        #     print("You need to train the model before you can run inference")
        inference_result = self.inference(self.training_set[b'data'])
        successes = 0
        for i in range(0, training_set_size):
            if inference_result[i][1] == self.training_set[b'labels'][i]:
                successes += 1
        print(f"number of successes for model {self.model_name}: {successes}")
        return successes

    def single_inference(self, x):
        max_result = None
        prediction_label = 0
        class_score = []
        for j in range(num_of_labels):
            theta, b = self.get_theta_b_i(j)
            result = np.inner(theta, x) + b
            class_score.append(result)
            max_result = result if not max_result else max(result, max_result)
            prediction_label = j if result == max_result else prediction_label
        return class_score, prediction_label

    def inference(self, set_of_instances: np.ndarray) -> np.ndarray:
        result = {}
        for i in range(set_of_instances.shape[0]):
            class_score, prediction_label = self.single_inference(set_of_instances[i])
            result[i] = (class_score, prediction_label)
        return result

    def get_theta_b_i(self, i):
        theta_i = self.wight_vectors[i:i + training_set_shape]
        b_i = self.wight_vectors[i + training_set_shape]
        return theta_i, b_i

    def calc_hinge_loss(self, training_set_data, training_set_labels):
        loss = 0
        for i in range(0, training_set_size):
            x = training_set_data[i]
            y = training_set_labels[i]
            cur_loss = 0
            r, prediction_label = self.single_inference(x)
            for label in range(0, num_of_labels):
                cur_loss = max(cur_loss, r[label] - r[y] + 1 if label != y else 0)
            loss += cur_loss
        return loss

    def calc_cross_entropy_loss(self, training_set_data, training_set_labels):
        loss = 0
        for i in range(0, training_set_size):
            x = training_set_data[i]
            y = training_set_labels[i]
            r, _ = self.single_inference(x)
            sum = 0
            for j in range(num_of_labels):
                sum += np.exp(r[j])
            loss += np.log(sum) - r[y]
        return loss

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
                    derivative = x_j * (self.calc_exp(theta_i, b_i, x) / self.calc_exp_sum(x) - r_i) + 2 * self.l2_regularization_coefficient * theta_i[j]
                except RuntimeWarning:
                    print(f"{theta_i=}")
                    print(f"{x=}")
                    print(f"{b_i=}")
                    print(f"{r_i=}")
                    print(self.calc_exp(theta_i, b_i, x))
                    print(self.calc_exp_sum(x))
                vector.append(derivative)
            derivative_b = (self.calc_exp(theta_i, b_i, x) / self.calc_exp_sum(x) ) - r_i
            vector.append(derivative_b)
        return np.array(vector)


if __name__ == "__main__":
    training_set, validation_set, meta = create_training_and_validation_sets()
    convert_pixel_intensity(training_set)
    convert_pixel_intensity(validation_set)
    parameters = {1: {
        'learning_rate': 0.01,
        'batch_size': 10,
        'momentum_coefficient': 0.2,
        'l2_regularization_coefficient': 0.5,
        'standard_deviation': 0.5,
        'plot_images': True,
    },
        2: {
            'learning_rate': 0.01,
            'batch_size': 10,
            'momentum_coefficient': 0.3,
            'l2_regularization_coefficient': 0.4,
            'standard_deviation': 0.5,
            'plot_images': False,
    },
        3: {
            'learning_rate': 0.01,
            'batch_size': 10,
            'momentum_coefficient': 0.3,
            'l2_regularization_coefficient': 0.4,
            'standard_deviation': 1.5,
            'plot_images': False,
    },
        4: {
            'learning_rate': 0.01,
            'batch_size': 10,
            'momentum_coefficient': 0.3,
            'l2_regularization_coefficient': 0.6,
            'standard_deviation': 1.5,
            'plot_images': False,
    }
    }
    num_of_iterations = 10
    for i in parameters:
        model_name = f"learning rate {parameters[i]['learning_rate']} number of iterations {num_of_iterations}"\
                     f" batch_size {parameters[i]['batch_size']} momentum coefficient {parameters[i]['momentum_coefficient']} "\
                     f"l2 regularization coefficient {parameters[i]['l2_regularization_coefficient']}" \
                     f"standard_deviation {parameters[i]['standard_deviation']}"
        print(model_name)
        model = Model(training_set, parameters[i]['learning_rate'], num_of_iterations, parameters[i]['batch_size'],
                      parameters[i]['momentum_coefficient'], parameters[i]['l2_regularization_coefficient'],
                      parameters[i]['standard_deviation'], model_name.replace(" ", "_").replace(".", "_"))
        model.training()
        model.plot_losses()
        if parameters[i]['plot_images']:
            plot_random_images(training_set, meta, "training_set_images")
            plot_random_images(validation_set, meta, "validation_set_images")

