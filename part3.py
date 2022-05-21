from textwrap import wrap

import numpy as np
import matplotlib.pyplot as plt

from part1 import create_training_and_validation_sets, convert_pixel_intensity, \
    plot_random_images
from part2 import Model


def instance_dimension_reduction(data_set, d):
    for index in range(data_set[b'data'].shape[0]):
        img = data_set[b'data'][index]
        img_r = img[0:1024].reshape(32, 32)
        img_g = img[1024:2048].reshape(32, 32)
        img_b = img[2048:].reshape(32, 32)
        img_r_new = np.zeros((32, 32))
        img_g_new = np.zeros((32, 32))
        img_b_new = np.zeros((32, 32))
        for i in range(33 - d):
            for j in range(33 - d):
                r_average = []
                g_average = []
                b_average = []
                for k in range(d):
                    r_average.append(img_r[i+k, j:j+d])
                    g_average.append(img_g[i + k, j:j + d])
                    b_average.append(img_b[i + k, j:j + d])
                r = np.average(r_average)
                g = np.average(g_average)
                b = np.average(b_average)
                img_r_new[i, j] = r
                img_g_new[i, j] = g
                img_b_new[i, j] = b
        new_img = np.concatenate((img_r_new.flatten(), img_g_new.flatten(),
                                  img_b_new.flatten()))
        data_set[b'data'][index] = new_img
    return data_set


def plot_hinge_loss_vs_model_capacity(losses, model_name):
    fig, axs = plt.subplots(2)
    plt_title = f'hing_loss_vs_model_capacity_for_{model_name}'
    fig.suptitle("\n".join(wrap(plt_title, 60)))
    axs[0].plot(list(losses['training'].values()))
    axs[1].plot(list(losses['validation'].values()))
    # plt.show()
    fig.savefig(f"{plt_title}.png")


if __name__ == "__main__":
    # For regularization
    # optimize over the training set
    training_set, validation_set, meta = create_training_and_validation_sets()
    convert_pixel_intensity(training_set)
    convert_pixel_intensity(validation_set)
    batch_size = 30
    learning_rate = 0.1
    num_of_iterations = 30
    momentum_coefficient = 0.3
    standard_deviation = 1
    l2_regularization = [0.05, 0.5, 1]
    losses = {'training': {}, 'validation': {}}
    for l in l2_regularization:
        model_name = f"l2_regularization_{l2_regularization}"
        model = Model(training_set, learning_rate, num_of_iterations,
                      batch_size, momentum_coefficient, l, standard_deviation,
                      model_name)
        model.training()
        # measure the classification error
        losses['training'][l] = model.hing_losses[num_of_iterations-1]
        losses['validation'][l] = model.calc_hinge_loss(validation_set[b'data'], validation_set[b'labels'])
        print(losses)
        # generate a plot showing training and validation errors vs. model capacity
        plot_hinge_loss_vs_model_capacity(losses, model_name)

    # For instance dimension reduction
    # optimize over the training set
    training_set, validation_set, meta = create_training_and_validation_sets()
    convert_pixel_intensity(training_set)
    convert_pixel_intensity(validation_set)
    l2_regularization = 0.5
    dimension_reduction = [2, 3, 4]
    losses = {'training': {}, 'validation': {}}
    for d in dimension_reduction:
        model_name = f"dimension_reduction_{d}"
        training_set_dim_d = instance_dimension_reduction(training_set, d)
        model = Model(training_set_dim_d, learning_rate, num_of_iterations,
                      batch_size, momentum_coefficient, l2_regularization,
                      standard_deviation,
                      model_name)
        model.training()
        # measure the classification error
        losses['training'][d] = model.hing_losses[num_of_iterations - 1]
        validation_set_dim_d = instance_dimension_reduction(validation_set, d)
        losses['validation'][d] = model.calc_hinge_loss(
            validation_set_dim_d[b'data'], validation_set_dim_d[b'labels'])
        print(losses)
        # generate a plot showing training and validation errors vs. model capacity
        plot_hinge_loss_vs_model_capacity(losses, model_name)

