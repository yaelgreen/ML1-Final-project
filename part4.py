import math
from part1 import create_training_and_validation_sets, plot_image, \
    convert_pixel_intensity
from part2 import Model
import part2
import matplotlib.pyplot as plt
import numpy as np

from part3 import plot_hinge_loss_vs_model_capacity

weight = height = 32
counter = 0

def to_greyscale(img):
    return np.average(img, axis=2)

def calc_gradient_images(img):
    global counter
    print(counter)
    counter += 1
    img = to_greyscale(img)
    new_img = []
    magnitudes = []
    angles = []
    for i in range(height):
        for j in range(weight):
            gx = img[i, min(height-1, j+1)] - img[i, max(0, j-1)]
            gy = img[max(0, i-1), j] - img[min(weight-1, i+1), j]
            magnitude = math.sqrt(pow(gx, 2) + pow(gy, 2))
            magnitudes.append(magnitude)
            angle = math.degrees(0.0) if gx == 0 else abs(math.atan(gy/gx))
            angles.append(angle)
    new_img.extend(magnitudes)
    new_img.extend(angles)
    return np.array(new_img)

def hogify(image_set):
    images = []
    for img in image_set:
        image_r = img[0:1024].reshape(32, 32)
        image_g = img[1024:2048].reshape(32, 32)
        image_b = img[2048:].reshape(32, 32)
        img = np.dstack((image_r, image_g, image_b))
        images.append(img)
    return [calc_gradient_images(img) for img in images]

def main_part_4(training_set, validation_set, meta):
    training_set[b'data'] = hogify(training_set[b'data'])
    validation_set[b'data'] = hogify(validation_set[b'data'])
    batch_size = 30
    learning_rate = 0.1
    num_of_iterations = 30
    momentum_coefficient = 0.3
    standard_deviation = 1
    l2_regularization = 0.5
    losses = {'training': {}, 'validation': {}}
    model_name = f"HOG"
    part2.training_set_shape = 32*32*2
    convert_pixel_intensity(training_set)
    convert_pixel_intensity(validation_set)
    model = Model(training_set, learning_rate, num_of_iterations,
                  batch_size, momentum_coefficient, l2_regularization,
                  standard_deviation, model_name)
    model.training()
    # measure the classification error
    losses['training'] = model.hing_losses[num_of_iterations-1]
    losses['validation'] = model.calc_hinge_loss(validation_set[b'data'], validation_set[b'labels'])
    print(losses)

if __name__ == '__main__':
    training_set, validation_set, meta = create_training_and_validation_sets()
    main_part_4(training_set, validation_set, meta)
