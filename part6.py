import os
from typing import Dict

from part1 import create_training_and_validation_sets, convert_pixel_intensity, \
    unpickle
from part2 import Model


def create_training_and_testing_sets() -> (Dict, Dict, Dict):
    # load dataset
    path = os.path.join(os.getcwd(), "cifar-10", "cifar-10-batches-py")
    batches = {}
    for i in range(1, 6):
        batch = unpickle(os.path.join(path, f"data_batch_{i}"))
        batches[i] = batch
    training_set = {
        b'batch_label': b'batches 1 to 5',
        b'labels': [label for i, batch in batches.items() for label in batch[b'labels']],
        b'data': [label for i, batch in batches.items() for label in batch[b'data']],
        b'filenames': [label for i, batch in batches.items() for label in batch[b'filenames']],
    }
    test_set = unpickle(os.path.join(path, "test_batch"))
    meta = unpickle(os.path.join(path, "batches.meta"))
    return training_set, test_set, meta

def main_part_6():
    training_set, test_set, meta = create_training_and_testing_sets()
    convert_pixel_intensity(training_set)
    convert_pixel_intensity(test_set)
    learning_rate = 0.01
    num_of_iterations = 50
    batch_size = 10
    momentum_coefficient = 0.3
    l2_regularization_coefficient = 0.4
    standard_deviation = 0.5
    model_name = f"learning rate {learning_rate} number of iterations {num_of_iterations}" \
                 f" batch_size {batch_size} momentum coefficient {momentum_coefficient} " \
                 f"l2 regularization coefficient {l2_regularization_coefficient}" \
                 f"standard_deviation {standard_deviation}"
    model = Model(training_set, learning_rate, num_of_iterations,
                  batch_size,
                  momentum_coefficient, l2_regularization_coefficient,
                  standard_deviation,
                  model_name.replace(" ", "_").replace(".", "_"))
    model.training()
    model.plot_losses()

if __name__ == '__main__':
    main_part_6()