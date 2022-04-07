from part1 import create_training_and_validation_sets, convert_pixel_intensity, download_and_extract_cifar_10_dataset
from part2 import Model


def main():
    # download_and_extract_cifar_10_dataset()
    training_set, validation_set, meta = create_training_and_validation_sets()
    convert_pixel_intensity(training_set)
    convert_pixel_intensity(validation_set)
    learning_rate = 0.3
    num_of_iterations = 5
    batch_size = 50
    momentum_coefficient = 0.1
    l2_regularization_coefficient = 0.2
    standard_deviation = 1
    model = Model(training_set, learning_rate, num_of_iterations, batch_size,
                   momentum_coefficient, l2_regularization_coefficient,
                   standard_deviation)
    model.training()
    model.inference_training_set()


if __name__ == "__main__":
    main()