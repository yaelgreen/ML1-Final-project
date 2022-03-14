from part1 import create_training_and_validation_sets, convert_pixel_intensity
from part2 import Model


def main():
    training_set, validation_set, meta = create_training_and_validation_sets()
    convert_pixel_intensity(training_set)
    convert_pixel_intensity(validation_set)
    model = Model()
    learning_rate = 10
    num_of_iterations = 5
    batch_size = 50
    momentum_coefficient = 0.2
    l2_regularization_coefficient = 0.2
    standard_deviation = 0.9
    model.training(training_set, learning_rate, num_of_iterations, batch_size,
                   momentum_coefficient, l2_regularization_coefficient,
                   standard_deviation)


if __name__ == "__main__":
    main()