from part1 import create_training_and_validation_sets, convert_pixel_intensity, download_and_extract_cifar_10_dataset
from part2 import Model


def main():
    # download_and_extract_cifar_10_dataset()
    training_set, validation_set, meta = create_training_and_validation_sets()
    convert_pixel_intensity(training_set)
    convert_pixel_intensity(validation_set)
    learning_rate = 0.01
    # num_of_iterations = [30, 40, 50]
    num_of_iterations = 3
    batch_size = 3
    momentum_coefficient = 0.3
    l2_regularization_coefficient = [0.4, 0.5, 0.6]
    # l2_regularization_coefficient = 0.5
    standard_deviation = 1
    for i in range(3):
        model_name = f"learning rate {learning_rate} number of iterations {num_of_iterations}"\
                     f" momentum coefficient {momentum_coefficient} "\
                     f"l2 regularization coefficient {l2_regularization_coefficient[i]}"
        print(model_name)
        model = Model(training_set, learning_rate, num_of_iterations, batch_size,
                      momentum_coefficient, l2_regularization_coefficient[i],
                      standard_deviation, model_name.replace(" ", "_").replace(".", "_"))
        model.training()
        model.plot_losses()
        model.inference_training_set()


if __name__ == "__main__":
    main()
