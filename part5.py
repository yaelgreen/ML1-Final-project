from part1 import create_training_and_validation_sets
from part3 import main_part_3
from part4 import main_part_4


def main_part_5():
    training_set, validation_set, meta = create_training_and_validation_sets(9999)
    main_part_3(training_set, validation_set, meta)
    main_part_4(training_set, validation_set, meta)

if __name__ == '__main__':
    main_part_5()
