import math
from augmentation.transformation import Transformation
import random


class Cropping(Transformation):
    def __init__(self, instruction_count, seed, **kwargs) -> None:
        super().__init__(seed=seed, **kwargs)
        self.instruction_count = instruction_count
        self.seed = seed
        random.seed(seed)
        self.rng = random.Random(seed)

    def get_augmentation_name(self):
        return "cropping"

    def get_augmentation_details(self):
        return {
            "instruction_count": self.instruction_count,
        }

    def augment_instructions(self, instructions):
        if self.instruction_count > (len(instructions) / 2) + 2:
            return None

        instruction_count = min(
            self.instruction_count,
            (math.ceil(len(instructions) / 2)) - 1
        )

        low_pc = instructions[0].address
        augmented_instructions = list(instructions)
        augmented_instructions[0:instruction_count] = []
        augmented_instructions[-instruction_count:] = []

        if len(augmented_instructions) < 1:
            return None
        
        augmented_instructions = self.recreate_instructions(augmented_instructions, low_pc)

        return augmented_instructions
