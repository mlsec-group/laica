import numpy as np
from augmentation.transformation import Transformation
import random


class RandomNoise(Transformation):
    def __init__(self, seed, p_noise, **kwargs) -> None:
        super().__init__(seed=seed, **kwargs)
        self.p_noise = p_noise
        self.seed = seed
        random.seed(seed)
        self.rng = random.Random(seed)

    def get_augmentation_name(self):
        return "random_noise"

    def get_augmentation_details(self):
        return {
            "p_noise": self.p_noise,
        }

    def create_nop_instruction(self, low_pc):
        return self.disasemble_assembly_string("nop", low_pc=low_pc)[0]

    def augment_instructions(self, instructions):
        augmented_instructions = []
        for instruction in list(instructions):
            if self.rng.random() >= self.p_noise:
                augmented_instructions.append(instruction)
            else:
                augmented_instructions.append(instruction)
                augmented_instructions.append(self.create_nop_instruction(0))

        return self.recreate_instructions(augmented_instructions, instructions[0].address)
