from augmentation.transformation import Transformation
import random


class RandomErasing(Transformation):
    def __init__(self, instruction_count, **kwargs) -> None:
        super().__init__(**kwargs)
        self.instruction_count = instruction_count

    def get_augmentation_name(self):
        return "random_erasing"

    def get_augmentation_details(self):
        return {
            "instruction_count": self.instruction_count,
        }
        
    def augment_instructions(self, instructions):
        if self.instruction_count > len(instructions) + 2:
            return None

        low_pc = instructions[0].address
        instruction_count = min(self.instruction_count, len(instructions) - 1)

        last_possible_index = len(instructions) - 1 - instruction_count

        start_index = random.randint(0, last_possible_index)
        end_index = start_index + instruction_count + 1

        instructions[start_index:end_index] = []
        if instructions is None or len(instructions) < 2:
            return None

        augmented_instructions = self.recreate_instructions(instructions, low_pc)

        return augmented_instructions
