import random

from capstone import CS_OP_MEM, CS_OP_REG, CsInsn
from augmentation.transformation import Transformation

SWAPPABLE_MNEMONICS = [
    'mov', 'add', 'sub', 'xor', 'or', 'and', 'neg', 'pop',
    'inc', 'dec', 'not', 'lea', 'shl', 'sal', 'sar', 'shr',
    'movsx', 'movsd', 'movzx', 'movzd',
]


class InstructionSwapping(Transformation):
    def __init__(self, p_swap, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p_swap = p_swap

    def get_augmentation_name(self):
        return "instruction_swapping"

    def get_augmentation_details(self):
        return {
            "p_swap": self.p_swap,
        }

    def swappable_registers(self):
        return list(self.registers["64"].values()) \
            + list(self.registers["32"].values()) \
            + list(self.registers["16"].values()) \
            + list(self.registers["8"].values())

    def get_register_name(self, instruction, operand):
        if operand.type == CS_OP_REG:
            return instruction.reg_name(operand.reg)
        if operand.type == CS_OP_MEM:
            return instruction.reg_name(operand.mem.base)

        return None

    def get_dst_register(self, instruction: CsInsn):
        if len(instruction.operands) == 0:
            return None

        return self.get_register_name(instruction, instruction.operands[0])

    def get_src_register(self, instruction: CsInsn):
        if len(instruction.operands) != 2:
            return None

        return self.get_register_name(instruction, instruction.operands[1])

    def has_dependency(self, dst_register_1, src_register_1, dst_register_2, src_register_2):
        if dst_register_2 is None:
            return False

        def is_included(registers_1, registers_2):
            for register in registers_1:
                if register in registers_2:
                    return True
            return False

        dst_register_1_names = self.register_names(dst_register_1)
        src_register_1_names = self.register_names(src_register_1)
        dst_register_2_names = self.register_names(dst_register_2)
        src_register_2_names = self.register_names(src_register_2)

        if is_included(dst_register_1_names, src_register_2_names):
            return True
        if is_included(src_register_1_names, dst_register_2_names):
            return True
        if is_included(dst_register_1_names, dst_register_2_names):
            return True

        return False

    def can_be_swapped(self, instruction_1: CsInsn, instruction_2: CsInsn):
        if instruction_1.mnemonic not in SWAPPABLE_MNEMONICS or instruction_2.mnemonic not in SWAPPABLE_MNEMONICS:
            return False

        dst_register_1 = self.get_dst_register(instruction_1)
        dst_register_2 = self.get_dst_register(instruction_2)

        src_register_1 = self.get_src_register(instruction_1)
        src_register_2 = self.get_src_register(instruction_2)

        for register in [dst_register_1, dst_register_2, src_register_1, src_register_2]:
            if register is not None and register not in self.swappable_registers():
                return False

        has_dependency = self.has_dependency(
            dst_register_1=dst_register_1,
            src_register_1=src_register_1,
            dst_register_2=dst_register_2,
            src_register_2=src_register_2
        )

        return not has_dependency

    def augment_instructions(self, instructions):
        low_pc = instructions[0].address

        augmented_instructions = []

        instruction_count = len(instructions)
        if instruction_count == 1:
            augmented_instructions.append(instructions[0])

        i = 0
        while i < instruction_count - 1:
            if self.can_be_swapped(instructions[i], instructions[i + 1]):
                if self.rng.random() <= self.p_swap:
                    augmented_instructions.append(instructions[i + 1])
                    augmented_instructions.append(instructions[i])
                    i += 2
                    continue

            augmented_instructions.append(instructions[i])
            i += 1

            if i == instruction_count - 1:
                augmented_instructions.append(instructions[i])

        augmented_instructions = self.recreate_instructions(
            instructions=augmented_instructions,
            low_pc=low_pc
        )

        return augmented_instructions
