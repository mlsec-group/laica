from capstone import CsInsn
from augmentation.transformation import Transformation
from capstone import CS_OP_REG, CS_OP_IMM, CS_OP_MEM, CS_OP_FP
from keystone import Ks, KS_ARCH_X86, KS_MODE_64
ks = Ks(KS_ARCH_X86, KS_MODE_64)
ks.detail = True

MNEMONICS_WITH_KNOWN_SUBSTITUTIONS = [
    'mov', 'add', 'sub', 'xor', 'or', 'and', 'neg', 'pop'
]


class InstructionSubstitution(Transformation):
    def __init__(self, p_substitute, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p_substitute = p_substitute

    def get_augmentation_name(self):
        return "instruction_substitution"

    def get_augmentation_details(self):
        return {
            "p_substitute": self.p_substitute,
        }

    def negation_substitution(self, instruction: CsInsn):
        return self.rng.choice([
            [
                f"not {instruction.op_str}",
                f"add {instruction.op_str}, 1",
            ],
            [
                f"not {instruction.op_str}",
                f"add {instruction.op_str}, 0x1",
            ],
        ])

    def add_two_registers_substitution(self, instruction: CsInsn):
        first_register = instruction.reg_name(instruction.operands[0].reg)
        second_register = instruction.reg_name(instruction.operands[1].reg)
        random_int = self.rng.randint(0, 1000)
        return self.rng.choice([
            [
                f"neg {second_register}",
                f"sub {first_register}, {second_register}",
                f"neg {second_register}",
            ],
            [
                f"neg {second_register}",
                f"neg {first_register}",
                f"add {first_register}, {second_register}",
                f"neg {second_register}",
                f"neg {first_register}",
            ],
            [
                f"neg {first_register}",
                f"neg {second_register}",
                f"add {first_register}, {second_register}",
                f"neg {first_register}",
                f"neg {second_register}",
            ],
            [
                f"add {first_register}, {random_int}",
                f"add {first_register}, {second_register}",
                f"sub {first_register}, {random_int}",
            ],
            [
                f"sub {first_register}, {random_int}",
                f"add {first_register}, {second_register}",
                f"add {first_register}, {random_int}",
            ],
        ])

    def add_immediate_substitution(self, instruction: CsInsn):
        first_register = instruction.reg_name(instruction.operands[0].reg)
        immediate_value = instruction.operands[1].imm
        tmp_register = self.get_tmp_register(instruction)
        random_int = self.rng.randint(0, 1000)
        return self.rng.choice([
            [
                f"add {first_register}, {random_int}",
                f"add {first_register}, {immediate_value}",
                f"sub {first_register}, {random_int}",
            ],
            [
                f"mov {tmp_register}, {first_register}",
                f"xor {first_register}, {first_register}",
                f"add {first_register}, {tmp_register}",
                f"add {first_register}, {immediate_value}",
            ],
            [
                f"mov {tmp_register}, {first_register}",
                f"mov {first_register}, 0",
                f"add {first_register}, {tmp_register}",
                f"add {first_register}, {immediate_value}",
            ],
            [
                f"add {first_register}, {hex(random_int)}",
                f"add {first_register}, {hex(immediate_value)}",
                f"sub {first_register}, {hex(random_int)}",
            ],
            [
                f"mov {tmp_register}, {first_register}",
                f"xor {first_register}, {first_register}",
                f"add {first_register}, {tmp_register}",
                f"add {first_register}, {hex(immediate_value)}",
            ],
            [
                f"mov {tmp_register}, {first_register}",
                f"mov {first_register}, 0x0",
                f"add {first_register}, {tmp_register}",
                f"add {first_register}, {hex(immediate_value)}",
            ],
        ])

    def add_into_memory_substitution(self, instruction: CsInsn):
        base_register = instruction.reg_name(instruction.operands[0].mem.base)
        tmp_register_1 = self.get_tmp_register(instruction)
        return self.rng.choice([
            [
                f"mov {tmp_register_1}, {base_register}",
                f"add {instruction.op_str.replace(base_register, tmp_register_1)}",
            ],
        ])

    def sub_two_registers_substitution(self, instruction: CsInsn):
        first_register = instruction.reg_name(instruction.operands[0].reg)
        second_register = instruction.reg_name(instruction.operands[1].reg)
        random_int = self.rng.randint(0, 1000)
        return self.rng.choice([
            [
                f"neg {second_register}",
                f"add {first_register}, {second_register}",
                f"neg {second_register}",
            ],
            [
                f"add {first_register}, {random_int}",
                f"sub {first_register}, {second_register}",
                f"sub {first_register}, {random_int}",
            ],
            [
                f"sub {first_register}, {random_int}",
                f"sub {first_register}, {second_register}",
                f"add {first_register}, {random_int}",
            ],
        ])

    def sub_immediate_substitution(self, instruction: CsInsn):
        first_register = instruction.reg_name(instruction.operands[0].reg)
        immediate_value = instruction.operands[1].imm
        tmp_register = self.get_tmp_register(instruction)
        random_int = self.rng.randint(0, 1000)
        return self.rng.choice([
            [
                f"add {first_register}, {random_int}",
                f"sub {first_register}, {immediate_value}",
                f"sub {first_register}, {random_int}",
            ],
            [
                f"mov {tmp_register}, {first_register}",
                f"xor {first_register}, {first_register}",
                f"add {first_register}, {tmp_register}",
                f"sub {first_register}, {immediate_value}",
            ],
            [
                f"mov {tmp_register}, {first_register}",
                f"mov {first_register}, 0",
                f"add {first_register}, {tmp_register}",
                f"sub {first_register}, {immediate_value}",
            ],
            [
                f"add {first_register}, {hex(random_int)}",
                f"sub {first_register}, {hex(immediate_value)}",
                f"sub {first_register}, {hex(random_int)}",
            ],
            [
                f"mov {tmp_register}, {first_register}",
                f"xor {first_register}, {first_register}",
                f"add {first_register}, {tmp_register}",
                f"sub {first_register}, {hex(immediate_value)}",
            ],
            [
                f"mov {tmp_register}, {first_register}",
                f"mov {first_register}, 0",
                f"add {first_register}, {tmp_register}",
                f"sub {first_register}, {hex(immediate_value)}",
            ],
        ])

    def sub_into_memory_substitution(self, instruction: CsInsn):
        base_register = instruction.reg_name(instruction.operands[0].mem.base)
        tmp_register_1 = self.get_tmp_register(instruction)
        return self.rng.choice([
            [
                f"mov {tmp_register_1}, {base_register}",
                f"sub {instruction.op_str.replace(base_register, tmp_register_1)}",
            ],
        ])


    def and_two_registers_substitution(self, instruction: CsInsn):
        first_register = instruction.reg_name(instruction.operands[0].reg)
        second_register = instruction.reg_name(instruction.operands[1].reg)
        tmp_register = self.get_tmp_register(instruction)
        return self.rng.choice([
            [
                f"mov {tmp_register}, {first_register}",
                f"not {second_register}",
                f"xor {first_register}, {second_register}",
                f"and {first_register}, {tmp_register}",
                f"not {second_register}",
            ],
        ])

    def and_immediate_substitution(self, instruction: CsInsn):
        first_register = instruction.reg_name(instruction.operands[0].reg)
        immediate_value = instruction.operands[1].imm
        tmp_register = self.get_tmp_register(instruction)
        return self.rng.choice([
            [
                f"mov {tmp_register}, {first_register}",
                f"and {tmp_register}, {immediate_value}",
                f"mov {first_register}, {tmp_register}",
            ],
        ])

    def or_two_registers_substitution(self, instruction: CsInsn):
        first_register = instruction.reg_name(instruction.operands[0].reg)
        second_register = instruction.reg_name(instruction.operands[1].reg)
        tmp_register_1 = self.get_tmp_register(instruction)
        tmp_register_2 = self.get_tmp_register(instruction, [tmp_register_1])
        return self.rng.choice([
            [
                f"mov {tmp_register_1}, {first_register}",
                f"mov {tmp_register_2}, {second_register}",
                f"and {first_register}, {tmp_register_2}",
                f"xor {tmp_register_1}, {tmp_register_2}",
                f"or {first_register}, {tmp_register_1}",
            ],
        ])

    def or_immediate_substitution(self, instruction: CsInsn):
        first_register = instruction.reg_name(instruction.operands[0].reg)
        immediate_value = instruction.operands[1].imm
        tmp_register = self.get_tmp_register(instruction)
        return self.rng.choice([
            [
                f"mov {tmp_register}, {first_register}",
                f"or {tmp_register}, {immediate_value}",
                f"mov {first_register}, {tmp_register}",
            ],
        ])

    def xor_two_registers_substitution(self, instruction: CsInsn):
        first_register = instruction.reg_name(instruction.operands[0].reg)
        second_register = instruction.reg_name(instruction.operands[1].reg)
        tmp_register_1 = self.get_tmp_register(instruction)
        tmp_register_2 = self.get_tmp_register(instruction, [tmp_register_1])
        return self.rng.choice([
            [
                f"mov {tmp_register_1}, {first_register}",
                f"mov {tmp_register_2}, {second_register}",
                f"not {first_register}",
                f"and {first_register}, {tmp_register_2}",
                f"not {tmp_register_2}",
                f"and {tmp_register_1}, {tmp_register_2}",
                f"or {first_register}, {tmp_register_1}",
            ],
        ])

    def xor_immediate_substitution(self, instruction: CsInsn):
        first_register = instruction.reg_name(instruction.operands[0].reg)
        immediate_value = instruction.operands[1].imm
        tmp_register = self.get_tmp_register(instruction)
        return self.rng.choice([
            [
                f"mov {tmp_register}, {first_register}",
                f"xor {tmp_register}, {immediate_value}",
                f"mov {first_register}, {tmp_register}",
            ],
        ])

    def mov_two_registers_substitution(self, instruction: CsInsn):
        first_register = instruction.reg_name(instruction.operands[0].reg)
        second_register = instruction.reg_name(instruction.operands[1].reg)
        tmp_register_1 = self.get_tmp_register(instruction)
        tmp_register_2 = self.get_tmp_register(instruction, [tmp_register_1])
        return self.rng.choice([
            [
                f"mov {tmp_register_1}, {second_register}",
                f"mov {first_register}, {tmp_register_1}",
            ],
            [
                f"mov {tmp_register_1}, {second_register}",
                f"mov {tmp_register_2}, {tmp_register_1}",
                f"mov {first_register}, {tmp_register_2}",
            ],
        ])

    def mov_register_memory_substitution(self, instruction: CsInsn):
        base_register = instruction.reg_name(instruction.operands[1].mem.base)
        tmp_register_1 = self.get_tmp_register(instruction, [base_register])
        return self.rng.choice([
            [
                f"mov {tmp_register_1}, {base_register}",
                f"mov {instruction.op_str.replace(f'[{base_register}]', f'[{tmp_register_1}]')}",
            ],
        ])

    def mov_into_memory_substitution(self, instruction: CsInsn):
        base_register = instruction.reg_name(instruction.operands[0].mem.base)
        tmp_register_1 = self.get_tmp_register(instruction)
        return self.rng.choice([
            [
                f"mov {tmp_register_1}, {base_register}",
                f"mov {instruction.op_str.replace(base_register, tmp_register_1)}",
            ],
        ])

    def mov_immediate_substitution(self, instruction: CsInsn):
        first_register = instruction.reg_name(instruction.operands[0].reg)
        immediate_value = instruction.operands[1].imm
        tmp_register = self.get_tmp_register(instruction)
        return self.rng.choice([
            [
                f"mov {tmp_register}, {immediate_value}",
                f"mov {first_register}, {tmp_register}",
            ],
        ])
        
    def pop_substitution(self, instruction: CsInsn):
        register = instruction.reg_name(instruction.operands[0].reg)
        if register in self.registers["8"].values():
            stack_pointer = self.registers["8"]["STACK_POINTER"]
            offset = 1
        if register in self.registers["16"].values():
            stack_pointer = self.registers["16"]["STACK_POINTER"]
            offset = 2
        if register in self.registers["32"].values():
            stack_pointer = self.registers["32"]["STACK_POINTER"]
            offset = 4
        else:
            stack_pointer = self.registers["64"]["STACK_POINTER"]
            offset = 8

        return self.rng.choice([
            [
                f"mov {register}, [{stack_pointer}]",
                f"add {stack_pointer}, {offset}",
            ]
        ])
            

    def find_possible_substitution(self, instruction: CsInsn):
        if instruction.mnemonic not in MNEMONICS_WITH_KNOWN_SUBSTITUTIONS:
            return None

        if len(instruction.operands) == 1:
            if instruction.mnemonic == "pop":
                return self.pop_substitution
        if len(instruction.operands) == 1:
            if instruction.mnemonic == "neg":
                return self.negation_substitution
            return None

        if len(instruction.operands) != 2:
            return None

        def uses_invalid_register(operand):
            if operand.type != CS_OP_REG:
                return False
            register = instruction.reg_name(operand.reg)
            if register in [self.registers["64"]["STACK_POINTER"], self.registers["32"]["STACK_POINTER"]]:
                return True
            return False

        first_operand, second_operand = instruction.operands
        if uses_invalid_register(first_operand) or uses_invalid_register(second_operand):
            return None

        if first_operand.type == CS_OP_MEM:
            if instruction.reg_name(instruction.operands[1].mem.base) is None:
                return None
            if instruction.mnemonic == "add":
                return self.add_into_memory_substitution
            if instruction.mnemonic == "sub":
                return self.sub_into_memory_substitution
            if instruction.mnemonic == "mov":
                return self.mov_into_memory_substitution

            return None

        if first_operand.type == CS_OP_REG:
            if second_operand.type == CS_OP_IMM:
                if instruction.mnemonic == "add":
                    return self.add_immediate_substitution
                if instruction.mnemonic == "sub":
                    return self.sub_immediate_substitution
                if instruction.mnemonic == "and":
                    return self.and_immediate_substitution
                if instruction.mnemonic == "or":
                    return self.or_immediate_substitution
                if instruction.mnemonic == "xor":
                    return self.xor_immediate_substitution
                if instruction.mnemonic == "mov":
                    return self.mov_immediate_substitution
                return None
            if second_operand.type == CS_OP_REG:
                if instruction.mnemonic == "add":
                    return self.add_two_registers_substitution
                if instruction.mnemonic == "sub":
                    return self.sub_two_registers_substitution
                if instruction.mnemonic == "and":
                    return self.and_two_registers_substitution
                if instruction.mnemonic == "or":
                    return self.or_two_registers_substitution
                if instruction.mnemonic == "xor":
                    return self.xor_two_registers_substitution
                if instruction.mnemonic == "mov":
                    return self.mov_two_registers_substitution
                return None
            if second_operand.type == CS_OP_MEM:
                if instruction.mnemonic == "mov":
                    if instruction.reg_name(instruction.operands[1].mem.base) is None:
                        return None
                    return self.mov_register_memory_substitution

        return None

    def augment_instructions(self, instructions):
        low_pc = instructions[0].address

        was_augmented = False
        augmented_instructions = []
        for instruction in instructions:
            possible_substitution = self.find_possible_substitution(
                instruction
            )
            if possible_substitution is None:
                augmented_instructions.append(instruction)
                continue

            if self.rng.random() > self.p_substitute:
                augmented_instructions.append(instruction)
                continue

            generated_assembly_string = possible_substitution(instruction)
            generated_instructions = self.disasemble_assembly_string("\n".join(generated_assembly_string), low_pc=low_pc)
            augmented_instructions.extend(generated_instructions)
            was_augmented = True

        if not was_augmented:
            return None

        augmented_instructions = self.recreate_instructions(
            augmented_instructions, low_pc
        )

        return augmented_instructions
