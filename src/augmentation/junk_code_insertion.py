from augmentation.transformation import Transformation

single_no_effect_instructions = [
    "nop",
    "add",
    "sub",
    "mov",
    "and",
    "or",
    "lea"
]


class JunkCodeInsertion(Transformation):
    def __init__(self, p_insert_junk_code, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p_insert_junk_code = p_insert_junk_code

    def get_augmentation_name(self):
        return "junk_code_insertion"

    def get_augmentation_details(self):
        return {
            "p_insert_junk_code": self.p_insert_junk_code,
        }

    def generate_single_no_effect_instruction(self):
        instruction_decision = self.rng.choice(single_no_effect_instructions)

        free_registers = self.free_registers("64") + self.free_registers("32")

        if instruction_decision == "nop":
            return "nop"

        if instruction_decision == "add":
            register = self.rng.choice(free_registers)
            return f"add {register}, 0"

        if instruction_decision == "sub":
            register = self.rng.choice(free_registers)
            return f"sub {register}, 0"

        if instruction_decision == "mov":
            register = self.rng.choice(free_registers)
            return f"mov {register}, {register}"

        if instruction_decision == "mul":
            register = self.rng.choice(free_registers)
            return f"mul {register}, 1"

        if instruction_decision == "and":
            register = self.rng.choice(free_registers)
            return f"and {register}, {register}"

        if instruction_decision == "or":
            register = self.rng.choice(free_registers)
            return f"or {register}, {register}"

        if instruction_decision == "lea":
            register = self.rng.choice(free_registers)
            return f"lea {register}, [{register}]"

        raise NotImplementedError(
            f"Unknown instruction {instruction_decision}"
        )

    def generate_dual_no_effect_instructions(self):
        decision = self.rng.randint(0, 6)
        register = self.rng.choice([
            self.registers["64"]["RETURN_VALUE"],
            self.registers["64"]["ARGUMENT_1"],
            self.registers["64"]["ARGUMENT_2"],
            self.registers["64"]["ARGUMENT_3"],
            self.registers["64"]["ARGUMENT_4"],
            self.registers["64"]["ARGUMENT_5"],
            self.registers["64"]["ARGUMENT_6"],
            self.registers["64"]["TEMPORARY_1"],
            self.registers["64"]["TEMPORARY_2"],
            self.registers["64"]["CALLEE_SAVED_1"],
            self.registers["64"]["CALLEE_SAVED_2"],
            self.registers["64"]["CALLEE_SAVED_3"],
            self.registers["64"]["CALLEE_SAVED_4"],
        ])

        if decision == 0:
            return [
                f"push {self.rng.randint(0, 2 ** 64)}",
                f"pop {register}",
            ]

        if decision == 1:
            return [
                f"push {register}",
                f"pop {register}",
            ]

        if decision == 2:
            return [
                f"dec {register}",
                f"inc {register}",
            ]

        if decision == 3:
            return [
                f"inc {register}",
                f"dec {register}",
            ]

        if decision == 4:
            offset = 2 ** self.rng.randint(0, 8)
            return [
                f"add {self.registers['64']['STACK_POINTER']}, {offset}",
                f"sub {self.registers['64']['STACK_POINTER']}, {offset}"
            ]

        if decision == 5:
            offset = 2 ** self.rng.randint(0, 8)
            return [
                f"sub {self.registers['64']['STACK_POINTER']}, {offset}",
                f"add {self.registers['64']['STACK_POINTER']}, {offset}"
            ]

        if decision == 6:
            random_int = self.rng.randint(0, 2048)
            return [
                f"add {register}, {random_int}",
                f"sub {register}, {random_int}",
            ]

        if decision == 7:
            random_int = self.rng.randint(0, 2048)
            return [
                f"sub {register}, {random_int}",
                f"add {register}, {random_int}",
            ]

    def create_no_effect_instructions(self, low_pc):
        if self.rng.random() > 0.5:
            sequence = self.generate_dual_no_effect_instructions()
        else:
            sequence = [self.generate_single_no_effect_instruction()]
        return self.disasemble_assembly_string("\n".join(sequence), low_pc=low_pc)

    def augment_instructions(self, instructions):
        low_pc = instructions[0].address

        was_augmented = False
        augmented_instructions = []
        for instruction in instructions:
            augmented_instructions.append(instruction)
            if self.rng.random() <= self.p_insert_junk_code:
                no_effect_instructions = self.create_no_effect_instructions(
                    low_pc=low_pc
                )
                if no_effect_instructions is not None:
                    augmented_instructions.extend(no_effect_instructions)
                    was_augmented = True

        if not was_augmented:
            return None

        augmented_instructions = self.recreate_instructions(
            augmented_instructions, low_pc
        )

        return augmented_instructions
