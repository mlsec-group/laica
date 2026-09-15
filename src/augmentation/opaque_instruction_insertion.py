from augmentation.transformation import Transformation
import random


class OpaqueInstructionInsertion(Transformation):
    def __init__(self, instruction_count, seed, **kwargs) -> None:
        super().__init__(seed=seed, **kwargs)
        self.instruction_count = instruction_count
        self.seed = seed
        random.seed(seed)
        self.rng = random.Random(seed)

    def generate_random_x86_64_assembly(self):
        instructions = [
            "mov", "add", "sub", "xor", "and", "or", "cmp", "inc", "dec",
            "push", "pop", "lea", "imul", "shr", "shl", "nop"
        ]

        registers = self.free_registers()

        immediate_values = [0, 1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

        assembly_code = []
        for _ in range(self.instruction_count):
            instruction = self.rng.choice(instructions)

            if instruction in ["mov", "add", "sub", "xor", "and", "or", "cmp", "imul"]:
                reg1 = self.rng.choice(registers)
                if self.rng.random() < 0.5:
                    reg2 = self.rng.choice(registers)
                    assembly_code.append(f"{instruction} {reg1}, {reg2}")
                else:
                    imm = self.rng.choice(immediate_values)
                    assembly_code.append(f"{instruction} {reg1}, {imm}")

            elif instruction in ["inc", "dec", "push", "pop"]:
                reg = self.rng.choice(registers)
                assembly_code.append(f"{instruction} {reg}")

            elif instruction == "lea":
                reg1 = self.rng.choice(registers)
                reg2 = self.rng.choice(registers)
                imm = self.rng.choice(immediate_values)
                assembly_code.append(f"{instruction} {reg1}, [{reg2} + {imm}]")

            elif instruction in ["shr", "shl"]:
                reg = self.rng.choice(registers)
                imm = self.rng.choice([1, 2, 3])
                assembly_code.append(f"{instruction} {reg}, {imm}")

            elif instruction == "nop":
                assembly_code.append("nop")


        return "\n".join(assembly_code)

    def get_augmentation_name(self):
        return "opaque_instruction_insertion"

    def get_augmentation_details(self):
        return {
            "instruction_count": self.instruction_count,
        }

    def generate_opaque_instructions(self, start_address):
        opaque_instruction_assembly_string = self.generate_random_x86_64_assembly()

        opaque_instructions = self.disasemble_assembly_string(
            assembly_string=opaque_instruction_assembly_string,
            low_pc=start_address
        )

        if opaque_instructions is None:
            return None, None

        total_size = sum(ins.size for ins in opaque_instructions) + 2

        jump_assembly = f"jmp {hex(start_address + total_size)}"
        jump_instruction = self.disasemble_assembly_string(
            assembly_string=jump_assembly,
            low_pc=start_address
        )[0]

        return opaque_instructions, jump_instruction

    def augment_instructions(self, instructions):
        if self.instruction_count == 0 or len(instructions) < 1:
            return None

        low_pc = instructions[0].address
        augmented_instructions = list(instructions)

        insertion_index = self.rng.randint(0, len(augmented_instructions) - 1)

        before_insertion_address = augmented_instructions[insertion_index].address

        opaque_instructions, jump_instruction = self.generate_opaque_instructions(
            start_address=before_insertion_address
        )
        if opaque_instructions is None:
            return None

        augmented_instructions.insert(insertion_index, jump_instruction)
        for idx, opaque_instruction in enumerate(opaque_instructions):
            augmented_instructions.insert(
                insertion_index + 1 + idx, opaque_instruction
            )

        augmented_instructions = self.recreate_instructions(augmented_instructions, low_pc)

        return augmented_instructions
