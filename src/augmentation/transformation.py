
import os
from keystone import Ks, KS_ARCH_X86, KS_MODE_64
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import math
from pathlib import Path
import random
from capstone import Cs, CS_ARCH_X86, CS_MODE_64, CsInsn, CS_OP_REG, CS_OP_MEM
from tqdm import tqdm
from instruction_formatter import base64_encode_instructions, format_instructions

from config import BASE_ADDRESS

md = Cs(CS_ARCH_X86, CS_MODE_64)
md.detail = True
ks = Ks(KS_ARCH_X86, KS_MODE_64)
ks.detail = True

REGISTERS = {
    "64": {
        "RETURN_VALUE": "rax",
        "CALLEE_SAVED_1": "rbx",
        "ARGUMENT_1": "rdi",
        "ARGUMENT_2": "rsi",
        "ARGUMENT_3": "rdx",
        "ARGUMENT_4": "rcx",
        "ARGUMENT_5": "r8",
        "ARGUMENT_6": "r9",
        "FRAME_POINTER": "rbp",
        "STACK_POINTER": "rsp",
        "INSTRUCTION_POINTER": "rip",
        "TEMPORARY_1": "r10",
        "TEMPORARY_2": "r11",
        "CALLEE_SAVED_2": "r12",
        "CALLEE_SAVED_3": "r13",
        "CALLEE_SAVED_4": "r14",
        "CALLEE_SAVED_5": "r15",
    },
    "32": {
        "RETURN_VALUE": "eax",
        "CALLEE_SAVED_1": "ebx",
        "ARGUMENT_1": "edi",
        "ARGUMENT_2": "esi",
        "ARGUMENT_3": "edx",
        "ARGUMENT_4": "ecx",
        "ARGUMENT_5": "r8d",
        "ARGUMENT_6": "r9d",
        "FRAME_POINTER": "ebp",
        "STACK_POINTER": "esp",
        "INSTRUCTION_POINTER": "eip",
        "TEMPORARY_1": "r10d",
        "TEMPORARY_2": "r11d",
        "CALLEE_SAVED_2": "r12d",
        "CALLEE_SAVED_3": "r13d",
        "CALLEE_SAVED_4": "r14d",
        "CALLEE_SAVED_5": "r15d",
    },
    "16": {
        "RETURN_VALUE": "ax",
        "CALLEE_SAVED_1": "bx",
        "ARGUMENT_1": "di",
        "ARGUMENT_2": "si",
        "ARGUMENT_3": "dx",
        "ARGUMENT_4": "cx",
        "ARGUMENT_5": "r8w",
        "ARGUMENT_6": "r9w",
        "FRAME_POINTER": "bp",
        "STACK_POINTER": "sp",
        "INSTRUCTION_POINTER": "ip",
        "TEMPORARY_1": "r10w",
        "TEMPORARY_2": "r11w",
        "CALLEE_SAVED_2": "r12w",
        "CALLEE_SAVED_3": "r13w",
        "CALLEE_SAVED_4": "r14w",
        "CALLEE_SAVED_5": "r15w",
    },
    "8": {
        "RETURN_VALUE_HIGH": "ah",
        "RETURN_VALUE": "al",
        "CALLEE_SAVED_1_HIGH": "bh",
        "CALLEE_SAVED_1": "bl",
        "ARGUMENT_1_HIGH": "dih",
        "ARGUMENT_1": "dil",
        "ARGUMENT_2_HIGH": "sih",
        "ARGUMENT_2": "sil",
        "ARGUMENT_3_HIGH": "dh",
        "ARGUMENT_3": "dl",
        "ARGUMENT_4_HIGH": "ch",
        "ARGUMENT_4": "cl",
        "ARGUMENT_5": "r8b",
        "ARGUMENT_6": "r9b",
        "FRAME_POINTER": "bpl",
        "STACK_POINTER": "spl",
        "INSTRUCTION_POINTER": "ip",
        "TEMPORARY_1": "r10b",
        "TEMPORARY_2": "r11b",
        "CALLEE_SAVED_2": "r12b",
        "CALLEE_SAVED_3": "r13b",
        "CALLEE_SAVED_4": "r14b",
        "CALLEE_SAVED_5": "r15b",
    }
}

FREE_REGISTER_NAMES = ["RETURN_VALUE",
                       "ARGUMENT_1", "ARGUMENT_2", "ARGUMENT_3", "ARGUMENT_4", "ARGUMENT_5", "ARGUMENT_6",
                       "TEMPORARY_1", "TEMPORARY_2"]


class Transformation:
    def __init__(self, seed) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def get_unaugmented_instructions(self, function):
        instructions = self.disasemble(function)
        return list(instructions)

    def asemble(self, instructions, low_pc):
        return b"".join([i.bytes for i in instructions])

    def disasemble(self, function):
        ops = function["bytes"]
        low_pc = BASE_ADDRESS #function["address"]

        return list(md.disasm(ops, low_pc))

    def create_assembly_string(self, instructions):
        return f"\n".join([f"{i.mnemonic} {i.op_str}" for i in instructions])

    def recreate_instructions(self, instructions, low_pc):
        assembly_string = self.create_assembly_string(instructions)
        return self.disasemble_assembly_string(assembly_string, low_pc)

    def disasemble_assembly_string(self, assembly_string, low_pc):
        try:
            encoding, count = ks.asm(assembly_string, addr=low_pc, as_bytes=True)
        except Exception:
            return None

        return list(md.disasm(encoding, low_pc))

    def equals(self, instructions_1, instructions_2):
        s1 = "".join(['0x%x:\t%s\t%s' % (i.address, i.mnemonic, i.op_str)
                     for i in instructions_1])
        s2 = "".join(['0x%x:\t%s\t%s' % (i.address, i.mnemonic, i.op_str)
                     for i in instructions_2])
        return s1 == s2

    def get_formatted_instruction(self, instruction):
        return '0x%x:\t%s\t%s' % (
            instruction.address,
            instruction.mnemonic,
            instruction.op_str
        )

    def get_formatted_instructions(self, instructions):
        return f"\n".join([self.get_formatted_instruction(i) for i in instructions])

    def print_instruction(self, instruction):
        print(self.get_formatted_instruction(instruction=instruction))

    def print_instructions(self, instructions):
        for instruction in instructions:
            self.print_instruction(instruction)

    def augment(self):
        raise NotImplementedError

    def get_augmentation_details(self):
        raise NotImplementedError

    def get_augmentation_name(self):
        raise NotImplementedError

    @property
    def registers(self):
        return REGISTERS

    def free_registers(self, bits=None):
        bits = "64" if bits is None else bits
        registers = []
        for register_name in FREE_REGISTER_NAMES:
            registers.append(REGISTERS[bits][register_name])
        return registers

    def register_names(self, register):
        name = None
        for bits in self.registers.keys():
            name = next((name for name, r in self.registers[bits].items() if r == register), None)
            if name is not None:
                break
            
        if name is None:
            return [register]

        register_names = []
        for bits in self.registers.keys():
            if "_HIGH" in name:
                register_names.append(self.registers[bits][name.replace("_HIGH", "")])
                if name in self.registers[bits].keys():
                    register_names.append(self.registers[bits][name])
                continue
            register_names.append(self.registers[bits][name])
            if f"{name}_HIGH" in self.registers[bits].keys():
                register_names.append(self.registers[bits][f"{name}_HIGH"])

        return register_names

    def get_tmp_register(self, instruction: CsInsn, used_registers=None):
        used_registers = [] if used_registers is None else used_registers
        for operand in instruction.operands:
            if operand.type == CS_OP_REG:
                used_registers.append(instruction.reg_name(operand.reg))
            if operand.type == CS_OP_MEM:
                base_register_name = instruction.reg_name(operand.mem.base)
                if base_register_name:
                    used_registers.append(base_register_name)

        sample_register = used_registers[0]
        if sample_register in REGISTERS["8"].values():
            if sample_register in self.free_registers("8"):
                free_registers = self.free_registers("8")
            else:
                return sample_register
        elif sample_register in REGISTERS["16"].values():
            free_registers = self.free_registers("16")
        elif sample_register in REGISTERS["32"].values():
            free_registers = self.free_registers("32")
        elif sample_register in REGISTERS["64"].values():
            free_registers = self.free_registers("64")
        else:
            free_registers = []

        for tmp_register in free_registers:
            if tmp_register not in used_registers:
                return tmp_register

        logging.error(f"Unable to find tmp register for instruction {instruction}")
        return used_registers[0]


    def create_from_trial(self, seed, trial):
        raise NotImplementedError

    def augment_instructions(self, low_pc):
        raise NotImplementedError()

    def augment_function(self, function):
        function = dict(function)

        unaugmented_instructions = self.get_unaugmented_instructions(function)

        augmented_instructions = self.augment_instructions(unaugmented_instructions)
        if augmented_instructions is None:
            return None

        function["base64_encoded_ops"] = base64_encode_instructions(augmented_instructions)
        function["data"] = format_instructions(augmented_instructions)
        function["bytes"] = self.asemble(augmented_instructions, BASE_ADDRESS)

        return function
