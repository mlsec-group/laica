from typing import List

import angr
import pyvex
import archinfo
from augmentation.transformation import Transformation
from capstone import CsInsn

arch = archinfo.ArchX86()

class BasicBlockReordering(Transformation):
    def __init__(self, p_reorder, keep_first_block=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p_reorder = p_reorder
        self.keep_first_block = keep_first_block

    def get_augmentation_name(self):
        return "basic_block_reordering"

    def get_augmentation_details(self):
        return {
            "keep_first_block": self.keep_first_block,
            "p_reorder": self.p_reorder,
        }

    def get_basic_blocks(self, function):
        arch = archinfo.ArchX86()
        binary_data = function["bytes"]
        start_addr = 0x4000 #function["low_pc"]
        disasembled_insns = self.disasemble(function=function)

        basic_blocks = []
        count = 0
        bytes_offset = 0
        while count < len(disasembled_insns):
            irsb = pyvex.lift(binary_data, start_addr, arch, bytes_offset=bytes_offset)
            instruction_count = max(irsb.instructions, 1)
            basic_blocks.append(disasembled_insns[count:count+instruction_count])
            count += instruction_count
            bytes_offset += irsb.size

        return basic_blocks

    def reorder(self, basic_blocks: List[CsInsn]):
        reordered_blocks = list(basic_blocks)
        basic_block_count = len(reordered_blocks)
        
        unused_blocks = []
        for i in range(1, basic_block_count):
            if self.rng.random() <= self.p_reorder:
                unused_blocks.append(basic_blocks[i])
                reordered_blocks[i] = None
            else:
                reordered_blocks[i] = basic_blocks[i]

        self.rng.shuffle(unused_blocks)
        for i in range(1, basic_block_count):
            if reordered_blocks[i] is None:
                reordered_blocks[i] = unused_blocks.pop(0)

        return reordered_blocks

    def reorder_basic_blocks(self, basic_blocks: List[List[CsInsn]]):
        basic_block_count = len(basic_blocks)
        if basic_block_count < 3:
            return None

        if self.keep_first_block:
            first_block = basic_blocks[0]
            remaining_blocks = basic_blocks[1:]

            reordered_blocks = self.reorder(remaining_blocks)
            if reordered_blocks is None:
                return None

            return [first_block] + reordered_blocks

        reordered_blocks = self.reorder(basic_blocks)
        return reordered_blocks

    def get_instructions_from_basic_blocks(self, basic_blocks: List[angr.Block]):
        instructions = []
        for block in basic_blocks:
            for instruction in block.capstone.insns:
                instructions.append(instruction)

        return instructions

    def augment_function(self, function):
        function = dict(function)
        
        basic_blocks = self.get_basic_blocks(function)

        reordered_basic_blocks = self.reorder_basic_blocks(basic_blocks)

        if reordered_basic_blocks is None:
            return None

        reordered_binary_data = b"".join(
            self.asemble(block, 0) for block in reordered_basic_blocks
        )

        function["bytes"] = reordered_binary_data

        return function
