import logging

import numpy as np
from capstone import *

logger = logging.getLogger(__name__)


def format(insn):
    def iter_parts():
        yield insn.mnemonic
        for operand in insn.operands:
            if operand.type == CS_OP_REG:
                yield insn.reg_name(operand.reg)
            elif operand.type == CS_OP_IMM:
                yield str(operand.imm)
            elif operand.type == CS_OP_MEM:
                if operand.mem.base != 0:
                    yield insn.reg_name(operand.mem.base)
                if operand.mem.index != 0:
                    yield insn.reg_name(operand.mem.index)
                if operand.mem.scale != 1:
                    yield str(operand.mem.scale)
                if operand.mem.disp:
                    yield str(operand.mem.disp)

    res = list(iter_parts())
    return res


def encode_single(insn, model):
    def encode(word):
        try:
            return model.wv[word].astype(np.float)
        except KeyError:
            logger.warning(f"Out of vocabulary: {word}")
            return np.zeros((model.vector_size,))

    size = model.vector_size
    max_operands = 2  # other operands will be discarded
    vector = np.zeros((9, size), dtype=np.float)
    vector[0] = encode(insn.mnemonic)
    for i, operand in enumerate(insn.operands[:max_operands]):
        if operand.type == CS_OP_REG:
            vector[4 * i + 1] = encode(insn.reg_name(operand.reg))
        elif operand.type == CS_OP_IMM:
            vector[4 * i + 2][size - 1] = operand.imm
        elif operand.type == CS_OP_MEM:
            if operand.mem.base != 0:
                vector[4 * i + 1] = encode(insn.reg_name(operand.mem.base))
            vector[4 * i + 2][size - 1] = operand.mem.disp
            if operand.mem.index != 0:
                vector[4 * i + 3] = encode(insn.reg_name(operand.mem.index))
                vector[4 * i + 4] = encode(str(operand.mem.scale))
        else:
            logger.warning(f"Unknown operand type: {operand.type}")

    return vector.flatten()


def encode(insns, model):
    vectors = []
    for insn in insns:
        vectors.append(encode_single(insn, model))
    return np.asarray(vectors)
