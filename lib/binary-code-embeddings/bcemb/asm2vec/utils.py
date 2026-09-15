import logging
import struct

import numpy as np
from capstone import CS_OP_REG, CS_OP_IMM, CS_OP_MEM

logger = logging.getLogger(__name__)


def format(insn):
    def to_x(value):
        return "".join("{0:02x}".format(b) for b in struct.pack(">q", value))

    def format_reg(operand):
        return [insn.reg_name(operand.reg)]

    def format_imm(operand):
        return [to_x(operand.imm)]

    def format_mem(operand):
        def parts():
            if operand.mem.base != 0:
                yield insn.reg_name(operand.mem.base)
            if operand.mem.index != 0:
                yield insn.reg_name(operand.mem.index)
            if operand.mem.scale != 1:
                yield str(operand.mem.scale)
            if operand.mem.disp != 0:
                yield to_x(operand.mem.disp)

        return list(parts())

    def format_operand(operand):
        if operand.type == CS_OP_REG:
            return format_reg(operand)
        elif operand.type == CS_OP_IMM:
            return format_imm(operand)
        elif operand.type == CS_OP_MEM:
            return format_mem(operand)
        else:
            raise RuntimeError(f"Unknown operand type: {operand.type}")

    return [insn.mnemonic] + [token for operand in insn.operands for token in format_operand(operand)]


def encode_single(insn, model, verbose = True):
    def encode_token(token):
        try:
            return model.wv[token].astype(np.float)
        except:
            if verbose:
                logger.warning(f"Out of vocabulary: {token}")
            return np.zeros((model.vector_size,))

    def encode(tokens):
        operation = tokens[0]
        v_operation = encode_token(operation)
        if len(tokens) > 1:
            v_operands = np.mean([encode_token(token) for token in tokens[1:]], axis=0)
        else:
            v_operands = np.zeros((model.vector_size,))
        v_instruction = np.concatenate([v_operation, v_operands])
        return v_instruction

    return encode(format(insn))


def encode(insns, model, verbose = True):
    vectors = []
    for insn in insns:
        vectors.append(encode_single(insn, model, verbose))
    return np.asarray(vectors)
