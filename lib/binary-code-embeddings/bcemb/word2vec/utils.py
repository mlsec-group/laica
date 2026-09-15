import logging

import numpy as np

from bcemb.asm2vec.utils import format as asm2vec_format

logger = logging.getLogger(__name__)


# def format(insn):
#     def format_reg(operand):
#         return insn.reg_name(operand.reg)
#
#     def format_imm(operand):
#         return f"$IMM{operand.size}"
#
#     def format_mem(operand):
#         mem = operand.mem
#         base_reg = insn.reg_name(mem.base)
#         base_reg = base_reg if base_reg != None else "0"
#         index_reg = insn.reg_name(mem.index)
#         index_reg = index_reg if index_reg != None else "0"
#         disp_sign = "-" if mem.disp < 0 else "+"
#         disp_str = f"{disp_sign}$DISP" if mem.disp else "0"
#         scale = mem.scale
#
#         return f"[{operand.size}] [{base_reg} {index_reg} {scale} {disp_str}]"
#
#     def format_operand(operand):
#         if operand.type == CS_OP_REG:
#             return format_reg(operand)
#         elif operand.type == CS_OP_IMM:
#             return format_imm(operand)
#         elif operand.type == CS_OP_MEM:
#             return format_mem(operand)
#         else:
#             raise RuntimeError(f"Unknown operand type: {operand.type}")
#
#     op_str = ', '.join(format_operand(operand) for operand in insn.operands)
#     return f"{insn.mnemonic}{' ' if op_str else ''}{op_str}"

def format(insn):
    tokens = asm2vec_format(insn)
    return tokens[0] + " " + ", ".join(tokens[1:])


def encode_single(insn, model):
    def encode(word):
        try:
            return model.wv[word].astype(np.float)
        except KeyError:
            logger.warning(f"Out of vocabulary: {word}")
            return np.zeros((model.vector_size,))

    return encode(format(insn))


def encode(insns, model):
    vectors = []
    for insn in insns:
        vectors.append(encode_single(insn, model))
    return np.asarray(vectors)
