import logging
import re
import sys

import torch.cuda
from pkg_resources import resource_filename

palmtree_zip = resource_filename(__name__, "/blob/PalmTree.zip")
sys.path.insert(0, f'{palmtree_zip}/PalmTree')

logger = logging.getLogger(__name__)

import config as palmtree_config

use_cuda = torch.cuda.is_available()
palmtree_config.USE_CUDA = use_cuda
import eval_utils as palmtree_utils

if not use_cuda:
    torch.set_num_threads(1)

palmtree = palmtree_utils.UsableTransformer(model_path=resource_filename(__name__, "blob/transformer.ep19"),
                                            vocab_path=resource_filename(__name__, "blob/vocab"))


def format(insn):
    insn = f"{insn.mnemonic} {insn.op_str}" if insn.op_str else f"{insn.mnemonic}"

    insn = insn.replace(":[", ": [")

    insn = palmtree_utils.parse_instruction(insn, {}, {})
    parts = insn.split()
    # replace decimal numbers with hex representation
    parts = [re.sub(r"^(\d)$", r"0x\g<1>", x) for x in parts]
    # drop 'ptr' token
    parts = [x for x in parts if x != "ptr"]
    insn = " ".join(parts)
    return insn


def encode(insns):
    insns = list(insns)[:128]
    text = list(map(format, insns))
    return palmtree.encode(text).astype(float)


def dump_vocab():
    for x, y in enumerate(palmtree.vocab.itos, 1):
        print(f"{x:4d} {y}")
