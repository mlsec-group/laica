import logging

from debsack.funsig import md

logger = logging.getLogger(__name__)


def iter_call_sites(function):
    return sorted(function['data']['call_sites'], key=lambda x: x['low_pc'])


def disasm_hex(hex_ops, low_pc):
    yield from disasm(bytes.fromhex(hex_ops), low_pc)


def disasm(ops, low_pc):
    for insn in md.disasm(ops, low_pc):
        yield insn


def disasm_function(function):
    yield from disasm_hex(function['data']['ops'], function['metadata']['low_pc'])


def load_global_name_func_map(functions):
    logging.info("Building function id to function mapping.")
    mapping = {}
    duplicate_function_ids = set()
    for func in functions:
        func_id = func['metadata']['id']
        if func_id in mapping:
            logger.warning(
                f"The function {func_id} is already in the  map. "
                f"This might happen due to partial function inlining/outlining. "
                f"All functions with this name will be removed from the map.")
            duplicate_function_ids.add(func_id)
        else:
            mapping[func_id] = func
    for func_id in duplicate_function_ids:
        logger.info(f"removing non unique function: {func_id}")
        del mapping[func_id]
    return mapping
