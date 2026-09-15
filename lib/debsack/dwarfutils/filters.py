from elftools.dwarf.constants import DW_INL_inlined, DW_INL_declared_inlined


def is_formal_parameter(die):
    return die.tag == 'DW_TAG_formal_parameter'


def is_call_site(die):
    return die.tag == 'DW_TAG_GNU_call_site'


def is_subprogram(die):
    return die.tag == 'DW_TAG_subprogram'


def is_real_subprogram(die):
    if not is_subprogram(die): return False

    # some DIEs don't have attributes, not sure what they mean
    if not die.attributes: return False

    # skip trampolines:
    if 'DW_AT_trampoline' in die.attributes: return False

    # skip artificial subroutines
    if 'DW_AT_artificial' in die.attributes: return False

    # skip incomplete, non-defining subroutines
    if 'DW_AT_declaration' in die.attributes: return False

    # skip inlined subroutines
    if 'DW_AT_inline' in die.attributes:
        inline_code = die.attributes['DW_AT_inline'].value
        if inline_code == DW_INL_inlined or inline_code == DW_INL_declared_inlined:
            return False

    return True
