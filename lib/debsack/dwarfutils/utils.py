import logging

from dwarfutils.dientries import FormalParameterDIE, GNUCallSiteDIE
from dwarfutils.filters import is_formal_parameter, is_call_site

logger = logging.getLogger(__name__)

def type2str(type):
    if type.tag == "DW_TAG_base_type":
        return type.attributes["DW_AT_name"].value.decode().replace(" ", "_")
    elif type.tag == "DW_TAG_structure_type":
        return type.tag
    elif type.tag == "DW_TAG_union_type":
        return type.tag
    elif type.tag == "DW_TAG_pointer_type":
        try:
            return f"{type.tag}.{type2str(type.get_DIE_from_attribute('DW_AT_type'))}"
        except KeyError:
            return f"{type.tag}.void"
    elif type.tag == "DW_TAG_const_type":
        try:
            return type2str(type.get_DIE_from_attribute('DW_AT_type'))
        except KeyError:
            return "void"
    elif type.tag == "DW_TAG_typedef":
        try:
            return type2str(type.get_DIE_from_attribute('DW_AT_type'))
        except KeyError:
            return "void"
    elif type.tag == "DW_TAG_enumeration_type":
        return type2str(type.get_DIE_from_attribute('DW_AT_type'))
    elif type.tag == "DW_TAG_restrict_type":
        try:
            return type2str(type.get_DIE_from_attribute('DW_AT_type'))
        except KeyError:
            return "void"
    else:
        logger.debug(f"Unknown tag: {type.tag}")
        try:
            return f"{type.tag}.{type2str(type.get_DIE_from_attribute('DW_AT_type'))}"
        except KeyError:
            return f"{type.tag}.void"


def iter_formal_parameters(subprogram):
    yield from map(FormalParameterDIE, filter(is_formal_parameter, subprogram.iter_children()))


def iter_call_sites(subprogram):
    yield from map(GNUCallSiteDIE, filter(is_call_site, subprogram.iter_children()))
