import logging
import os

from elftools.dwarf.constants import DW_LANG_C, DW_LANG_C89, DW_LANG_C99, DW_LANG_C11

from dwarfutils.dientries import CompileUnitDIE, SubProgramDIE
from dwarfutils.filters import is_real_subprogram

logger = logging.getLogger(__name__)


def extract_metadata(subprogram):
    compile_unit = CompileUnitDIE(subprogram._die.cu.get_top_DIE())
    full_path = compile_unit.get_full_path()
    norm_full_path = os.path.normpath(full_path)
    metadata = {
        'name': subprogram.name,
        'low_pc': subprogram.low_pc,
        'full_path': norm_full_path,
        'external': bool(subprogram.external),
        'producer': compile_unit.producer,
        'language': compile_unit.language,
        'offset': hex(subprogram.offset),
        'cu_offset': hex(subprogram.cu.cu_offset),
    }

    # Only names of external (global scope) functions are unique.
    # We add the file path to the id of non-external functions.
    if subprogram.external:
        metadata['id'] = subprogram.name
    else:
        metadata['id'] = f"{norm_full_path.replace('/', '__')}:{subprogram.name}"

    return metadata


class BaseModule:
    def __init__(self, elffile):
        self.__elffile = elffile
        self.dwarfinfo = self.__elffile.get_dwarf_info()
        code = self.__elffile.get_section_by_name('.text')
        self.__operations = code.data()
        self.__sh_addr = code['sh_addr']

    def iter_data(self):
        pass

    def _ops(self, low_pc, high_pc):
        ops = self.__operations[low_pc - self.__sh_addr: high_pc - self.__sh_addr]
        return ops


class FunctionModule(BaseModule):

    def _process_compile_unit(self, compile_unit):
        for subprogram in map(SubProgramDIE, filter(is_real_subprogram, compile_unit.cu.iter_DIEs())):
            try:
                yield self._process_subprogram(subprogram)
            except NotImplementedError as e:
                logger.debug(str(e))
            except Exception:
                logger.exception("Error processing subprogram.")

    def _process_subprogram(self, subprogram):
        # Skip all programs with non-contiguous ranges
        if subprogram.low_pc is None or subprogram.high_pc is None:
            raise NotImplementedError("Function with non-continuous address range")

        return {
            'metadata': extract_metadata(subprogram),
            'data': self.extract_data(subprogram),
            'label': self.label(subprogram),
        }

    def iter_data(self):
        for cu in self.dwarfinfo.iter_CUs():
            top_DIE = cu.get_top_DIE()
            if top_DIE.tag != 'DW_TAG_compile_unit':
                # The top DIE can be a partial unit (DW_TAG_partial_unit)
                # We ignore these for now.
                continue
            compile_unit = CompileUnitDIE(top_DIE)
            if compile_unit.language in [DW_LANG_C, DW_LANG_C89, DW_LANG_C99, DW_LANG_C11]:
                yield from self._process_compile_unit(compile_unit)

    def extract_data(self, subprogram):
        pass

    def label(self, subprogram):
        pass
