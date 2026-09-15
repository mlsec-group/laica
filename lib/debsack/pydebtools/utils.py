import logging
from dataclasses import dataclass

from elftools.common.exceptions import ELFRelocationError, ELFError
from elftools.dwarf.constants import DW_LANG_C, DW_LANG_C89, DW_LANG_C99, DW_LANG_C11
from elftools.elf.elffile import ELFFile

from dwarfutils.dientries import CompileUnitDIE, SubProgramDIE
from dwarfutils.filters import is_real_subprogram
from pydebtools.debfile import DebFile

logger = logging.getLogger(__name__)


def iter_elffiles_from_deb(pkg):
    with DebFile(pkg) as debfile:
        yield from debfile.iter_elffiles()


def iter_elffiles_from_deb_with_filter(pkg):
    for elf_fileobj, binary in iter_elffiles_from_deb(pkg):
        elffile = ELFFile(elf_fileobj)

        if not elffile.has_dwarf_info():
            logger.error(f"Debug information not found: Package {pkg} (binary: {binary})")
            continue

        if not elffile.get_section_by_name(".text"):
            logger.error(f"Section (.text) not found: Package {pkg} (binary: {binary})")
            continue

        yield elffile


@dataclass(frozen=True)
class FunctionObject:
    ops: bytes
    die: SubProgramDIE

    @property
    def addr(self):
        return self.die.low_pc

    @property
    def length(self):
        return len(self.ops)

    @property
    def name(self):
        return self.die.name

    def __str__(self) -> str:
        return f"Function {self.name} ({self.length} bytes @ 0x{self.addr:x})"


def iter_func_objs_from_elf(elffile):
    """
    Yields the bytes of each function in the given elf file.
    :param elffile: the elf file
    """
    dwarfinfo = elffile.get_dwarf_info()
    code = elffile.get_section_by_name('.text')
    code_data = code.data()
    sh_addr = code['sh_addr']

    for cu in dwarfinfo.iter_CUs():
        top_DIE = cu.get_top_DIE()
        if top_DIE.tag != 'DW_TAG_compile_unit':
            # The top DIE can be a partial unit (DW_TAG_partial_unit)
            # We ignore these for now.
            continue
        compile_unit = CompileUnitDIE(cu.get_top_DIE())
        if compile_unit.language not in [DW_LANG_C, DW_LANG_C89, DW_LANG_C99, DW_LANG_C11]:
            logger.debug(f"Skipping compilation unit. Language ({compile_unit.language}) not supported.")
            continue

        for subprogram in map(SubProgramDIE, filter(is_real_subprogram, compile_unit.cu.iter_DIEs())):
            low_pc = subprogram.low_pc
            high_pc = subprogram.high_pc

            if low_pc is None or high_pc is None:
                logger.debug(f"Skipping subprogram. Unsupported address range.")
                continue

            ops = code_data[low_pc - sh_addr: high_pc - sh_addr]
            yield FunctionObject(ops, subprogram)


def iter_func_ops_from_elf(elffile):
    """
    Yields the bytes of each function in the given elf file.
    :param elffile: the elf file
    """
    for fun_obj in iter_func_objs_from_elf(elffile):
        yield fun_obj.ops, fun_obj.die.low_pc


def iter_func_ops_from_deb(pkg):
    """
    Yields the bytes of each function in the given deb package.
    :param pkg: the deb package
    """
    for elffile in iter_elffiles_from_deb_with_filter(pkg):
        try:
            yield from iter_func_ops_from_elf(elffile)
        except ELFRelocationError as e:
            logger.debug(str(e))
            continue
        except ELFError:
            logger.exception(f"Error accessing dwarf info. Package: {pkg}")
            continue
        except Exception:
            logger.exception(f"Error processing ELF file. Package: {pkg}")
            continue
