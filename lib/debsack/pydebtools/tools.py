import argparse
import logging
import pathlib

from elftools.common.exceptions import ELFError, ELFRelocationError
from elftools.elf.elffile import ELFFile

from dwarfutils.dientries import CompileUnitDIE
from pydebtools.utils import iter_elffiles_from_deb

logger = logging.getLogger(__name__)


def deb_info():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkg", type=pathlib.Path, help='the debian package')
    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s] | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    for elf_fileobj, binary in iter_elffiles_from_deb(args.pkg):
        try:
            elffile = ELFFile(elf_fileobj)
            if not elffile.has_dwarf_info():
                logger.error("No dwarf debug information found")
            dwarfinfo = elffile.get_dwarf_info()
            for cu in dwarfinfo.iter_CUs():
                cu = CompileUnitDIE(cu.get_top_DIE())
                print(f"\tcompilation unit: {cu.get_full_path()}")
                print(f"\t\tDW_AT_name:     {cu.name}")
                print(f"\t\tDW_AT_producer: {cu.producer}")
                print(f"\t\tDW_AT_language: {cu.language}")
        except ELFRelocationError as e:
            logger.error(str(e))
        except ELFError:
            logger.exception("Error accessing dwarf info")
            continue
