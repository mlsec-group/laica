import argparse
import json
import logging
import pathlib

from elftools.elf.elffile import ELFFile

from debsack.funsig.module import FunctionSignatureModule
from pydebtools.utils import iter_elffiles_from_deb

logger = logging.getLogger(__name__)


def extract_data_from_elffile(elf_fileobj, binary):
    elffile = ELFFile(elf_fileobj)

    if not elffile.has_dwarf_info():
        logger.error(f"Debug information not found: {binary}")
        return

    if not elffile.get_section_by_name(".text"):
        logger.error(f"Section (.text) not found: {binary}")
        return

    module = FunctionSignatureModule(elffile)

    functions = list(module.iter_data())
    data = {
        'binary': binary,
        'functions': functions,
    }
    print(json.dumps(data, indent=2))


def extract_data(pkg):
    for elffile in iter_elffiles_from_deb(pkg):
        extract_data_from_elffile(*elffile)


def funsig():
    parser = argparse.ArgumentParser()
    parser.add_argument("pkg", type=pathlib.Path, help='the debian package')
    args = parser.parse_args()

    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s] | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    extract_data(args.pkg)
