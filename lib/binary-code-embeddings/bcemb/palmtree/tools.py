import logging

import click
from capstone import *

from pydebtools.utils import iter_func_ops_from_deb
from .utils import dump_vocab as dump_vocab_
from .utils import encode, format

logger = logging.getLogger(__name__)

md = Cs(CS_ARCH_X86, CS_MODE_64)
md.detail = True


def configure_loggers(level):
    pass


@click.group()
@click.option('--debug', 'loglevel', flag_value=logging.DEBUG)
@click.option('--verbose', 'loglevel', flag_value=logging.INFO, default=True)
@click.option('--quiet', 'loglevel', flag_value=logging.ERROR)
@click.option('--warning', 'loglevel', flag_value=logging.WARNING)
def cli(loglevel):
    logging.basicConfig(format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    configure_loggers(loglevel)


@click.command()
@click.argument("data", default='-', type=click.File('rt'))
@click.argument("savepath", default='-', type=click.File('wt'))
def embed(data, savepath):
    for i, ops in enumerate(data, 1):
        ops = ops.strip()
        low_pc, ops = ops.split(": ")
        low_pc = int(low_pc)
        x = encode(md.disasm(bytes.fromhex(ops), low_pc))
        savepath.write(str(x))
        savepath.write("\n")


@click.command()
@click.argument("pkg", type=click.Path(exists=True))
@click.argument("out", default='-', type=click.File('wt'))
@click.option("--format", "format_", is_flag=True, default=False)
def stream(pkg, out, format_):
    for ops, low_pc in iter_func_ops_from_deb(pkg):
        if format_:
            insns = md.disasm(ops, low_pc)
            insns = [format(insn) + f"\t{insn.mnemonic} {insn.op_str}" for insn in insns]
            out.write(f"{'|'.join(insns)}\n")
        out.write(f"{low_pc}: {ops.hex()}\n")


@click.command()
def dump_vocab():
    dump_vocab_()


@click.command()
def train():
    raise NotImplementedError()


@click.command()
def gen_corpus():
    raise NotImplementedError()


cli.add_command(train)
cli.add_command(embed)
cli.add_command(stream)
cli.add_command(dump_vocab)
cli.add_command(gen_corpus)
