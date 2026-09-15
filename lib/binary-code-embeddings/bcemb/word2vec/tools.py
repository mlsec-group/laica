import itertools
import logging
import pathlib
import time

import click
import numpy as np
from capstone import *
from gensim.models import Word2Vec

from pydebtools.utils import iter_func_ops_from_deb
from .utils import encode_single, format
from ..core.corpus import Word2VecCorpus, ParallelCorpus, MemoryCorpus, FileCacheCorpus
from ..core.utils import convert_model, EpochLogger, collect_packages, pkg2cachefile

logger = logging.getLogger(__name__)

md = Cs(CS_ARCH_X86, CS_MODE_64)
md.detail = True


def configure_loggers(level):
    logging.getLogger('gensim').setLevel(level)
    logging.getLogger('pydebtools').setLevel(level)
    logging.getLogger('bcemb').setLevel(level)


@click.group()
@click.option('--debug', 'loglevel', flag_value=logging.DEBUG)
@click.option('--verbose', 'loglevel', flag_value=logging.INFO, default=True)
@click.option('--quiet', 'loglevel', flag_value=logging.ERROR)
@click.option('--warning', 'loglevel', flag_value=logging.WARNING)
def cli(loglevel):
    logging.basicConfig(format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    configure_loggers(loglevel)


@click.command()
@click.argument("path")
@click.argument("savepath")
def convert(path, savepath):
    """Load a Word2Vec model from >>path<< and convert it to plain text file stored at >>savepath<<."""
    logger.info("Loading Word2Vec model.")
    model = Word2Vec.load(path)
    convert_model(model, savepath)


@click.command()
@click.argument("path")
@click.argument("data", default='-', type=click.File('rt'))
@click.argument("out", default='-', type=click.File('wt'))
@click.option("--delimiter", default='|')
def embed(path, data, out, delimiter):
    logger.info("Loading Word2Vec model.")
    model = Word2Vec.load(path)

    for ops in data:
        low_pc, ops = ops.split(": ")
        low_pc = int(low_pc)
        encoded_insns = []
        for insn in md.disasm(bytes.fromhex(ops), low_pc):
            vector = encode_single(insn, model)
            vector_str = ' '.join(np.char.mod('%f', vector))
            encoded_insns.append(vector_str)

        out.write(delimiter.join(map(lambda vector_str: f"[{vector_str}]", encoded_insns)))
        out.write("\n")


@click.command()
@click.argument("pkg", type=click.Path(exists=True))
@click.argument("out", default='-', type=click.File('wt'))
@click.option("--format", "format_", is_flag=True, default=False)
def stream(pkg, out, format_):
    for ops, low_pc in iter_func_ops_from_deb(pkg):
        if format_:
            insns = md.disasm(ops, low_pc)
            insns = [format(insn) for insn in insns]
            out.write(f"{'|'.join(insns)}\n")
        out.write(f"{low_pc}: {ops.hex()}\n")


@click.command()
@click.argument("pkgs", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("savepath", type=click.Path(exists=False, path_type=pathlib.Path))
@click.option("--vectorsize", "vector_size", default=64)
@click.option("--window", default=5)
@click.option("--mincount", "min_count", default=1)
@click.option("--workers", default=8)
@click.option("--sg", default=1)
@click.option("--hs", default=0)
@click.option("--negative", default=5)
@click.option("--sample", default=1e-3)
@click.option("--epochs", default=100)
@click.option("--alpha", default=1e-3)
@click.option("--min-alpha", default=1e-3)
@click.option("--cachedir", default=None, type=click.Path(exists=False, path_type=pathlib.Path))
def train(pkgs, savepath, workers, cachedir, *args, **kwargs):
    """Train a Word2Vec-Model using debian packages as corpus.

    Train a Word2Vec-Model using binary code from the debian
    packages >>pkgs<< and save the model as >>savepath<<.

    See https://radimrehurek.com/gensim/models/word2vec.html
    for detailed information of the available options.
    \f

    Parameters
    ----------
    pkgs : list[pathlib.Path]
        Path list of debian packages to process.
    savepath : pathlib.Path
        Filepath to save the trained model at.

    Returns
    -------
    bool
        Return >>True<< on success, >>False<< otherwise.

    """
    if savepath.exists():
        logger.error(f"File already exists: {savepath}")
        return False

    pkgs = list(collect_packages(pkgs))

    if cachedir:
        dataset = MemoryCorpus(ParallelCorpus(
            [FileCacheCorpus(Word2VecCorpus(pkg), cachefile=cachedir / pkg2cachefile(pkg)) for pkg in pkgs],
            n_jobs=workers))
    else:
        dataset = MemoryCorpus(ParallelCorpus([Word2VecCorpus(pkg) for pkg in pkgs], n_jobs=workers))
    callbacks = [EpochLogger()]
    logger.info(f"Training Word2Vec model on {len(pkgs)} debian packages.")
    model = Word2Vec(dataset, compute_loss=True, callbacks=callbacks, workers=workers, *args, **kwargs)

    logger.info("Saving trained model.")
    savepath.parents[0].mkdir(parents=True, exist_ok=True)
    model.save(str(savepath))

    return True


@click.command()
@click.argument("pkgs", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path))
def iter_corpus(pkgs):
    pkgs = list(collect_packages(pkgs))
    dataset = itertools.chain(*(Word2VecCorpus(pkg) for pkg in pkgs))
    for function in dataset:
        print(function)


@click.command()
@click.argument("pkgs", nargs=-1, type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("cachedir", type=click.Path(exists=False, path_type=pathlib.Path))
@click.option("--workers", default=2)
def gen_corpus(pkgs, cachedir, workers):
    pkgs = list(collect_packages(pkgs))
    dataset = ParallelCorpus(
        [FileCacheCorpus(Word2VecCorpus(pkg), cachefile=cachedir / pkg2cachefile(pkg)) for pkg in pkgs], n_jobs=workers)

    unit = 1000
    start = time.time()
    c = 0
    i = 0
    for i, _ in enumerate(dataset, 1):
        c += 1
        if c == unit:
            elapsed = time.time() - start
            logger.info(f"Progress: at {i} samples, {int(unit / elapsed)} samples/s")
            start = time.time()
            c = 0
    elapsed = time.time() - start
    logger.info(f"Progress: at {i} samples, {int(unit / elapsed)} samples/s")


cli.add_command(train)
cli.add_command(convert)
cli.add_command(embed)
cli.add_command(stream)
cli.add_command(iter_corpus)
cli.add_command(gen_corpus)
