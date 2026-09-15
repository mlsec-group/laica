import gzip
import json
import logging
import pathlib
from multiprocessing import Pool, Manager

from capstone import *
from iterators import TimeoutIterator

from bcemb.asm2vec.utils import format as asm2vec_format
from bcemb.instruction2vec.utils import format as instruction2vec_format
from bcemb.word2vec.utils import format as word2vec_format
from pydebtools.utils import iter_func_ops_from_deb

md = Cs(CS_ARCH_X86, CS_MODE_64)
md.detail = True

logger = logging.getLogger(__name__)


def disasm_func(ops, low_pc):
    return md.disasm(ops, low_pc)


class Corpus:
    def __iter__(self):
        raise NotImplementedError()


class DebPkgCorpus(Corpus):

    def __init__(self, deb_pkg):
        self.deb_pkg = deb_pkg

    def _iter_functions(self):
        try:
            for ops, low_pc in iter_func_ops_from_deb(self.deb_pkg):
                yield disasm_func(ops, low_pc)
        except:
            logger.exception(f"Error processing package: {str(self.deb_pkg)}")
            raise

    def __str__(self):
        return f"DebPkgCorpus({str(self.deb_pkg)})"


class Word2VecCorpus(DebPkgCorpus):
    def __init__(self, deb_pkg):
        super().__init__(deb_pkg)

    def __iter__(self):
        for instructions in self._iter_functions():
            yield list(map(word2vec_format, instructions))

    def __str__(self):
        return f"Word2VecCorpus({str(self.deb_pkg)})"


class Instruction2VecCorpus(DebPkgCorpus):
    def __init__(self, deb_pkg):
        super().__init__(deb_pkg)

    def __iter__(self):
        for instructions in self._iter_functions():
            yield [token for insn in instructions for token in instruction2vec_format(insn)]

    def __str__(self):
        return f"Instruction2VecCorpus({str(self.deb_pkg)})"


class Asm2VecCorpus(DebPkgCorpus):
    def __init__(self, deb_pkg):
        super().__init__(deb_pkg)
        self.length = 10000

    def __iter__(self):
        for instructions in self._iter_functions():
            function = list(map(asm2vec_format, instructions))
            data = []
            n_tokens = 0
            for instruction in function:
                if n_tokens + len(instruction) >= self.length:
                    yield data
                    n_tokens = len(instruction)
                    data = [instruction]
                else:
                    n_tokens += len(instruction)
                    data.append(instruction)
            yield data

    def __str__(self):
        return f"Asm2VecCorpus({str(self.deb_pkg)})"


class MemoryCorpus(Corpus):
    def __init__(self, corpus):
        self.corpus = corpus
        self.cache = None

    def __iter__(self):
        if self.cache is not None:
            yield from self.cache
        else:
            self.cache = []
            for data in self.corpus:
                self.cache.append(data)
                yield data


class FileCacheCorpus(Corpus):

    def __init__(self, corpus, cachefile):
        self.corpus = corpus
        self.cachefile = pathlib.Path(cachefile)
        self.cachefile.parent.mkdir(exist_ok=True, parents=True)

    def __iter__(self):
        if self.cachefile.exists():
            logger.debug("Loading data from cache.")
            with gzip.open(self.cachefile, "rt") as cache:
                for data in cache:
                    yield json.loads(data)
        else:
            with gzip.open(self.cachefile, "wt") as cache:
                for data in self.corpus:
                    yield data
                    print(json.dumps(data), file=cache)

    def __str__(self):
        return f"FileCacheCorpus({self.corpus})"


class TimeoutCorpus(Corpus):
    def __init__(self, corpus, timeout):
        self.corpus = corpus
        self.timeout = timeout

    def __iter__(self):
        for data in TimeoutIterator(iter(self.corpus), timeout=self.timeout, sentinel=None):
            if data is None:
                logger.warning(f"No data available after timeout. Stop iteration of {str(self.corpus)}.")
                return
            else:
                yield data

    def __str__(self):
        return f"TimeoutCorpus({str(self.corpus)})"


class ParallelCorpus(Corpus):
    QUEUE_SIZE_FACTOR = 2048

    def __init__(self, corpora, n_jobs=None):
        self.corpora = corpora
        if n_jobs is None:
            self.n_jobs = max(1, len(corpora))
        else:
            self.n_jobs = max(1, min(len(corpora), n_jobs))
        self.queue = Manager().Queue(self.n_jobs * self.QUEUE_SIZE_FACTOR)

    def __iter__(self):
        remaining = len(self.corpora)

        with Pool(self.n_jobs) as p:
            for corpus in self.corpora:
                p.apply_async(self._iter_corpus, (corpus,))

            while remaining > 0:
                data = self.queue.get()
                if data is None:
                    remaining -= 1
                    logger.info(f"{remaining} package(s) remaining")
                else:
                    yield data

    def _iter_corpus(self, corpus):
        try:
            logger.debug(f"Starting iteration of corpus {str(corpus)}.")
            for data in corpus:
                self.queue.put(data)
            logger.debug(f"Iteration of corpus {str(corpus)} complete.")
        except:
            logger.exception("Error while iterating corpus: " + str(corpus))
        finally:
            # Put sentinel
            self.queue.put(None)

    def __str__(self):
        return f"ParallelCorpus({len(self.corpora)})"
