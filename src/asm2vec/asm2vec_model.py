import base64
import os
import random
import numpy as np
from tqdm import tqdm
from bcemb.asm2vec.utils import format as asm2vec_format, encode
from bcemb.core.corpus import ParallelCorpus, MemoryCorpus, Corpus
from bcemb.core.utils import EpochLogger
from gensim.models import Asm2Vec
from capstone import CS_ARCH_X86, CS_MODE_64, Cs

md = Cs(CS_ARCH_X86, CS_MODE_64)
md.detail = True

def disasm_func(ops, low_pc):
    return md.disasm(ops, low_pc)


class Asm2VecModel():
    def __init__(self, output_path, seed, max_length, nprocs) -> None:
        self.savepath = output_path
        self.model = None
        self.nprocs = nprocs
        self.embedding_size = None
        self.seed = seed
        self.max_length = max_length
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)

    def split_functions(self, functions, n):
        chunk_size = len(functions) // n
        parts = [functions[i * chunk_size:(i + 1) * chunk_size]
                 for i in range(n)]
        if len(functions) % n != 0:
            parts[-1].extend(functions[chunk_size * n:])

        return parts

    def train(self, functions, widow_size, n_epochs, **kwargs):
        n_jobs = self.nprocs
        if self.savepath.exists():
            print(f"File already exists: {self.savepath}")

        print(f"Training Asm2Vec model on {len(functions)} functions.")
        functions = self.split_functions(
            functions,
            n_jobs
        )

        print("Creating dataset")
        dataset = MemoryCorpus(ParallelCorpus(
            [Asm2VecCorpus(split) for split in functions],
            n_jobs=n_jobs
        ))

        options = dict(
            **kwargs,
            window=widow_size,
            epochs=n_epochs,
            min_count=1,
            sg=0,
            hs=0,
            sample=0,
            negative=5,
            seed=self.seed
        )

        callbacks = [EpochLogger()]
        model = Asm2Vec(dataset,
                        compute_loss=True,
                        callbacks=callbacks,
                        workers=n_jobs,
                        **options)

        print(f"Saving trained model to {str(self.savepath)}.")
        self.savepath.parents[0].mkdir(parents=True, exist_ok=True)
        model.save(str(self.savepath))
        self.model = model

        return True

    def get_model(self):
        if self.model is not None:
            return self.model
        if self.savepath.exists():
            self.model = Asm2Vec.load(str(self.savepath))
            return self.model
        raise Exception("Model not trained")

    def get_embedding_size(self):
        if self.embedding_size is None:
            self.embedding_size = self.get_model().vector_size * 2
        return self.embedding_size

    def pad_embedding(self, embedding, max_length, embedding_size):
        padded = np.zeros((max_length, embedding_size))
        trunc_len = min(len(embedding), max_length)
        padded[:trunc_len] = embedding[:trunc_len]
        return padded

    def get_embedding(self, function):
        ops = base64.b64decode(function["base64_encoded_ops"])
        low_pc = function["low_pc"]
        instructions = disasm_func(ops, low_pc)
        return encode(instructions, self.get_model(), verbose=False)[:self.max_length]

    def get_embeddings(self, functions):
        return [self.get_embedding(function) for function in functions]


class FunctionCorpus(Corpus):
    def __init__(self, functions):
        self.functions = functions

    def _iter_functions(self):
        try:
            for function in tqdm(self.functions, leave=False, desc="Disasembling functions"):
                yield disasm_func(ops=base64.b64decode(function["base64_encoded_ops"]), low_pc=function["low_pc"])
        except:
            print(f"Error processing function: {str(function['id'])}")
            raise

    def __str__(self):
        return f"FunctionCorpus()"


class Asm2VecCorpus(FunctionCorpus):
    def __init__(self, functions):
        super().__init__(functions)
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
        return f"Asm2VecCorpus()"
