import base64
import logging
import pathlib
import sys

import numpy as np
from gensim.models.callbacks import CallbackAny2Vec

logger = logging.getLogger(__name__)


class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training"""

    def __init__(self):
        self.epoch = 1
        self.last_loss = 0

    def on_epoch_begin(self, model):
        # logger.info(f"Epoch #{self.epoch} start.")
        print(f"Epoch #{self.epoch} start.", file=sys.stderr)

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        # logger.info(f"Epoch #{self.epoch} end. Loss: {loss - self.last_loss: .03f}.")
        print(f"Epoch #{self.epoch} end. Loss: {loss - self.last_loss: .03f}.", file=sys.stderr)
        self.last_loss = loss
        self.epoch += 1


def convert_model(model, savepath):
    wv = model.wv
    size = len(model.wv)
    logger.info("Convert Word2Vec model.")
    with open(savepath, 'wt') as f:
        for index, word in enumerate(wv.index_to_key, 1):
            logger.info(f'Processing "{word}" ({index}/{size})')
            print(f"\"{word}\" {' '.join(np.char.mod('%f', wv[word]))}", file=f)


def collect_packages(input):
    """
    Collects all debian packages (.deb) from a list of files and directories.
    Debian package files will be returned unchanged. Directories will be search recursively.
    Assume for all other files that each line contains a file path of a debian pacakge.
    Args:
        input: A list of pathlib.Path objects

    Returns:
        A generator yielding the collected debian packages.
    """
    for pkg in input:
        if pkg.is_dir():
            yield from pkg.rglob('*.deb')
        elif pkg.suffix == '.deb':
            yield pkg
        else:
            for line in pkg.read_text().splitlines():
                yield pathlib.Path(line)


def pkg2cachefile(pkg):
    pkg = pathlib.Path(pkg)
    pkg = pkg.resolve(strict=True)

    return base64.b32encode(bytes(pkg)).decode()
