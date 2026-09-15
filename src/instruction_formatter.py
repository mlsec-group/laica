import base64

from bcemb.asm2vec.utils import format as asm2vec_format


def base64_encode_instructions(ops):
    return base64.b64encode(b"".join([i.bytes for i in ops])).decode("utf-8")


def format_instructions(ops):
    return "|".join(map(lambda x: "/".join(asm2vec_format(x)), ops))
