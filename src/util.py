from hashlib import md5


def md5_hash(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif isinstance(data, bytes):
        pass
    else:
        raise ValueError(f"Invalid data type: {type(data)}")

    return md5(data).hexdigest()


def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)
