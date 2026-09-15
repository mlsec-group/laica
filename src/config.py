E3_DATASET = "mlsec-group/laica-sp"

DATASETS = {
    "n_param": "mlsec-group/hf-laica",
    "param_type": "mlsec-group/hf-laica",
    "function_similarity": "mlsec-group/hf-laica",
    "compiler": "mlsec-group/hf-laica-compiler",
}

TASKS = sorted(DATASETS.keys())

MAX_NUM_PARAMS = 10

COMPILER_LABELS = {
    "clang": 0,
    "gcc": 1,
}

PARAM_TYPE_LABELS = {
    "pointer": 0,
    "struct": 1,
    "union": 2,
    "enum": 3,
    "float": 4,
    "char": 5,
    "int": 6,
}

PAIRS_OPT_LEVELS = ["O3"]

PAIRS_NEGATIVE_RATIO = 1

BASE_ADDRESS = 0x4000
