import os
from pathlib import Path


from .parser import (
    parse_engine,
    create_session,
)
from ...architecture import NnTensorrtArchitecture

def is_model_generic(model: Path|str) -> bool:
    return True


MODEL_ARCHITECTURES: tuple[NnTensorrtArchitecture] = (
    NnTensorrtArchitecture(
        name='generic',
        detect=is_model_generic,
        parse=parse_engine,
        create_session=create_session,
        dtypes=('fp32', 'fp16', 'bf16'),
    ),
)
