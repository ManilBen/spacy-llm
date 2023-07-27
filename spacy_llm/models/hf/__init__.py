from .base import HuggingFace
from .dolly import dolly_hf
from .falcon import falcon_hf
from .openllama import openllama_hf
from .stablelm import stablelm_hf
from .vicuna import vicuna_hf
from .vigogne import vigogne2_hf
from .llama2 import llama2_hf

__all__ = [
    "HuggingFace",
    "dolly_hf",
    "falcon_hf",
    "openllama_hf",
    "stablelm_hf",
    "vicuna_hf",
    "vigogne2_hf",
    "llama2_hf"
]
