from typing import Any, Callable, Dict, Iterable, Optional, Tuple
from spacy.util import SimpleFrozenDict
import ctranslate2
from ...compat import Literal, torch, transformers
from ...registry.util import registry
from .base import HuggingFace


class FastLlama2(HuggingFace):
    MODEL_NAMES = Literal[
        "llama-2-7b-ct2"
    ]

    def __init__(
        self,
        name: str,
        config_init: Optional[Dict[str, Any]],
        config_run: Optional[Dict[str, Any]],
    ):
        self._tokenizer: Optional["transformers.AutoTokenizer"] = None
        self._device: Optional[str] = None
        super().__init__(name=name, config_init=config_init, config_run=config_run)

    def init_model(self) -> "transformers.AutoModelForCausalLM":
        """Sets up HF model and needed utilities.
        RETURNS (Any): HF model.
        """
        # Initialize tokenizer and model.
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self._name)
        init_cfg = self._config_init
        if "device" in init_cfg:
            #self._device = init_cfg.pop("device")
            model = ctranslate2.Generator(
                self._name, device="cuda"
            )
        else:
            model = ctranslate2.Generator(
                self._name
            )

        return model

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:  # type: ignore[override]
        assert callable(self._tokenizer)
        tokenized_input_ids = [
            self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(prompt)) for prompt in prompts
        ]
        if self._device:
            tokenized_input_ids = [tii.to(self._device) for tii in tokenized_input_ids]
        
        assert hasattr(self._model, "generate_batch")
        results = self._model.generate_batch(tokenized_input_ids, sampling_topk=10, max_length=128, include_prompt_in_result=False)
        return [
            self._tokenizer.decode(res.sequences_ids[0]) 
                for res in results
        ]

    @property
    def hf_account(self) -> str:
        return "/home/intern_jupyter/manil"

    @staticmethod
    def compile_default_configs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        default_cfg_init, default_cfg_run = HuggingFace.compile_default_configs()
        return (
            {
                **default_cfg_init,
                "torch_dtype": torch.float16,
            },
            {**default_cfg_run, "max_new_tokens": 256},
        )


@registry.llm_models("spacy.FastLlama2.v1")
def fastllama2_hf(
    name: FastLlama2.MODEL_NAMES,
    config_init: Optional[Dict[str, Any]] = SimpleFrozenDict(),
    config_run: Optional[Dict[str, Any]] = SimpleFrozenDict(),
) -> Callable[[Iterable[str]], Iterable[str]]:
    """Generates Vicuna instance that can execute a set of prompts and return the raw responses.
    name (Literal): Name of the Vicuna model. Has to be one of Vicuna.get_model_names().
    config_init (Optional[Dict[str, Any]]): HF config for initializing the model.
    config_run (Optional[Dict[str, Any]]): HF config for running the model.
    RETURNS (Callable[[Iterable[str]], Iterable[str]]): Vicuna instance that can execute a set of prompts and return
        the raw responses.
    """
    return FastLlama2(name=name, config_init=config_init, config_run=config_run)
