import os

import torch
import transformers
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from modules.lm_head import LMHeadModel
from modules.modeling_phi import PhiForCausalLM

os.environ["TQDM_DISABLE"] = "1"

# Code from https://github.com/state-spaces/mamba/tree/main


class BaseEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(
        self,
        pretrained=None,
        max_length=2048,
        batch_size=None,
        device=f"cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
    ):  # training is everything 32
        LM.__init__(self)
        # Parameters
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)
        self._dtype = dtype
        # Tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/phi-1_5")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, **kwargs):
        raise NotImplementedError()


@register_model("phi-mamba")
class PhiMambaEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        self._model = LMHeadModel.from_pretrained("goombalab/Phi-Mamba", strict=True)
        self._model.to(self._device).to(self._dtype).eval()


@register_model("hybrid-phi-mamba")
class PhiMambaEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        self._model = LMHeadModel.from_pretrained(
            "goombalab/Hybrid-Phi-Mamba",
            attn_type="flash_attention_2" if torch.is_autocast_enabled() else "eager",
            strict=True,
        )
        self._model.to(self._device).to(self._dtype).eval()


@register_model("phi")
class PhiEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        self._model = PhiForCausalLM.from_pretrained("microsoft/phi-1_5", strict=True)
        self._model.to(self._device).to(self._dtype).eval()


if __name__ == "__main__":
    cli_evaluate()
