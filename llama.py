import asyncio
import threading
import retico_core
import transformers
from retico_core.text import TextIU, SpeechRecognitionIU
from transformers import (
    LlamaForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch


class Llama:

    HF_TOKEN_FILE = "./hf_token.txt"

    def __init__(
        self,
        model_name,
        top_k=10,
        do_sample=True,
        num_return_sequences=1,
        max_length=400,
        **kwargs
    ):
        self.model_name = model_name
        # self.access_token = "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"
        with open(self.HF_TOKEN_FILE, "r") as f:
            self.access_token = f.readline()

        self.top_k = top_k
        self.do_sample = do_sample
        self.num_return_sequences = num_return_sequences
        self.max_length = max_length

        # Model is not loaded for the moment
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def load_model(self):
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_name, token=self.access_token
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, token=self.access_token
        )

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            torch_dtype=torch.float16,
            device_map="auto",
            token=self.access_token,
        )

    def generate(self, prompt):
        sequences = self.pipeline(
            prompt,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=self.do_sample,
            top_k=self.top_k,
            num_return_sequences=self.num_return_sequences,
            max_length=self.max_length,
        )
        return sequences


class LlamaModule(retico_core.AbstractModule):
    """An implementation of Llama-2 with retico"""

    @staticmethod
    def name():
        return "Llama-2 Module"

    @staticmethod
    def description():
        return "A Module providing Natural Language Generation by llama-2"

    @staticmethod
    def input_ius():
        return [SpeechRecognitionIU]

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(self, model_name, **kwargs):
        """Initializes the LlamaModule.

        Args:
            model_name (str): The name of the model that will be loaded.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.llama = Llama(self.model_name)

    def setup(self):
        # We create the model
        self.llama.load_model()

    def process_update(self, update_message):
        if not update_message:
            return None
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.process_ui(iu)
            elif ut == retico_core.UpdateType.REVOKE:
                self.process_revoke(iu)
            # elif ut == retico_core.UpdateType.COMMIT:
            #     self.commit(iu)

    def process_iu(self, iu):
        async def async_generate(async_iu):
            result = await self.llama.generate(async_iu)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        coroutine = async_generate(iu)
        loop.run_until_complete(coroutine)
