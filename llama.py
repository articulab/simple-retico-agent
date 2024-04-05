import asyncio
import threading
import retico_core
import transformers
from retico_core.text import TextIU, SpeechRecognitionIU
from transformers import (
    BitsAndBytesConfig,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch


class Llama:

    # HF_TOKEN_FILE = "./hf_token.txt"

    def __init__(
        self,
        model_name,
        top_k=10,
        do_sample=True,
        num_return_sequences=1,
        max_length=400,
        device_map="auto",
        load_in_4bit=True,
        chat_history={},
        **kwargs
    ):
        self.model_name = model_name
        # self.access_token = "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"
        # with open(self.HF_TOKEN_FILE, "r") as f:
        #     self.access_token = f.readline()

        # self.access_token = "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"
        self.model_name = "mediocredev/open-llama-3b-v2-chat"
        self.chat_history = chat_history

        # inference linguistics args
        self.top_k = top_k
        self.do_sample = do_sample
        self.num_return_sequences = num_return_sequences
        self.max_length = max_length

        # inference hardware args
        self.device_map = device_map
        # self.load_in_4bit = load_in_4bit
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_compute_dtype=getattr(torch, "float16"),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # Model is not loaded for the moment
        self.model = None
        self.tokenizer = None
        self.pipeline = None

    def load_model(self):
        # load_in_4bit=True,
        # device_map="cuda",
        # self.model = LlamaForCausalLM.from_pretrained(
        #     self.model_name,
        #     token=self.access_token,
        #     device_map=self.device_map,
        #     load_in_4bit=self.load_in_4bit,
        # )
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.model_name, token=self.access_token
        # )
        # self.pipeline = transformers.pipeline(
        #     "text-generation",
        #     model=self.model,
        #     torch_dtype=torch.float16,
        #     device_map="auto",
        #     token=self.access_token,
        # )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            quantization_config=self.bnb_config,
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

    def generate_next_sentence(self):
        # print("chat_history\n")
        # print(chat_history)
        input_ids = self.tokenizer.apply_chat_template(
            self.chat_history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        output_tokens = self.model.generate(
            input_ids,
            repetition_penalty=1.05,
            max_new_tokens=1000,
        )
        output_text = self.tokenizer.decode(
            output_tokens[0][len(input_ids[0]) :], skip_special_tokens=True
        )
        self.chat_history.append({"role": "assistant", "content": output_text})
        return output_text


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
        # self.llama.load_model()
        pass

    def process_update(self, update_message):
        if not update_message:
            return None
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                continue
            elif ut == retico_core.UpdateType.REVOKE:
                # self.process_revoke(iu)
                continue
            elif (
                ut == retico_core.UpdateType.COMMIT
            ):  # we want to call process full sentence only if this is an end-of-sentence token
                # self.commit(iu)
                # process the full sentence, append the update message and then pass to stop processing the rest or ius ?? do we want this behavior ?
                self.process_full_sentence(update_message)
                pass

    def recreate_sentence_from_um(um):
        sentence = ""
        for iu, ut in um:
            sentence += iu.get_text() + " "
        print("sentence recreated = " + str(sentence))

    def process_full_sentence(self, um):
        sentence = self.recreate_sentence_from_um(um)

        async def async_generate(sentence):
            # result = await self.llama.generate(sentence)
            result = sentence
            self.append(result)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        coroutine = async_generate(sentence)
        loop.run_until_complete(coroutine)

    # def process_iu(self, iu):
    #     async def async_generate(async_iu):
    #         result = await self.llama.generate(async_iu)

    #     loop = asyncio.new_event_loop()
    #     asyncio.set_event_loop(loop)
    #     coroutine = async_generate(iu)
    #     loop.run_until_complete(coroutine)
