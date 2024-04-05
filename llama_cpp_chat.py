import asyncio
import threading
import retico_core
import torch
from llama_cpp import Llama


class LlamaCpp:

    def __init__(
        self, model_path, chat_history={}, context_size=512, n_gpu_layers=100, **kwargs
    ):
        self.model_path = model_path
        self.chat_history = chat_history

        # llamap-cpp-python args
        self.context_size = context_size
        self.n_gpu_layers = n_gpu_layers

        # Model is not loaded for the moment
        self.model = None

    def load_model(self):
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=self.context_size,
            n_gpu_layers=self.n_gpu_layers,
        )

    def generate_next_sentence(
        self,
        max_tokens=100,
        temperature=0.3,
        top_p=0.1,
        stop=["Q", "\n"],
    ):

        # Define the parameters
        model_output = self.model.create_chat_completion(
            self.chat_history,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        role = model_output["choices"][0]["message"]["role"]
        content = model_output["choices"][0]["message"]["content"]
        assert role == "assistant"
        self.chat_history.append({"role": role, "content": content})
        return content

    def add_user_sentence(self, sentence):
        self.chat_history.append({"role": "user", "content": sentence})


class LlamaCppModule(retico_core.AbstractModule):
    """An implementation of a conversational LLM (Llama-2 type) using a chat template with retico"""

    @staticmethod
    def name():
        return "LlamaCpp Module"

    @staticmethod
    def description():
        return "A Module providing Natural Language Generation using the llama-cpp-python library"

    @staticmethod
    def input_ius():
        return [retico_core.text.SpeechRecognitionIU]

    @staticmethod
    def output_iu():
        return retico_core.text.TextIU

    def __init__(self, model_path, chat_history, **kwargs):
        """Initializes the LlamaCpp Module.

        Args:
            model_path (str): The path to the desired local model file (.gguf for example).
        """
        super().__init__(**kwargs)
        self.model_path = model_path
        self.model_wrapper = LlamaCpp(self.model_path, chat_history=chat_history)

    def setup(self):
        self.model_wrapper.load_model()

    def process_update(self, update_message):
        if not update_message:
            return None
        commit = False
        msg = []
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                continue
            elif ut == retico_core.UpdateType.REVOKE:
                continue
            elif ut == retico_core.UpdateType.COMMIT:
                msg.append(iu)
                commit = True
                pass
        if commit:
            self.process_full_sentence(msg)

    def recreate_sentence_from_um(self, msg):
        sentence = ""
        for iu in msg:
            sentence += iu.get_text() + " "
        return sentence

    def process_full_sentence(self, msg):
        sentence = self.recreate_sentence_from_um(msg)
        print("user sentence : " + str(sentence))
        self.model_wrapper.add_user_sentence(sentence)
        agent_sentence = self.model_wrapper.generate_next_sentence()
        print("agent sentence : " + str(agent_sentence))
        # should trigger modules subscribed to this llama cpp module (for example TTS) :
        payload = agent_sentence
        output_ui = self.create_iu()
        output_ui.payload = payload
        next_um = retico_core.abstract.UpdateMessage()
        next_um.add_iu(output_ui, update_type=retico_core.UpdateType.ADD)
        next_um.add_iu(output_ui, update_type=retico_core.UpdateType.COMMIT)
        self.append(next_um)
