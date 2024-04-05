import asyncio
import threading
import retico_core
import torch
from llama_cpp import Llama


class LlamaCppMemory:

    def __init__(
        self,
        model_path,
        model_repo,
        model_name,
        # chat_history={},
        initial_prompt,
        context_size=512,
        short_memory_context_size=300,
        n_gpu_layers=100,
        **kwargs
    ):
        # Model loading method 1
        self.model_path = model_path
        # Model loading method 2
        self.model_repo = model_repo
        self.model_name = model_name

        # self.chat_history = chat_history
        self.initial_prompt = initial_prompt
        self.stop_token_ids = [2, 13]
        self.stop_token_text = [b"\n", b"        "]
        self.my_prompt = initial_prompt
        self.utterances = []
        self.size_per_utterance = []
        self.short_memory_context_size = short_memory_context_size

        # llamap-cpp-python args
        self.context_size = context_size
        self.n_gpu_layers = n_gpu_layers

        # Model is not loaded for the moment
        self.model = None

    def load_model(self):
        if self.model_path is not None:

            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_size,
                n_gpu_layers=self.n_gpu_layers,
            )

        elif self.model_repo is not None and self.model_name is not None:

            self.model = Llama.from_pretrained(
                repo_id=self.model_repo,
                filename=self.model_name,
                device_map="cuda",
                n_ctx=self.context_size,
                n_gpu_layers=self.n_gpu_layers,
            )

        else:
            raise NotImplementedError(
                "Please, when creating the module, you must give a model_path or model_repo and model_name"
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

    # def stop_criteria(tokens, logits):
    #     # OLD
    #     # is_eos = tokens[-1] in my_model.token_eos()
    #     # is_next_line = my_model.detokenize([tokens[-1]]) == b"\n"
    #     # is_tab = my_model.detokenize([tokens[-1]]) == b"      "
    #     # if is_eos or is_next_line or is_tab:
    #     #     print(tokens[-1])
    #     #     print(my_model.detokenize([tokens[-1]]))
    #     #     print(my_model.detokenize(tokens))
    #     # return is_eos or is_next_line or is_tab

    #     is_stopping_id = tokens[-1] in STOP_TOKEN_IDS
    #     is_stopping_text = my_model.detokenize([tokens[-1]]) in STOP_TOKEN_TEXT
    #     # if is_stopping_id or is_stopping_text:
    #     #     print(tokens[-1])
    #     #     print(my_model.detokenize([tokens[-1]]))
    #     #     print(my_model.detokenize(tokens))
    #     return is_stopping_id or is_stopping_text

    def generate_next_sentence(
        self,
        # max_tokens = 100,
        # temperature = 0.3,
        # top_p = 0.1,
        top_k=40,
        top_p=0.95,
        temp=1.0,
        repeat_penalty=1.1,
    ):

        def stop_criteria(tokens, logits):
            is_stopping_id = tokens[-1] in self.stop_token_ids
            is_stopping_text = (
                self.model.detokenize([tokens[-1]]) in self.stop_token_text
            )
            return is_stopping_id or is_stopping_text

        # Define the parameters
        tokens = self.model.tokenize(self.my_prompt)
        last_sentence = b""
        last_sentence_nb_tokens = 0
        for token in self.model.generate(
            tokens,
            stopping_criteria=stop_criteria,
            top_k=top_k,
            top_p=top_p,
            temp=temp,
            repeat_penalty=repeat_penalty,
        ):
            last_sentence += self.model.detokenize([token])
            last_sentence_nb_tokens += 1
        return last_sentence, last_sentence_nb_tokens

    # def add_user_sentence(self, sentence):
    #     self.chat_history.append({"role": "user", "content": sentence})


class LlamaCppMemoryModule(retico_core.AbstractModule):
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

    def __init__(self, model_path, model_repo, model_name, initial_prompt, **kwargs):
        """Initializes the LlamaCpp Module.

        Args:
            model_path (str): The path to the desired local model file (.gguf for example).
        """
        super().__init__(**kwargs)
        # Model loading method 1
        self.model_path = model_path
        # Model loading method 2
        self.model_repo = model_repo
        self.model_name = model_name
        self.model_wrapper = LlamaCppMemory(
            self.model_path, self.model_repo, self.model_name, initial_prompt, **kwargs
        )

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
        user_sentence = self.recreate_sentence_from_um(msg)
        print("user sentence : " + str(user_sentence))
        # self.model_wrapper.add_user_sentence(sentence)
        user_sentence_complete = (
            b"[INST]Child : " + bytes(user_sentence, "utf-8") + b"[/INST]"
        )

        # Add user sentence to short term memory
        # append last sentence from user as the last utterance
        user_sentence_complete_nb_tokens = len(
            self.model_wrapper.model.tokenize(user_sentence_complete)
        )
        self.utterances.append(user_sentence_complete)
        self.size_per_utterance.append(user_sentence_complete_nb_tokens)
        self.my_prompt += user_sentence_complete

        # Calculate short term memory
        nb_tokens = sum(self.size_per_utterance)
        print("num token : " + str(nb_tokens))
        if nb_tokens >= self.model_wrapper.short_memory_context_size:
            print("nb QA = " + str(len(self.utterances)))
            print("size_per_utterance = " + str(self.size_per_utterance))
            while (
                sum(self.size_per_utterance)
                >= self.model_wrapper.short_memory_context_size
            ):
                self.utterances.pop(
                    1
                )  # do not pop the system prompt explaining the scenario
                res = self.size_per_utterance.pop(1)
                print("POP " + str(res))
            self.my_prompt = b"".join(self.utterances)
            print("my prompt = \n" + self.my_prompt.decode("utf-8"))

        # Agent sentence
        agent_sentence, agent_sentence_mn_tokens = (
            self.model_wrapper.generate_next_sentence()
        )
        # Add model sentence to short term memory
        # add the last sentence from model to the last utterance which contains only the sentence from user
        self.utterances[-1] += agent_sentence
        self.size_per_utterance[-1] += agent_sentence_mn_tokens
        self.my_prompt += agent_sentence + b"\n"
        print("agent sentence : " + str(agent_sentence))

        # should trigger modules subscribed to this llama cpp module (for example TTS) :
        payload = agent_sentence
        output_ui = self.create_iu()
        output_ui.payload = payload
        next_um = retico_core.abstract.UpdateMessage()
        next_um.add_iu(output_ui, update_type=retico_core.UpdateType.ADD)
        next_um.add_iu(output_ui, update_type=retico_core.UpdateType.COMMIT)
        self.append(next_um)
