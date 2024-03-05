import asyncio
import threading
import retico_core
import torch
from llama_cpp import Llama
import re


class LlamaCppMemoryIncremental:

    def __init__(
        self,
        model_path,
        model_repo,
        model_name,
        # chat_history={},
        initial_prompt,
        context_size = 512,
        short_memory_context_size = 300,
        n_gpu_layers = 100,
        **kwargs
    ):
        # Model loading method 1
        self.model_path = model_path
        # Model loading method 2
        self.model_repo = model_repo
        self.model_name = model_name

        # self.chat_history = chat_history
        self.initial_prompt = initial_prompt
        self.stop_token_ids = [2,13]
        self.stop_token_text = [b"\n", b"        "]
        self.ponctuation_text = [b".", b",", b";", b":", b"!", b"?", b"..."]
        self.my_prompt = initial_prompt
        self.utterances = []
        self.size_per_utterance = []
        self.short_memory_context_size=short_memory_context_size

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
                n_gpu_layers=self.n_gpu_layers)
            
        elif self.model_repo is not None and self.model_name is not None :

            self.model = Llama.from_pretrained(
                repo_id=self.model_repo, 
                filename=self.model_name,
                device_map="cuda",
                n_ctx=self.context_size,
                n_gpu_layers=self.n_gpu_layers)
        
        else : 
            raise NotImplementedError("Please, when creating the module, you must give a model_path or model_repo and model_name")

        self.initialize_prompt()
        
    def initialize_prompt(self):
        DELIMITER = b"[INST]"
        # Initialize short term memory from first prompt
        split_utterances = self.my_prompt.split(DELIMITER)
        # print("nb QA = "+str(len(split_utterances)))
        self.utterances = [split_utterances[0]]
        self.size_per_utterance = []
        for u in split_utterances[1:]:
            self.utterances.append(DELIMITER+u)
            self.size_per_utterance.append(len(self.model.tokenize(self.utterances[-1])))
        print("my prompt = \n"+self.my_prompt.decode("utf-8"))
        self.print_prompt()

    def print_prompt(self):
        # ["[INST]", "[/INST]"]
        delimiters = ["[INST]", "[/INST]"]
        regex_pattern = '|'.join(map(re.escape, delimiters))
        splited = re.split(regex_pattern, self.my_prompt.decode("utf-8"))
        for s in splited:
            print(s+"\n")

    def process_full_sentence(self, user_sentence, subprocess):
        print("user sentence : "+str(user_sentence))
        user_sentence_complete =  b"[INST]Child : " + bytes(user_sentence, 'utf-8') + b"[/INST]     Teacher: "
        
        # Add user sentence to short term memory
        # append last sentence from user as the last utterance
        user_sentence_complete_nb_tokens = len(self.model.tokenize(user_sentence_complete))
        self.utterances.append(user_sentence_complete)
        self.size_per_utterance.append(user_sentence_complete_nb_tokens)
        self.my_prompt += user_sentence_complete

        # Calculate short term memory
        nb_tokens = sum(self.size_per_utterance)
        # print("num token : "+str(nb_tokens))
        if nb_tokens >= self.short_memory_context_size :
            # print("nb QA = "+str(len(self.utterances)))
            # print("size_per_utterance = "+str(self.size_per_utterance))
            while sum(self.size_per_utterance) >= self.short_memory_context_size:
                self.utterances.pop(1) # do not pop the system prompt explaining the scenario
                res = self.size_per_utterance.pop(1)
                # print("POP "+str(res))
            self.my_prompt = b"".join(self.utterances)
            # print("my prompt = \n"+self.my_prompt.decode("utf-8"))

        # # Agent sentence
        agent_sentence, agent_sentence_mn_tokens = self.generate_next_sentence(subprocess)
        # Add model sentence to short term memory
        # add the last sentence from model to the last utterance which contains only the sentence from user
        self.utterances[-1] += agent_sentence
        self.size_per_utterance[-1] += agent_sentence_mn_tokens
        self.my_prompt += agent_sentence + b"\n"
        print("agent sentence : "+agent_sentence.decode('utf-8'))

    def generate_next_sentence(
        self,
        subprocess,
        top_k=40,
        top_p=0.95,
        temp=1.0,
        repeat_penalty=1.1
        ):

        def stop_criteria(tokens, logits):
            is_stopping_id = tokens[-1] in self.stop_token_ids
            is_stopping_text = self.model.detokenize([tokens[-1]]) in self.stop_token_text
            return is_stopping_id or is_stopping_text

        def is_ponctuation(token):
            is_ponctuation = self.model.detokenize([token]) in self.ponctuation_text
            return is_ponctuation

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
            repeat_penalty=repeat_penalty
            ):
            # Update module IUS
            payload = self.model.detokenize([token])
            payload_text = payload.decode("utf-8")
            # remove_name = payload_text.split("Teacher :")
            is_ponct = is_ponctuation(token)
            subprocess(payload_text, is_ponct)
            # Update model short term memory
            last_sentence += payload
            last_sentence_nb_tokens += 1

        return last_sentence, last_sentence_nb_tokens
    

    # def add_user_sentence(self, sentence):
    #     self.chat_history.append({"role": "user", "content": sentence})


class LlamaCppMemoryIncrementalModule(retico_core.AbstractModule):
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
        self.model_wrapper = LlamaCppMemoryIncremental(
            self.model_path,
            self.model_repo,
            self.model_name,
            initial_prompt,
            **kwargs
        )
        self.latest_input_iu = None

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
            elif (
                ut == retico_core.UpdateType.COMMIT
            ):
                msg.append(iu)
                commit = True
                pass
        if commit:
            self.process_incremental(msg)

    def recreate_sentence_from_um(self, msg):
        sentence = ""
        for iu in msg:
            sentence += iu.get_text() + " "
        return sentence

    # def process_full_sentence(self, msg):
    #     user_sentence = self.recreate_sentence_from_um(msg)
    #     print("user sentence : "+str(user_sentence))
    #     # self.model_wrapper.add_user_sentence(sentence)
    #     user_sentence_complete =  b"[INST]Child : " + user_sentence + b"[/INST]"
        
    #     # Add user sentence to short term memory
    #     # append last sentence from user as the last utterance
    #     user_sentence_complete_nb_tokens = len(self.my_model.tokenize(user_sentence_complete))
    #     self.utterances.append(user_sentence_complete)
    #     self.size_per_utterance.append(user_sentence_complete_nb_tokens)
    #     self.my_prompt += user_sentence_complete

    #     # Calculate short term memory
    #     nb_tokens = sum(self.size_per_utterance)
    #     print("num token : "+str(nb_tokens))
    #     if nb_tokens >= self.model_wrapper.short_memory_context_size :
    #         print("nb QA = "+str(len(self.utterances)))
    #         print("size_per_utterance = "+str(self.size_per_utterance))
    #         while sum(self.size_per_utterance) >= self.model_wrapper.short_memory_context_size:
    #             self.utterances.pop(1) # do not pop the system prompt explaining the scenario
    #             res = self.size_per_utterance.pop(1)
    #             print("POP "+str(res))
    #         self.my_prompt = b"".join(self.utterances)
    #         print("my prompt = \n"+self.my_prompt.decode("utf-8"))

    #     # # Agent sentence
    #     # agent_sentence, agent_sentence_mn_tokens = self.model_wrapper.generate_next_sentence()
    #     # # Add model sentence to short term memory
    #     # # add the last sentence from model to the last utterance which contains only the sentence from user
    #     # self.utterances[-1] += agent_sentence
    #     # self.size_per_utterance[-1] += agent_sentence_mn_tokens
    #     # self.my_prompt += agent_sentence + b"\n"
    #     # print("agent sentence : "+str(agent_sentence))

    #     # # should trigger modules subscribed to this llama cpp module (for example TTS) :
    #     # payload = agent_sentence
    #     # output_ui = self.create_iu()
    #     # output_ui.payload = payload
    #     # next_um = retico_core.abstract.UpdateMessage()
    #     # next_um.add_iu(output_ui, update_type=retico_core.UpdateType.ADD)
    #     # next_um.add_iu(output_ui, update_type=retico_core.UpdateType.COMMIT)
    #     # self.append(next_um)
            
    #     self.process_incremental()
        

    def process_not_incremental(self, msg): 
        # this function is not incremental as it waits for the end of the generated 
        # sentence to append the next update message
        next_um = retico_core.abstract.UpdateMessage()

        def subprocess(payload):
            output_iu = self.create_iu(self.latest_input_iu)
            output_iu.set_text(payload)
            if not self.latest_input_iu:
                self.latest_input_iu = output_iu
            self.current_output.append(output_iu)
            next_um.add_iu(output_iu, retico_core.UpdateType.ADD)
        
        self.model_wrapper.process_full_sentence(self.recreate_sentence_from_um(msg), subprocess)
        # COMMIT all current output IUs because it is the end of sentence
        for iu in self.current_output:
            self.commit(iu)
            next_um.add_iu(iu, retico_core.UpdateType.COMMIT)
        self.current_output = []
        self.latest_input_iu = None
        self.append(next_um)
    

    # def process_incremental(self, msg):        

    #     def subprocess(payload):
    #         next_um = retico_core.abstract.UpdateMessage()
    #         output_iu = self.create_iu(self.latest_input_iu)
    #         output_iu.set_text(payload)
    #         if not self.latest_input_iu:
    #             self.latest_input_iu = output_iu
    #         self.current_output.append(output_iu)
    #         next_um.add_iu(output_iu, retico_core.UpdateType.ADD)
    #         self.append(next_um)
    #         print("ADD")

    #     self.model_wrapper.process_full_sentence(self.recreate_sentence_from_um(msg), subprocess)

    #     # COMMIT all current output IUs because it is the end of sentence
    #     next_um = retico_core.abstract.UpdateMessage()
    #     for iu in self.current_output:
    #         self.commit(iu)
    #         next_um.add_iu(iu, retico_core.UpdateType.COMMIT)
    #     self.current_output = []
    #     self.latest_input_iu = None
    #     self.append(next_um)
    #     print("COMMIT")

    def process_incremental(self, msg):        

        def subprocess(payload, ponctuation=None):
            next_um = retico_core.abstract.UpdateMessage()
            output_iu = self.create_iu(self.latest_input_iu)
            output_iu.set_text(payload)
            if not self.latest_input_iu:
                self.latest_input_iu = output_iu
            self.current_output.append(output_iu)
            next_um.add_iu(output_iu, retico_core.UpdateType.ADD)

            # this works
            # if len(self.current_output) >= 5 :
            #     for iu in self.current_output:
            #         self.commit(iu)
            #         next_um.add_iu(iu, retico_core.UpdateType.COMMIT)
            #     self.current_output = []

            # this works
            if ponctuation:
                for iu in self.current_output:
                    self.commit(iu)
                    next_um.add_iu(iu, retico_core.UpdateType.COMMIT)
                self.current_output = []
            self.append(next_um)

        self.model_wrapper.process_full_sentence(self.recreate_sentence_from_um(msg), subprocess)
        # reset because it is end of sentence
        self.current_output = []
        self.latest_input_iu = None