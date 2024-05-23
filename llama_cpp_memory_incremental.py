import asyncio
import csv
import datetime
import threading
import time
import retico_core
import torch
from llama_cpp import Llama
import re

from utils import *


class LlamaCppMemoryIncremental:

    def __init__(
        self,
        model_path,
        model_repo,
        model_name,
        # chat_history={},
        initial_prompt=None,
        system_prompt=None,
        context_size=2000,
        short_memory_context_size=500,
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
        self.system_prompt = system_prompt
        self.prompt = None
        self.stop_token_ids = []
        self.stop_token_text = []
        self.stop_token_text_patterns = [b"Child:", b"Child :"]
        self.stop_token_patterns = []
        self.role_token_text_patterns = [
            b"Teacher:",
            b"Teacher :",
            b" Teacher:",
            b" Teacher :",
        ]
        self.role_token_patterns = []
        self.ponctuation_text = [b".", b",", b";", b":", b"!", b"?", b"..."]
        self.utterances = []
        self.size_per_utterance = []
        self.short_memory_context_size = short_memory_context_size

        # template attributes
        # template 1
        # prompt = "[INST] <<SYS>>\
        # This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        # The teacher is teaching mathemathics to the child student. \
        # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
        # You play the role of a teacher. Here is the beginning of the conversation : \
        # <</SYS>>\
        # \
        # Child : Hello ! \
        # Teacher : Hi! How are your today ? \
        # Child : I am fine, and I can't wait to learn mathematics ! \
        # [/INST]"
        self.start_prompt = b"[INST] "
        self.end_prompt = b" [/INST]"
        self.nb_tokens_end_prompt = 0
        self.sys_pre = b"<<SYS>>"
        self.sys_suf = b"<</SYS>>"
        self.user_pre = b""
        self.user_suf = b"\n\n"
        self.agent_pre = b""
        self.agent_suf = b""
        self.user_role = b"Child"
        self.agent_role = b"Teacher"

        # llamap-cpp-python args
        self.context_size = context_size
        self.n_gpu_layers = n_gpu_layers

        # Model is not loaded for the moment
        self.model = None

    def setup(self):
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

        self.initialize_prompt()
        self.init_stop_criteria()

    def init_stop_criteria(self):
        self.stop_token_ids.append(self.model.token_eos)
        for pat in self.stop_token_text_patterns:
            self.stop_token_patterns.append(self.model.tokenize(pat, add_bos=False))
        for pat in self.role_token_text_patterns:
            self.role_token_patterns.append(self.model.tokenize(pat, add_bos=False))

    def initialize_prompt(self):
        self.nb_tokens_end_prompt = len(self.model.tokenize(self.end_prompt))
        if self.initial_prompt is None:
            if self.system_prompt is None:
                pass
            else:
                complete_system_prompt = (
                    self.start_prompt + self.sys_pre + self.system_prompt + self.sys_suf
                )
                self.prompt = complete_system_prompt
                self.utterances = [complete_system_prompt]
                self.size_per_utterance = [
                    len(self.model.tokenize(complete_system_prompt))
                ]
        else:
            raise NotImplementedError(
                "for now, only support starting from system prompt"
            )

    def new_user_sentence(self, user_sentence):
        user_sentence_complete = (
            self.user_pre
            + self.user_role
            + b" : "
            + bytes(user_sentence, "utf-8")
            + self.user_suf
        )
        user_sentence_complete_nb_tokens = len(
            self.model.tokenize(user_sentence_complete)
        )
        self.utterances.append(user_sentence_complete)
        self.size_per_utterance.append(user_sentence_complete_nb_tokens)
        self.prompt += user_sentence_complete
        print(self.user_role.decode("utf-8") + " : " + user_sentence)

    def new_agent_sentence(self, agent_sentence, agent_sentence_nb_tokens):
        agent_sentence_reduced, nb_token_removed = self.remove_stop_patterns(
            agent_sentence
        )
        agent_sentence_complete = (
            self.agent_pre + agent_sentence_reduced + self.agent_suf
        )
        nb_token_added = len(self.model.tokenize(self.agent_pre, add_bos=False)) + len(
            self.model.tokenize(self.agent_suf, add_bos=False)
        )
        agent_sentence_complete_nb_tokens = (
            agent_sentence_nb_tokens + nb_token_added - nb_token_removed
        )
        self.utterances.append(agent_sentence_complete)
        self.size_per_utterance.append(agent_sentence_complete_nb_tokens)
        self.prompt += agent_sentence_complete
        # not adding the agent role because it is already generated by the model.
        # TODO : change this by placing the role directly on the prompt?
        print(agent_sentence_complete.decode("utf-8"))

    def remove_stop_patterns(self, sentence):
        last_chunck_string = sentence
        nb_token_removed = 0
        for i, pat in enumerate(self.stop_token_text_patterns):
            if pat == last_chunck_string[-len(pat) :]:
                sentence = sentence[: -len(pat)]
                nb_token_removed = len(self.stop_token_patterns[i])
                break
        # while sentence[-1:] == b"\n":
        #     sentence = sentence[:-1]
        #     nb_token_removed += 1
        return sentence, nb_token_removed

    def remove_role_patterns(self, sentence):
        first_chunck_string = sentence
        nb_token_removed = 0
        for i, pat in enumerate(self.role_token_text_patterns):
            if pat == first_chunck_string[: -len(pat)]:
                sentence = sentence[-len(pat) :]
                nb_token_removed = len(self.role_token_patterns[i])
                break
        # while sentence[-1:] == b"\n":
        #     sentence = sentence[:-1]
        #     nb_token_removed += 1
        return sentence, nb_token_removed

    def prepare_prompt_memory(self):
        # Calculate short term memory
        nb_tokens = sum(self.size_per_utterance)
        if nb_tokens + self.nb_tokens_end_prompt >= self.short_memory_context_size:
            nb_tokens_removed = 0
            nb_tokens_to_remove = (
                nb_tokens + self.nb_tokens_end_prompt - self.short_memory_context_size
            )
            while nb_tokens_to_remove >= nb_tokens_removed:
                self.utterances.pop(
                    1
                )  # pop oldest non system utterance. do not pop the system prompt explaining the scenario
                nb_tokens_removed += self.size_per_utterance.pop(1)
            self.prompt = b"".join(self.utterances)

    def is_ponctuation(self, token):
        # return True if the token corresponds to a ponctuation
        is_ponctuation = self.model.detokenize([token]) in self.ponctuation_text
        return is_ponctuation

    def is_stop_pattern(self, sentence):
        # return True, and the stop pattern if last n characters of the sentence is a stop pattern.
        for i, pat in enumerate(self.stop_token_text_patterns):
            if pat == sentence[-len(pat) :]:
                return True, self.stop_token_patterns[i]
        return False, None

    def is_role_pattern(self, sentence):
        max_pattern_size = max([len(p) for p in self.role_token_text_patterns])
        if max_pattern_size < len(
            sentence
        ):  # We want to only check at the very beginning of the sentence
            return False, None
        # return True, and the stop pattern if last n characters of the sentence is a stop pattern.
        for i, pat in enumerate(self.role_token_text_patterns):
            if pat == sentence[-len(pat) :]:
                return True, self.role_token_patterns[i]
        return False, None

    def process_full_sentence(self, user_sentence, subprocess):
        self.new_user_sentence(user_sentence)
        self.prepare_prompt_memory()
        agent_sentence, agent_sentence_nb_tokens = self.generate_next_sentence(
            subprocess
        )
        self.new_agent_sentence(agent_sentence, agent_sentence_nb_tokens)

    def generate_next_sentence(
        self, subprocess, top_k=40, top_p=0.95, temp=1.0, repeat_penalty=1.1
    ):

        def stop_criteria(tokens, logits):
            """
            Function used by the LLM to stop generate tokens when it meets certain criteria.

            Args:
                tokens (_type_): tokens generated by the LLM
                logits (_type_): _description_

            Returns:
                bool: returns True if it the last generated token corresponds to self.stop_token_ids or self.stop_token_text.
            """
            is_stopping_id = tokens[-1] in self.stop_token_ids
            is_stopping_text = (
                self.model.detokenize([tokens[-1]]) in self.stop_token_text
            )
            return is_stopping_id or is_stopping_text

        def stop_multiple_utterances_generation(tokens, logits):
            """
            Function used by the LLM to stop generate tokens when it meets certain criteria.\
            This function stops the generation if a particular pattern of token is generated by the model.\
            It is used to stop the generation if the model starts generating multiple dialogue utterances (starting with "Child :").

            Args:
                tokens (_type_): tokens generated by the LLM
                logits (_type_): _description_

            Returns:
                bool: returns True if it the last generated token corresponds to self.stop_token_ids or self.stop_token_text, \
                Or if the last generated token match one of the self.stop_token_patterns
            """

            is_stopping_id = tokens[-1] in self.stop_token_ids
            is_stopping_text = (
                self.model.detokenize([tokens[-1]]) in self.stop_token_text
            )

            # is_stopping_pattern = False
            # max_pattern_size = max([len(p) for p in self.stop_token_patterns])
            # last_chunck_string = self.model.detokenize(tokens[-max_pattern_size:])
            # for i, pat in enumerate(self.stop_token_text_patterns):
            #     if pat == last_chunck_string[-len(pat):]:
            #         return True

            max_pattern_size = max([len(p) for p in self.stop_token_patterns])
            last_chunck_string = self.model.detokenize(tokens[-max_pattern_size:])
            is_stopping_pattern, _ = self.is_stop_pattern(last_chunck_string)

            return is_stopping_id or is_stopping_text or is_stopping_pattern

        # Define the parameters
        final_prompt = self.prompt + self.end_prompt
        # print("final_prompt = ", final_prompt)
        tokens = self.model.tokenize(final_prompt, special=True)
        last_sentence = b""
        last_sentence_nb_tokens = 0
        for token in self.model.generate(
            tokens,
            # stopping_criteria=stop_criteria,
            stopping_criteria=stop_multiple_utterances_generation,
            top_k=top_k,
            top_p=top_p,
            temp=temp,
            repeat_penalty=repeat_penalty,
        ):
            # Update module IUS
            payload = self.model.detokenize([token])
            payload_text = payload.decode("utf-8")
            # TODO : Here we have to remove the "Teacher :" from the agent utterance to not make it spoken by the TTS

            # Update model short term memory
            last_sentence += payload
            last_sentence_nb_tokens += 1

            # call retico module subprocess to send update message to subscribed modules.
            is_ponct = self.is_ponctuation(token)
            _, stop_pattern = self.is_stop_pattern(last_sentence)
            _, role_pattern = self.is_role_pattern(last_sentence)
            subprocess(payload_text, is_ponct, stop_pattern, role_pattern)

        return last_sentence, last_sentence_nb_tokens


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

    def __init__(
        self,
        model_path,
        model_repo,
        model_name,
        initial_prompt,
        system_prompt,
        printing=False,
        log_file="llm.csv",
        log_folder="logs/test/16k/Recording (1)/demo",
        **kwargs
    ):
        """Initializes the LlamaCpp Module.

        Args:
            model_path (str): The path to the desired local model file (.gguf for example).
        """
        super().__init__(**kwargs)
        self.printing = printing
        # logs
        self.log_file = manage_log_folder(log_folder, log_file)
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
            system_prompt,
            **kwargs
        )
        self.latest_input_iu = None

    def setup(self):
        self.model_wrapper.setup()

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
            self.process_incremental(msg)

    def recreate_sentence_from_um(self, msg):
        sentence = ""
        for iu in msg:
            sentence += iu.get_text() + " "
        return sentence

    def process_incremental(self, msg):
        if self.printing:
            print(
                "LLM : process sentence ",
                datetime.datetime.now().strftime("%T.%f")[:-3],
            )

        write_logs(
            self.log_file,
            [["Start", datetime.datetime.now().strftime("%T.%f")[:-3]]],
        )

        def subprocess(
            payload, is_ponctuation=None, stop_pattern=None, role_pattern=None
        ):
            """
            This function will be called by the LLM wrapper at each token generation.
            It handles the communication with the subscribed module (TTS for example),
            by updating and publishing new UpdateMessage containing the new IUS.
            IUs are :
                - ADDED in every situation (the generated words are sent to the subscribed modules)
                - COMMITED if the last token generated is a ponctuation (The TTS can start generating the voice corresponding to the clause)
                - REVOKED if the last tokens generated corresponds to a stop pattern (so that the subscribed module delete the stop pattern)


            Args:
                payload (string): the text corresponding to the last generated token
                is_ponctuation (bool, optional): True if the last generated token correspond to a ponctuation. Defaults to None.
                stop_pattern (string, optional): Text corresponding to the generated stop_pattern. Defaults to None.
            """
            # Construct UM and IU
            next_um = retico_core.abstract.UpdateMessage()
            output_iu = self.create_iu(self.latest_input_iu)
            output_iu.set_text(payload)
            if not self.latest_input_iu:
                self.latest_input_iu = output_iu
            self.current_output.append(output_iu)

            # ADD
            next_um.add_iu(output_iu, retico_core.UpdateType.ADD)

            # REVOKE if stop patterns
            if stop_pattern is not None:
                for id, token in enumerate(
                    stop_pattern
                ):  # take all IUs corresponding to stop pattern
                    iu = self.current_output.pop(
                        -1
                    )  # the IUs corresponding to the stop pattern are the last n ones where n=len(stop_pattern).
                    self.revoke(iu)
                    next_um.add_iu(iu, retico_core.UpdateType.REVOKE)
                # print("REVOKE :\n".join([iu.payload for iu in self.current_output]))

            # REVOKE if role patterns
            if role_pattern is not None:
                for id, token in enumerate(
                    role_pattern
                ):  # take all IUs corresponding to stop pattern
                    iu = self.current_output.pop(
                        -1
                    )  # the IUs corresponding to the stop pattern are the last n ones where n=len(stop_pattern).
                    self.revoke(iu)
                    next_um.add_iu(iu, retico_core.UpdateType.REVOKE)
                # print("REVOKE :\n".join([iu.payload for iu in self.current_output]))

            # COMMIT if ponctuation and not role patterns
            if (
                is_ponctuation and role_pattern is None and stop_pattern is None
            ):  # this works because role patterns end with a ponctuation
                for iu in self.current_output:
                    self.commit(iu)
                    next_um.add_iu(iu, retico_core.UpdateType.COMMIT)
                if self.printing:
                    print(
                        "LLM : send sentence after ponct ",
                        datetime.datetime.now().strftime("%T.%f")[:-3],
                    )
                write_logs(
                    self.log_file,
                    [["Stop", datetime.datetime.now().strftime("%T.%f")[:-3]]],
                )
                # print("COMMIT :\n"+"".join([iu.payload for iu in self.current_output]))
                self.current_output = []
            self.append(next_um)

        self.model_wrapper.process_full_sentence(
            self.recreate_sentence_from_um(msg), subprocess
        )
        # reset because it is end of sentence
        self.current_output = []
        self.latest_input_iu = None
