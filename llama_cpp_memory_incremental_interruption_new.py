"""
LlamaCppMemoryIncrementalModule
==================

A retico module that provides Natural Language Generation (NLG) using a conversational LLM (Llama-2 type).
The LlamaCppMemoryIncrementalModule class handles the aspects related to retico architecture : messaging (update message, IUs, etc), incremental, etc.
The LlamaCppMemoryIncremental subclass handles the aspects related to the LLM engineering.

Definition :
- LlamaCpp : Using the optimization library llama-cpp-python (execution in C++) for faster inference.
- Memory : Record the dialogue history by saving the dialogue turns from both the user and the system.
Update the dialogue history do that it doesn't exceed a certain threshold of token size.
Put the dialogue history in the prompt at each new system sentence generation.
- Incremental : During a new system sentence generation, send smaller chunks of sentence,
instead of waiting for the generation end to send the whole sentence.

Inputs : SpeechRecognitionIU

Outputs : TextIU


example of the prompt template :
prompt = "[INST] <<SYS>>\
This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
<</SYS>>\
\
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
[/INST]"
"""

import datetime
import threading
import time
import retico_core
from llama_cpp import Llama
from utils import *


class LlamaCppMemoryIncrementalInterruption:
    """Sub-class of LlamaCppMemoryIncrementalModule, LLM wrapper. Handles all the LLM engineering part.
    Called with the process_full_sentence function that generates a system answer from a constructed prompt when a user full sentence is received from the ASR.
    """

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
        device=None,
        **kwargs,
    ):
        """initialize LlamaCppMemoryIncrementalModule submodule and its attributes related to prompts, template and llama-cpp-python.
        Args:
            model_path (string): local model instantiation. The path to the desired local model weights file (my_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf for example).
            model_repo (string): HF model instantiation. The path to the desired remote hugging face model (TheBloke/Mistral-7B-Instruct-v0.2-GGUF for example).
            model_name (string): HF model instantiation. The name of the desired remote hugging face model (mistral-7b-instruct-v0.2.Q4_K_M.gguf for example).
            initial_prompt (string): _description_. Deprecated. Defaults to None.
            system_prompt (string): The dialogue scenario that you want the system to base its interactions on. Ex : "This is spoken dialogue, you are a teacher...". (will be inputted into every prompt). Defaults to None.
            context_size (int, optional): Max number of tokens that the total prompt can contain. Defaults to 2000.
            short_memory_context_size (int, optional): Max number of tokens that the short term memory (dialogue history) can contain. Has to be lower than context_size. Defaults to 500.
            n_gpu_layers (int, optional): Number of model layers you want to run on GPU. Take the model nb layers if greater. Defaults to 100.
            device (string, optional): the device the module will run on (cuda for gpu, or cpu)
        """
        # prompts attributes
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
        self.punctuation_text = [b".", b",", b";", b":", b"!", b"?", b"..."]
        self.punctuation_ids = [b[0] for b in self.punctuation_text]
        self.utterances = []
        self.size_per_utterance = []
        self.short_memory_context_size = short_memory_context_size

        self.start_prompt = b"[INST] "
        self.end_prompt = b" [/INST]"
        self.nb_tokens_end_prompt = 0
        self.sys_pre = b"<<SYS>>"
        self.sys_suf = b"<</SYS>>"
        self.user_pre = b""
        self.user_suf = b"\n\n"
        self.agent_pre = b""
        self.agent_suf = b"\n\n"
        self.user_role = b"Child :"
        self.agent_role = b"Teacher :"

        # llama-cpp-python args
        self.context_size = context_size
        self.device = device_definition(device)
        print("self.device LLM = ", self.device)
        self.n_gpu_layers = 0 if self.device != "cuda" else n_gpu_layers
        # Model loading method 1 (local init)
        self.model_path = model_path
        # Model loading method 2 (hf init)
        self.model_repo = model_repo
        self.model_name = model_name

        # Model is not loaded for the moment
        self.model = None

        # interruption
        self.interruption = False
        self.last_punctuation_id = None
        self.nb_token_since_punct = None
        self.nb_clauses = None
        self.interrupted_speaker_iu = None

    def setup(self):
        """Instantiate the model with the given model info, if insufficient info given, raise an NotImplementedError.
        Init the prompt with the initialize_prompt function.
        Calculates the stopping with the init_stop_criteria function.
        """

        n_gpu_layers = 0 if self.device != "cuda" else self.n_gpu_layers

        if self.model_path is not None:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_size,
                n_gpu_layers=n_gpu_layers,
            )

            # self.model = Llama(
            #     model_path=self.model_path,
            #     n_ctx=self.context_size,
            #     n_gpu_layers=n_gpu_layers,
            #     split_mode=2,
            # )

        elif self.model_repo is not None and self.model_name is not None:
            self.model = Llama.from_pretrained(
                repo_id=self.model_repo,
                filename=self.model_name,
                device_map=self.device,
                n_ctx=self.context_size,
                n_gpu_layers=n_gpu_layers,
            )

        else:
            raise NotImplementedError(
                "Please, when creating the module, you must give a model_path or model_repo and model_name"
            )

        self.initialize_prompt()
        self.init_stop_criteria()

    def init_stop_criteria(self):
        """Calculates the stopping token patterns using the instantiated model tokenizer."""
        self.stop_token_ids.append(self.model.token_eos)
        for pat in self.stop_token_text_patterns:
            self.stop_token_patterns.append(self.model.tokenize(pat, add_bos=False))
        for pat in self.role_token_text_patterns:
            self.role_token_patterns.append(self.model.tokenize(pat, add_bos=False))

    def initialize_prompt(self):
        """Init and format the prompt with the system prompt (dialogue scenario) and the corresponding prompt suffixes and prefixes."""
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
        """Function called to register a new user sentence into the dialogue history (utterances attribute).
        Calculates the exact token number added to the dialogue history (with template prefixes and suffixes).

        Args:
            user_sentence (string): the new user sentence to register.
        """
        user_sentence_complete = (
            self.user_pre
            + self.user_role
            + bytes(user_sentence, "utf-8")
            + self.user_suf
        )
        user_sentence_complete_nb_tokens = len(
            self.model.tokenize(user_sentence_complete)
        )
        self.utterances.append(user_sentence_complete)
        self.size_per_utterance.append(user_sentence_complete_nb_tokens)
        self.prompt += user_sentence_complete
        print(self.user_role.decode("utf-8") + user_sentence)

    def new_agent_sentence_interruption(self, new_agent_sentence):
        print("NEW AGENT INTERRUPTION")
        print("new_agent_sentence = ", new_agent_sentence)

        # remove role
        new_agent_sentence = new_agent_sentence[
            len(self.agent_role) : len(new_agent_sentence)
        ]
        print("new_agent_sentence = ", new_agent_sentence)

        # split the sentence into clauses
        sentence_clauses = []
        old_i = 0
        for i, c in enumerate(new_agent_sentence):
            if c in self.punctuation_ids or i == len(new_agent_sentence) - 1:
                sentence_clauses.append(new_agent_sentence[old_i : i + 1])
                old_i = i
        print("sentence_clauses = ", sentence_clauses)

        # remove all clauses after clause_id (the interrupted clause)
        sentence_clauses = sentence_clauses[: self.interrupted_speaker_iu.clause_id + 1]
        print("sentence_clauses = ", sentence_clauses)

        # Shorten the last agent utterance until the last char outputted by the speakermodule before the interruption
        sentence_clauses[-1] = sentence_clauses[-1][
            : self.interrupted_speaker_iu.char_id + 1
        ]
        print("sentence_clauses = ", sentence_clauses)

        # Merge the clauses back together
        new_agent_sentence = b"".join(sentence_clauses)
        print("new_agent_sentence = ", new_agent_sentence)

        # Add the prefix and sufix back
        new_agent_sentence = (
            self.agent_pre + self.agent_role + new_agent_sentence + self.agent_suf
        )
        print("new_agent_sentence = ", new_agent_sentence)

        # Calculate the number of tokens contained in the sentence
        nb_tokens = len(self.model.tokenize(new_agent_sentence))

        self.utterances.append(new_agent_sentence)
        self.size_per_utterance.append(nb_tokens)
        self.prompt += new_agent_sentence
        # not adding the agent role because it is already generated by the model.
        # TODO : change this by placing the role directly on the prompt?
        print(new_agent_sentence.decode("utf-8"))
        self.interrupted_speaker_iu = None

    def new_agent_sentence(
        self, agent_sentence, agent_sentence_nb_tokens, which_stop_criteria
    ):
        """Function called to register a new agent sentence into the dialogue history (utterances attribute).
        Calculates the exact token number added to the dialogue history (with template prefixes and suffixes).

        Args:
            agent_sentence (string): the new agent sentence to register.
            agent_sentence_nb_tokens (int): the number of token corresponding to the agent sentence (without template prefixes and suffixes).
        """
        # print("which_stop_criteria = ", which_stop_criteria)
        # print("self.interruption = ", self.interruption)
        # print("sentence = ", agent_sentence)
        if self.interruption:
            agent_sentence_reduced, nb_token_removed = self.remove_until_punctuation(
                agent_sentence
            )
            if self.interrupted_speaker_iu is not None:
                self.new_agent_sentence_interruption(agent_sentence)
                return
        elif which_stop_criteria == "stop_pattern":
            agent_sentence_reduced, nb_token_removed = self.remove_stop_patterns(
                agent_sentence
            )
        else:
            raise NotImplementedError(
                "this which_stop_criteria has not been implemented"
            )
        agent_sentence_complete = (
            self.agent_pre + agent_sentence_reduced + self.agent_suf
        )
        # TODO: do that in prepare_run() method so that it is not compute every user EOT
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

    def modify_new_agent_sentence(self, iu):
        print("MODIFY NEW AGENT SENTENCE")
        if iu.turn_id >= len(self.utterances):
            self.interrupted_speaker_iu = iu
        else:
            # TODO: check that the last sentence in self.utterances is the agent sentence
            # that has been interrupted, and that needs to be shortened
            if False:
                raise NotImplementedError(
                    "The interrupted agent sentence is not in the self.utterances list"
                )

            # get interrupted agent sentence
            new_agent_sentence = self.utterances[iu.turn_id]

            # check if last utterance is the correct turn_id
            print(f"len utterances and turn {len(self.utterances), iu.turn_id}")

            # remove it from the prompt
            self.prompt = self.prompt[0 : len(self.prompt) - len(new_agent_sentence)]

            # remove prefix and suffix from the sentence
            new_agent_sentence = new_agent_sentence[
                len(self.agent_pre) : len(new_agent_sentence) - len(self.agent_suf)
            ]
            print("new_agent_sentence = ", new_agent_sentence)

            # remove role
            new_agent_sentence = new_agent_sentence[
                len(self.agent_role) : len(new_agent_sentence)
            ]
            print("new_agent_sentence = ", new_agent_sentence)

            # split the sentence into clauses
            sentence_clauses = []
            old_i = 0
            for i, c in enumerate(new_agent_sentence):
                if c in self.punctuation_ids:
                    sentence_clauses.append(new_agent_sentence[old_i : i + 1])
                    old_i = i + 1
            print("sentence_clauses = ", sentence_clauses)

            # remove all clauses after clause_id (the interrupted clause)
            sentence_clauses = sentence_clauses[: iu.clause_id + 1]
            print("sentence_clauses = ", sentence_clauses)

            # Shorten the last agent utterance until the last char outputted by the speakermodule before the interruption
            sentence_clauses[-1] = sentence_clauses[-1][: iu.char_id + 1]
            print("sentence_clauses = ", sentence_clauses)

            # Merge the clauses back together
            new_agent_sentence = b"".join(sentence_clauses)
            print("new_agent_sentence = ", new_agent_sentence)

            # Add the prefix and sufix back
            new_agent_sentence = (
                self.agent_pre + self.agent_role + new_agent_sentence + self.agent_suf
            )
            print("new_agent_sentence = ", new_agent_sentence)

            # Calculate the number of tokens contained in the sentence
            self.size_per_utterance[-1] = len(self.model.tokenize(new_agent_sentence))

            # Add it back to the prompt
            self.prompt += new_agent_sentence

            # Add it back to utterances
            self.utterances[-1] = new_agent_sentence
            print("NEW AGENT SENTENCE : ", new_agent_sentence.decode("utf-8"))

    def remove_stop_patterns(self, sentence):
        """Function called when a stopping token pattern has been encountered during the sentence generation.
        Remove the encountered pattern from the generated sentence.

        Args:
            sentence (string): Agent new generated sentence containing a stopping token pattern.

        Returns:
            string: Agent new generated sentence without the stopping token pattern encountered.
            int: nb tokens removed (from the stopping token pattern).
        """
        last_chunck_string = sentence
        nb_token_removed = 0
        for i, pat in enumerate(self.stop_token_text_patterns):
            if pat == last_chunck_string[-len(pat) :]:
                sentence = sentence[: -len(pat)]
                nb_token_removed = len(self.stop_token_patterns[i])
                break
        while sentence[-1:] == b"\n":
            sentence = sentence[:-1]
            nb_token_removed += 1
        return sentence, nb_token_removed

    def remove_until_punctuation(self, sentence):
        """Function called when the LLm generation has been interrupted because the user started talking.
        Remove the clause that was beeing generated by the LLM, so remove words until the last punctuation.

        Args:
            sentence (string): Agent new generated sentence containing an unfinished clause, that will be removed.

        Returns:
            string: Agent new generated sentence without the unfinished clause.
            int: nb tokens removed.
        """

        # print("SNETENTCE = ", sentence)
        # print("self.last_punctuation_id = ", self.last_punctuation_id)
        # print("char = ", chr(sentence[self.last_punctuation_id - 1]))

        new_sentence = sentence[: self.last_punctuation_id]
        nb_token_removed = self.nb_token_since_punct

        nb_token_old_sentence = len(self.model.tokenize(sentence))
        nb_token_new_sentence = len(self.model.tokenize(new_sentence))
        # print("new_sentence = ", new_sentence)
        # print("nb_token_old_sentence = ", nb_token_old_sentence)
        # print("nb_token_new_sentence = ", nb_token_new_sentence)
        # print("nb_token_removed = ", nb_token_removed)

        # assert nb_token_old_sentence == nb_token_new_sentence + nb_token_removed
        # assert (
        #     bytes(chr(sentence[self.last_punctuation_id - 1]), "utf-8")
        #     in self.punctuation_text
        # )

        return new_sentence, nb_token_removed

    def remove_role_patterns(self, sentence):
        """Function called when a role token pattern has been encountered during the sentence generation.
        Remove the encountered pattern from the generated sentence.

        Args:
            sentence (string): Agent new generated sentence containing a role token pattern.

        Returns:
            (bytes, int): the agent new generated sentence without the role token pattern, and the number of token removed while removing the role token pattern from the sentence.
        """
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
        """Calculate if the current dialogue history is bigger than the size threshold (short_memory_context_size).
        If the dialogue history contains too many tokens, remove the older dialogue turns until its size is smaller than the threshold.
        """
        nb_tokens = sum(self.size_per_utterance)
        if nb_tokens + self.nb_tokens_end_prompt >= self.short_memory_context_size:
            nb_tokens_removed = 0
            nb_tokens_to_remove = (
                nb_tokens + self.nb_tokens_end_prompt - self.short_memory_context_size
            )
            while nb_tokens_to_remove >= nb_tokens_removed:
                self.utterances.pop(
                    1
                )  # pop oldest non system utterance. do not pop the system prompt (= the dialogue scenario)
                nb_tokens_removed += self.size_per_utterance.pop(1)
            self.prompt = b"".join(self.utterances)

    def is_punctuation(self, token):
        """Returns True if the token correspond to a punctuation.

        Args:
            token (list): a LLM token

        Returns:
            bool: True if the token correspond to a punctuation.
        """
        return self.model.detokenize([token]) in self.punctuation_text

    def is_stop_pattern(self, sentence):
        """Returns True if one of the stopping token patterns matches the end of the sentence.

        Args:
            sentence (string): a sentence.

        Returns:
            bool: True if one of the stopping token patterns matches the end of the sentence.
        """
        for i, pat in enumerate(self.stop_token_text_patterns):
            if pat == sentence[-len(pat) :]:
                return True, self.stop_token_patterns[i]
        return False, None

    def is_role_pattern(self, sentence):
        """Returns True if one of the role token patterns matches the beginning of the sentence.

        Args:
            sentence (string): a sentence.

        Returns:
            bool: True if one of the role token patterns matches the beginning of the sentence.
        """
        max_pattern_size = max([len(p) for p in self.role_token_text_patterns])
        # We want to only check at the very beginning of the sentence
        if max_pattern_size < len(sentence):
            return False, None
        # return True, and the stop pattern if last n characters of the sentence is a stop pattern.
        for i, pat in enumerate(self.role_token_text_patterns):
            if pat == sentence[-len(pat) :]:
                return True, self.role_token_patterns[i]
        return False, None

    def process_full_sentence(
        self, user_sentence, subprocess, subprocess_interruption, subprocess_EOT
    ):
        self.new_user_sentence(user_sentence)
        self.prepare_prompt_memory()
        agent_sentence, agent_sentence_nb_tokens, which_stop_criteria = (
            self.generate_next_sentence(
                subprocess, subprocess_interruption, subprocess_EOT
            )
        )
        self.new_agent_sentence(
            agent_sentence, agent_sentence_nb_tokens, which_stop_criteria
        )

    def generate_next_sentence(
        self,
        subprocess,
        subprocess_interruption,
        subprocess_EOT,
        top_k=40,
        top_p=0.95,
        temp=1.0,
        repeat_penalty=1.1,
    ):
        """Generates the agent next sentence from the constructed prompt (dialogue scenario, dialogue history, instruct...).
        At each generated token, check is the end of the sentence corresponds to a stopping pattern, role pattern, or punctuation.
        Sends the info to the retico Module using the submodule function.
        Stops the generation if a stopping token pattern is encountered (using the stop_multiple_utterances_generation as the stopping criteria).

        Args:
            subprocess (function): the function to call during the sentence generation to possibly send chunks of sentence to the children modules.
            top_k (int, optional): _description_. Defaults to 40.
            top_p (float, optional): _description_. Defaults to 0.95.
            temp (float, optional): _description_. Defaults to 1.0.
            repeat_penalty (float, optional): _description_. Defaults to 1.1.

        Returns:
            string: Agent new generated sentence.
            int: nb tokens in new agent sentence.
        """

        def stop_criteria(tokens, logits):
            """Deprecated.
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
            is_stop_token = stop_criteria(tokens, logits)

            max_pattern_size = max([len(p) for p in self.stop_token_patterns])
            last_chunck_string = self.model.detokenize(tokens[-max_pattern_size:])
            is_stopping_pattern, _ = self.is_stop_pattern(last_chunck_string)

            return is_stop_token or is_stopping_pattern

        def stop_multiple_utterances_generation_interruption(tokens, logits):
            """
            Enhance the previous stop_multiple_utterances_generation stopping criteria,
            with the interruption parameter beeing an additional stop,
            which means that generation stop if the user started talking.

            Args:
                tokens (_type_): tokens generated by the LLM
                logits (_type_): _description_

            Returns:
                bool: returns True if it the last generated token corresponds to self.stop_token_ids or self.stop_token_text, \
                Or if the last generated token match one of the self.stop_token_patterns.
                Or if the user started talking (self.interruption)
            """
            is_stop_token_or_pattern = stop_multiple_utterances_generation(
                tokens, logits
            )
            return is_stop_token_or_pattern or self.interruption

        # Define the parameters
        final_prompt = self.prompt + self.end_prompt
        # print("final_prompt = ", final_prompt)
        tokens = self.model.tokenize(final_prompt, special=True)

        last_sentence = b""
        last_sentence_nb_tokens = 0
        which_stop_criteria = None
        self.last_punctuation_id = 0
        self.nb_token_since_punct = 0
        self.nb_clauses = 0

        for token in self.model.generate(
            tokens,
            stopping_criteria=stop_multiple_utterances_generation_interruption,
            top_k=top_k,
            top_p=top_p,
            temp=temp,
            repeat_penalty=repeat_penalty,
        ):
            # Update module IUS
            payload = self.model.detokenize([token])
            payload_text = payload.decode("utf-8")

            # Update model short term memory
            last_sentence += payload
            last_sentence_nb_tokens += 1

            # call retico module subprocess to send update message to subscribed modules.
            is_p, stop_pattern = self.is_stop_pattern(last_sentence)
            is_r, role_pattern = self.is_role_pattern(last_sentence)

            # set last punct param if there is an interruption
            is_ponct = self.is_punctuation(token)
            if is_ponct:
                self.last_punctuation_id = len(last_sentence)
                self.nb_token_since_punct = 0
            else:
                self.nb_token_since_punct += 1
            if is_p:
                which_stop_criteria = "stop_pattern"
            len_unfinished_clause = len(last_sentence) - self.last_punctuation_id
            subprocess(
                payload_text,
                is_ponct,
                stop_pattern,
                role_pattern,
                len_unfinished_clause,
            )

        # TODO : Call the subprocess after stop criteria hit ? if the stop pattern has been uncountered, it should be called after the stop criteria hit.
        # maybe not for role pattern but at least for stop pattern and interruption ?
        # len_unfinished_clause = len(last_sentence) - self.last_punctuation_id
        # REVOKE all words from unfinished clause + empty the current_output
        if self.interruption:
            print("LLM : INT")
            subprocess_interruption()
        else:
            print("LLM : EOT")
            # TODO : An IU significating that the sentence is over should be sent here with a subprocess_EOT() fct.
            subprocess_EOT()

        return last_sentence, last_sentence_nb_tokens, which_stop_criteria


class LlamaCppMemoryIncrementalInterruptionModule(retico_core.AbstractModule):
    """A retico module that provides Natural Language Generation (NLG) using a conversational LLM (Llama-2 type).
    This class handles the aspects related to retico architecture : messaging (update message, IUs, etc), incremental, etc.
    Has a subclass, LlamaCppMemoryIncremental, that handles the aspects related to LLM engineering.

    Definition :
    - LlamaCpp : Using the optimization library llama-cpp-python (execution in C++) for faster inference.
    - Memory : Record the dialogue history by saving the dialogue turns from both the user and the system.
    Update the dialogue history do that it doesn't exceed a certain threshold of token size.
    Put the dialogue history in the prompt at each new system sentence generation.
    - Incremental : During a new system sentence generation, send smaller chunks of sentence,
    instead of waiting for the generation end to send the whole sentence.

    Inputs : SpeechRecognitionIU

    Outputs : TextIU
    """

    @staticmethod
    def name():
        return "LlamaCppMemoryIncremental Module"

    @staticmethod
    def description():
        return "A module that provides NLG using an LLM."

    @staticmethod
    def input_ius():
        return [retico_core.text.SpeechRecognitionIU, AudioVADIU, TurnAudioIU]

    @staticmethod
    def output_iu():
        return TurnTextIU

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
        device=None,
        **kwargs,
    ):
        """Initializes the LlamaCppMemoryIncremental Module.

        Args:
            model_path (string): local model instantiation. The path to the desired local model weights file (my_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf for example).
            model_repo (string): HF model instantiation. The path to the desired remote hugging face model (TheBloke/Mistral-7B-Instruct-v0.2-GGUF for example).
            model_name (string): HF model instantiation. The name of the desired remote hugging face model (mistral-7b-instruct-v0.2.Q4_K_M.gguf for example).
            initial_prompt (string): _description_. Deprecated.
            system_prompt (string): The dialogue scenario that you want the system to base its interactions on. Ex : "This is spoken dialogue, you are a teacher...". (will be inputted into every prompt)
            printing (bool, optional): You can choose to print some running info on the terminal. Defaults to False.
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
        self.model_wrapper = LlamaCppMemoryIncrementalInterruption(
            self.model_path,
            self.model_repo,
            self.model_name,
            initial_prompt,
            system_prompt,
            device=device,
            **kwargs,
        )
        self.latest_input_iu = None
        self.time_logs_buffer = []

        # interruption
        self.thread_active = False
        self.full_sentence = False
        self.msg = []
        self.interruption = False

    def recreate_sentence_from_um(self, msg):
        """recreate the complete user sentence from the strings contained in every COMMIT update message IU (msg).

        Args:
            msg (list): list of every COMMIT IUs contained in the UpdateMessage.

        Returns:
            string: the complete user sentence calculated by the ASR.
        """
        sentence = ""
        for iu in msg:
            sentence += iu.get_text() + " "
        return sentence

    def process_incremental(self):
        """Function that calls the submodule LLamaCppMemoryIncremental to generates a system answer (text) using the chosen LLM.
        Incremental : Use the subprocess function as a callback function for the submodule to call to check if the current chunk
        of generated sentence has to be sent to the Children Modules (TTS for example).

        Args:
            msg (list): list of every COMMIT IUs contained in the UpdateMessage.
        """
        if self.printing:
            print(
                "LLM : process sentence ",
                datetime.datetime.now().strftime("%T.%f")[:-3],
            )

        self.time_logs_buffer.append(
            ["Start", datetime.datetime.now().strftime("%T.%f")[:-3]]
        )

        def subprocess_interruption_2():
            """optimal function for the current Module: REVOKE all words from unfinished clause + empty the current_output

            As current_output is emptied after a punctuation has been encountered,
            the content of current_output, when an interruption happen,
            is only the ADDED words from the unfinished clause we want to remove.
            It means we just have to empty the current_output at each interruption.
            """
            if self.interruption:
                next_um = retico_core.abstract.UpdateMessage()
                # print(
                #     "REVOKING CURRENT OUTPUT : ",
                #     [iu.payload for iu in self.current_output],
                # )
                for iu in self.current_output:
                    self.revoke(iu, remove_revoked=False)
                    next_um.add_iu(iu, retico_core.UpdateType.REVOKE)
                self.current_output = []
                self.append(next_um)

        def subprocess_interruption(
            len_unfinished_clause=None, nb_token_since_punct=None
        ):
            """This function works, but is too complicated for nothing.
            As current_output is emptied after a punctuation has been encountered,
            the content of current_output, when an interruption happen,
            is only the ADDED words from the unfinished clause we want to remove.
            It means we just have to empty the current_output at each interruption.

            Args:
                len_unfinished_clause (_type_, optional): _description_. Defaults to None.
                nb_token_since_punct (_type_, optional): _description_. Defaults to None.
            """
            next_um = retico_core.abstract.UpdateMessage()
            # REVOKE if interruption
            if self.interruption:
                for i in range(nb_token_since_punct):
                    # take all IUs corresponding to unfinished clause
                    iu = self.current_output.pop(-1)
                    self.revoke(iu)
                    next_um.add_iu(iu, retico_core.UpdateType.REVOKE)
            self.append(next_um)

        def subprocess(
            payload,
            is_punctuation=None,
            stop_pattern=None,
            role_pattern=None,
            len_unfinished_clause=None,
        ):
            """
            This function will be called by the submodule at each token generation.
            It handles the communication with the subscribed module (TTS for example),
            by updating and publishing new UpdateMessage containing the new IUS.
            IUs are :
                - ADDED in every situation (the generated words are sent to the subscribed modules)
                - COMMITTED if the last token generated is a punctuation (The TTS can start generating the voice corresponding to the clause)
                - REVOKED if the last tokens generated corresponds to a stop pattern (so that the subscribed module delete the stop pattern)

            Args:
                payload (string): the text corresponding to the last generated token
                is_punctuation (bool, optional): True if the last generated token correspond to a punctuation. Defaults to None.
                stop_pattern (string, optional): Text corresponding to the generated stop_pattern. Defaults to None.
            """
            # if user started talking, do not send any IU
            # as the stopping_criteria is triggered when interruption IU is received, we could send the last IUs generated ?
            # if self.interruption:
            #     return None

            # Construct UM and IU
            next_um = retico_core.abstract.UpdateMessage()
            output_iu = self.create_iu(self.latest_input_iu)
            output_iu.set_data(
                text=payload,
                turn_id=len(self.model_wrapper.utterances),
                clause_id=self.model_wrapper.nb_clauses,
            )
            if not self.latest_input_iu:
                self.latest_input_iu = output_iu
            self.current_output.append(output_iu)

            # ADD
            next_um.add_iu(output_iu, retico_core.UpdateType.ADD)

            # # REVOKE if interruption
            # if self.interruption:
            #     print("SUBPROCESS INTERRUPTED")
            #     print(len(len_unfinished_clause))
            #     for i in range(len_unfinished_clause):
            #         # take all IUs corresponding to unfinished clause
            #         iu = self.current_output.pop(-1)
            #         self.revoke(iu)
            #         next_um.add_iu(iu, retico_core.UpdateType.REVOKE)
            #         print("Interruption revoke : ", iu.payload)
            #     # return None # TODO ?

            # REVOKE if stop patterns
            # TODO : This should not be done here, but in the generate_next_sentence fct after stop criteria hit
            if stop_pattern is not None:
                for id, token in enumerate(
                    stop_pattern
                ):  # take all IUs corresponding to stop pattern
                    iu = self.current_output.pop(
                        -1
                    )  # the IUs corresponding to the stop pattern are the last n ones where n=len(stop_pattern).
                    self.revoke(iu)
                    next_um.add_iu(iu, retico_core.UpdateType.REVOKE)

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

            # COMMIT if punctuation and not role patterns and not stop_pattern ?
            if is_punctuation and role_pattern is None and stop_pattern is None:
                self.model_wrapper.nb_clauses += 1
                for iu in self.current_output:
                    self.commit(iu)
                    next_um.add_iu(iu, retico_core.UpdateType.COMMIT)
                if self.printing:
                    print(
                        "LLM : send sentence after punctuation ",
                        datetime.datetime.now().strftime("%T.%f")[:-3],
                    )
                self.time_logs_buffer.append(
                    ["Stop", datetime.datetime.now().strftime("%T.%f")[:-3]]
                )
                self.current_output = []
            self.append(next_um)

        def subprocess_EOT():
            next_um = retico_core.abstract.UpdateMessage()
            iu = self.create_iu()
            iu.final = True
            next_um.add_iu(iu, retico_core.UpdateType.COMMIT)
            self.append(next_um)

        self.model_wrapper.process_full_sentence(
            self.recreate_sentence_from_um(self.msg),
            subprocess,
            subprocess_interruption_2,
            subprocess_EOT,
        )
        # reset because it is end of sentence
        self.current_output = []
        self.latest_input_iu = None
        self.msg = []

    def process_update(self, update_message):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L402

        Args:
            update_message (UpdateType): UpdateMessage that contains new IUs, if the IUs are COMMIT,
            it means that these IUs correspond to a complete sentence. All COMMIT IUs (msg) are processed calling the process_incremental function.

        Returns:
            _type_: returns None if update message is None.
        """
        if not update_message:
            return None
        msg = []
        for iu, ut in update_message:
            if isinstance(iu, retico_core.text.SpeechRecognitionIU):
                if ut == retico_core.UpdateType.ADD:
                    continue
                elif ut == retico_core.UpdateType.REVOKE:
                    continue
                elif ut == retico_core.UpdateType.COMMIT:
                    msg.append(iu)
                    # set self.full_sentence after modifying the self.msg ? so that it is not possible to start process_incremental witout the full sentence
                    # TODO : check if when the interruption is set to false here, it doesn't sometimes cancel the interruption from the same turn (from a AudioVADIU just received).
                    self.full_sentence = True
                    self.interruption = False
                    self.model_wrapper.interruption = False
            elif isinstance(iu, AudioVADIU):
                if ut == retico_core.UpdateType.ADD:
                    if iu.vad_state == "interruption":
                        self.interruption = True
                        self.model_wrapper.interruption = True
                        # print("LLM interruption")
                elif ut == retico_core.UpdateType.REVOKE:
                    continue
                elif ut == retico_core.UpdateType.COMMIT:
                    continue
            elif isinstance(iu, TurnAudioIU):
                if ut == retico_core.UpdateType.ADD:
                    if iu.final:
                        print(
                            "LLM : iu final, no interrupt, agent just stopped talking, ignore iu."
                        )
                    else:
                        print(
                            f"len turns = {len(self.model_wrapper.utterances), iu.turn_id}"
                        )
                        print(f"utterances = {self.model_wrapper.utterances}")
                        print(
                            f"grounded_word, word_id, iu.char_id, iu.turn_id, iu.clause.id = {iu.grounded_word, iu.word_id, iu.char_id, iu.turn_id, iu.clause_id}"
                        )
                        # print(
                        #     f"grounded_word, word_id, iu.char_id, iu.turn_id, iu.clause.id, utterances[turn_id] = {iu.grounded_word, iu.word_id, iu.char_id, iu.turn_id, iu.clause_id, self.model_wrapper.utterances[iu.turn_id]}"
                        # )
                        self.model_wrapper.modify_new_agent_sentence(iu)
                elif ut == retico_core.UpdateType.REVOKE:
                    continue
                elif ut == retico_core.UpdateType.COMMIT:
                    continue
        self.msg.extend(msg)

    def _llm_thread(self):
        """
        function running the LLM, executed ina separated thread so that the generation can be interrupted,
        if the user starts talking (the reception of an interruption AudioVADIU).
        """
        while self.thread_active:
            time.sleep(0.01)
            if self.full_sentence:
                self.process_incremental()
                self.full_sentence = False

    def setup(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L402
        """
        self.model_wrapper.setup()

    def prepare_run(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L808
        """
        self.thread_active = True
        threading.Thread(target=self._llm_thread).start()
        print("LLM started")

    def shutdown(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L819
        """
        write_logs(
            self.log_file,
            self.time_logs_buffer,
        )
        self.thread_active = False
