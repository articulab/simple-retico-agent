"""
LlamaCppMemoryIncrementalModule
==================

A retico module that provides Natural Language Generation (NLG) using a Large Language Model
(LLM), and handles user interruption.

when a full user sentence (COMMIT SpeechRecognitionIUs) is received from the ASR, the module
adds the sentence to the previous dialogue turns stored in the dialogue history to build a
constructed prompt (with a defined template), it then use it to generates a system answer.
During the generation, if a punctuation is encountered (end of clause), the IUs corresponding
to the generated clause are COMMITED.
Record the dialogue history by saving the dialogue turns from both the user and the agent.
Update the dialogue history do that it doesn't exceed a certain threshold of token size.
Put the dialogue history in the prompt at each new system sentence generation.
The module stops its generation if it receives the information that the user started talking
(user barge-in/interruption of agent turn). The interruption information is recognized by
an VADTurnAudioIU with a parameter vad_state="interruption". After an interruption, it aligns the
agent interrupted sentence in dialogue history with the last word spoken by the agent (these
informations are recognized when a TextAlignedAudioIU with parameter final = False is received after
an interruption).

The llama-cpp-python library is used to speed up the LLM inference (execution in C++).

Inputs : SpeechRecognitionIU, VADTurnAudioIU, TextAlignedAudioIU

Outputs : TurnTextIU


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
from additional_IUs import *
from retico_core.log_utils import *


class LlamaCppMemoryIncrementalInterruptionModule(retico_core.AbstractModule):
    """A retico module that provides Natural Language Generation (NLG) using a Large Language Model
    (LLM), and handles user interruption.

    when a full user sentence (COMMIT SpeechRecognitionIUs) is received from the ASR, the module
    adds the sentence to the previous dialogue turns stored in the dialogue history to build a
    constructed prompt (with a defined template), it then use it to generates a system answer.
    During the generation, if a punctuation is encountered (end of clause), the IUs corresponding
    to the generated clause are COMMITED.
    Record the dialogue history by saving the dialogue turns from both the user and the agent.
    Update the dialogue history do that it doesn't exceed a certain threshold of token size.
    Put the dialogue history in the prompt at each new system sentence generation.
    The module stops its generation if it receives the information that the user started talking
    (user barge-in/interruption of agent turn). The interruption information is recognized by
    an VADTurnAudioIU with a parameter vad_state="interruption". After an interruption, it aligns the
    agent interrupted sentence in dialogue history with the last word spoken by the agent (these
    informations are recognized when a TextAlignedAudioIU with parameter final = False is received after
    an interruption).

    The llama-cpp-python library is used to speed up the LLM inference (execution in C++).

    Inputs : SpeechRecognitionIU, VADTurnAudioIU, TextAlignedAudioIU

    Outputs : TurnTextIU
    """

    @staticmethod
    def name():
        return "LlamaCppMemoryIncremental Module"

    @staticmethod
    def description():
        return "A module that provides NLG using an LLM."

    @staticmethod
    def input_ius():
        return [
            retico_core.text.SpeechRecognitionIU,
            VADTurnAudioIU,
            TextAlignedAudioIU,
        ]

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
        # log_file="llm.csv",
        # log_folder="logs/test/16k/Recording (1)/demo",
        device=None,
        context_size=2000,
        short_memory_context_size=500,
        n_gpu_layers=100,
        **kwargs,
    ):
        """
        Initializes the LlamaCppMemoryIncremental Module.

        Args:
            model_path (string): local model instantiation. The path to the desired local model weights file (my_models/mistral-7b-instruct-v0.2.Q4_K_M.gguf for example).
            model_repo (string): HF model instantiation. The path to the desired remote hugging face model (TheBloke/Mistral-7B-Instruct-v0.2-GGUF for example).
            model_name (string): HF model instantiation. The name of the desired remote hugging face model (mistral-7b-instruct-v0.2.Q4_K_M.gguf for example).
            initial_prompt (string): _description_. Deprecated. Defaults to None.
            system_prompt (string): The dialogue scenario that you want the system to base its interactions on. Ex : "This is spoken dialogue, you are a teacher...". (will be inputted into every prompt)
            context_size (int, optional): Max number of tokens that the total prompt can contain. Defaults to 2000.
            short_memory_context_size (int, optional): Max number of tokens that the short term memory (dialogue history) can contain. Has to be lower than context_size. Defaults to 500.
            n_gpu_layers (int, optional): Number of model layers you want to run on GPU. Take the model nb layers if greater. Defaults to 100.
            device (string, optional): the device the module will run on (cuda for gpu, or cpu)
            printing (bool, optional): You can choose to print some running info on the terminal. Defaults to False.
        """
        super().__init__(**kwargs)

        # model
        self.model = None
        self.model_path = model_path
        self.model_repo = model_repo
        self.model_name = model_name
        self.context_size = context_size
        self.device = device_definition(device)
        self.n_gpu_layers = 0 if self.device != "cuda" else n_gpu_layers

        # general
        self.printing = printing
        self.time_logs_buffer = []
        # self.log_file = manage_log_folder(log_folder, log_file)
        self.thread_active = False
        self.full_sentence = False
        self.interruption = False
        self.nb_clauses = None
        self.interrupted_speaker_iu = None
        self.which_stop_criteria = None

        # prompts attributes
        self.initial_prompt = initial_prompt
        self.system_prompt = system_prompt
        self.prompt = None
        self.utterances = []
        self.utterances_length = []
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
        self.stop_token_ids = []
        self.stop_token_patterns = []
        self.stop_token_text_patterns = [b"Child:", b"Child :"]
        self.role_token_patterns = []
        self.role_token_text_patterns = [
            b"Teacher:",
            b"Teacher :",
            b" Teacher:",
            b" Teacher :",
        ]
        self.max_role_pattern_length = max(
            [len(p) for p in self.role_token_text_patterns]
        )
        self.punctuation_text = [b".", b",", b";", b":", b"!", b"?", b"..."]
        self.punctuation_ids = [b[0] for b in self.punctuation_text]

    #######
    # LLM MODULE
    #######

    def init_stop_criteria(self):
        """Calculates the stopping token patterns using the instantiated model tokenizer."""
        self.stop_token_ids.append(self.model.token_eos())
        for pat in self.stop_token_text_patterns:
            self.stop_token_patterns.append(self.model.tokenize(pat, add_bos=False))
        for pat in self.role_token_text_patterns:
            self.role_token_patterns.append(self.model.tokenize(pat, add_bos=False))

    def initialize_prompt(self):
        """Init and format the prompt with the system prompt (dialogue scenario) and the corresponding prompt suffixes and prefixes."""
        self.nb_tokens_end_prompt = len(
            self.model.tokenize(self.end_prompt, add_bos=False)
        )
        if self.initial_prompt is None:
            if self.system_prompt is None:
                pass
            else:
                complete_system_prompt = (
                    self.start_prompt + self.sys_pre + self.system_prompt + self.sys_suf
                )
                self.prompt = complete_system_prompt
                self.utterances = [complete_system_prompt]
                self.utterances_length = [
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
        self.utterances_length.append(user_sentence_complete_nb_tokens)
        self.prompt += user_sentence_complete
        print(self.user_role.decode("utf-8") + user_sentence + "\n\n")

    def new_agent_sentence(self, agent_sentence, agent_sentence_nb_tokens):
        """Function called to register a new agent sentence into the dialogue history (utterances attribute).
        Calculates the exact token number added to the dialogue history (with template prefixes and suffixes).

        Args:
            agent_sentence (string): the new agent sentence to register.
            agent_sentence_nb_tokens (int): the number of token corresponding to the agent sentence (without template prefixes and suffixes).
        """
        agent_sentence_complete = self.agent_pre + agent_sentence + self.agent_suf
        # TODO: do that in prepare_run() method so that it is not compute every user EOT
        nb_token_added = len(self.model.tokenize(self.agent_pre, add_bos=False)) + len(
            self.model.tokenize(self.agent_suf, add_bos=False)
        )
        agent_sentence_complete_nb_tokens = agent_sentence_nb_tokens + nb_token_added
        self.utterances.append(agent_sentence_complete)
        self.utterances_length.append(agent_sentence_complete_nb_tokens)
        self.prompt += agent_sentence_complete
        # not adding the agent role because it is already generated by the model.
        # TODO : change this by placing the role directly on the prompt?
        print(agent_sentence_complete.decode("utf-8"))

    def interruption_alignment_new_agent_sentence(self, new_agent_sentence):
        """After an interruption, this function will align the sentence stored in dialogue history with the last word spoken by the agent.

        This function is triggered if the interrupted speaker IU has been received before the module has stored the new agent sentence in the dialogue history.
        If that is not the case, the function interruption_alignment_last_agent_sentence is triggered instead.

        With the informations stored in self.interrupted_speaker_iu, this function will shorten the new_agent_sentence to be aligned with the last words spoken by the agent.

        Args:
            new_agent_sentence (string): the utterance generated by the LLM, that has been interrupted by the user and needs to be aligned.
        """
        # remove role
        new_agent_sentence = new_agent_sentence[
            len(self.agent_role) : len(new_agent_sentence)
        ]

        # split the sentence into clauses
        sentence_clauses = []
        old_i = 0
        for i, c in enumerate(new_agent_sentence):
            if c in self.punctuation_ids or i == len(new_agent_sentence) - 1:
                sentence_clauses.append(new_agent_sentence[old_i : i + 1])
                old_i = i + 1

        # remove all clauses after clause_id (the interrupted clause)
        sentence_clauses = sentence_clauses[: self.interrupted_speaker_iu.clause_id + 1]

        # Shorten the last agent utterance until the last char outputted by the speakermodule before the interruption
        sentence_clauses[-1] = sentence_clauses[-1][
            : self.interrupted_speaker_iu.char_id + 1
        ]

        # Merge the clauses back together
        new_agent_sentence = b"".join(sentence_clauses)

        # Add the prefix and sufix back
        new_agent_sentence = (
            self.agent_pre + self.agent_role + new_agent_sentence + self.agent_suf
        )

        # Calculate the number of tokens contained in the sentence
        nb_tokens = len(self.model.tokenize(new_agent_sentence))

        # Add it to the prompt and utterances
        self.utterances.append(new_agent_sentence)
        self.utterances_length.append(nb_tokens)
        self.prompt += new_agent_sentence
        # not adding the agent role because it is already generated by the model.
        # TODO : change this by placing the role directly on the prompt?
        print("INTERRUPTED AGENT SENTENCE : ", new_agent_sentence.decode("utf-8"))
        # print("NEW PROMPT : ", self.prompt.decode("utf-8"))
        self.interrupted_speaker_iu = None

    def interruption_alignment_last_agent_sentence(self, iu):
        """After an interruption, this function will align the sentence stored in dialogue history with the last word spoken by the agent.

        If the turn that the speaker IU is a turn further from the last utterance in self.utterances,
        store the speaker IU in self.interrupted_speaker_iu and wait for the sentence generation to end,
        the interruption_alignment_new_agent_sentence function will then be called.

        Args:
            iu (AudioTurnIU): The IU received from the Speaker module, that correspond to the last IU that has been outputted through the speakers.
        """
        self.interrupted_speaker_iu = iu
        if iu.turn_id < len(self.utterances):
            # get interrupted agent sentence and remove it from the prompt and utterances
            self.utterances_length.pop(iu.turn_id)
            new_agent_sentence = self.utterances.pop(iu.turn_id)
            # TODO: This only works if the interrupted utterance is the last utterance added in prompt
            self.prompt = self.prompt[0 : len(self.prompt) - len(new_agent_sentence)]
            self.interruption_alignment_new_agent_sentence(new_agent_sentence)

    def remove_stop_patterns(self, sentence, pattern_id):
        """Function called when a stopping token pattern has been encountered during the sentence generation.
        Remove the encountered pattern from the generated sentence.

        Args:
            sentence (string): Agent new generated sentence containing a stopping token pattern.

        Returns:
            string: Agent new generated sentence without the stopping token pattern encountered.
            int: nb tokens removed (from the stopping token pattern).
        """
        # print("patttern_id = ", pattern_id)
        # print(
        #     "self.stop_token_text_patterns[pattern_id] = ",
        #     self.stop_token_text_patterns[pattern_id],
        # )
        sentence = sentence[: -len(self.stop_token_text_patterns[pattern_id])]
        nb_token_removed = len(self.stop_token_patterns[pattern_id])
        while sentence[-1:] == b"\n":
            sentence = sentence[:-1]
            nb_token_removed += 1
        return sentence, nb_token_removed

    def identify_and_remove_stop_patterns(self, sentence):
        """Function called when a stopping token pattern has been encountered during the sentence generation.
        Remove the encountered pattern from the generated sentence.

        Args:
            sentence (string): Agent new generated sentence containing a stopping token pattern.

        Returns:
            string: Agent new generated sentence without the stopping token pattern encountered.
            int: nb tokens removed (from the stopping token pattern).
        """
        nb_token_removed = 0
        for i, pattern in enumerate(self.stop_token_text_patterns):
            if pattern == sentence[-len(pattern) :]:
                sentence = sentence[: -len(pattern)]
                nb_token_removed = len(self.stop_token_patterns[i])
                break
        while sentence[-1:] == b"\n":
            sentence = sentence[:-1]
            nb_token_removed += 1
        return sentence, nb_token_removed

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
        return sentence, nb_token_removed

    def prepare_dialogue_history(self):
        """Calculate if the current dialogue history is bigger than the size threshold (short_memory_context_size).
        If the dialogue history contains too many tokens, remove the older dialogue turns until its size is smaller than the threshold.
        """
        nb_tokens = sum(self.utterances_length)
        if nb_tokens + self.nb_tokens_end_prompt >= self.short_memory_context_size:
            nb_tokens_removed = 0
            nb_tokens_to_remove = (
                nb_tokens + self.nb_tokens_end_prompt - self.short_memory_context_size
            )
            while nb_tokens_to_remove >= nb_tokens_removed:
                self.utterances.pop(
                    1
                )  # pop oldest non system utterance. do not pop the system prompt (= the dialogue scenario)
                nb_tokens_removed += self.utterances_length.pop(1)
            self.prompt = b"".join(self.utterances)

    def is_punctuation(self, word):
        """Returns True if the token correspond to a punctuation.

        Args:
            word (bytes): a detokenized word corresponding to last generated LLM token

        Returns:
            bool: True if the token correspond to a punctuation.
        """
        return word in self.punctuation_text

    def is_stop_token(self, token):
        """
        Function used by the LLM to stop generate tokens when it meets certain criteria.

        Args:
            token (int): last token generated by the LLM
            word (bytes): the detokenized word corresponding to last generated token

        Returns:
            bool: returns True if it the last generated token corresponds to self.stop_token_ids or self.stop_token_text.
        """
        is_stopping_id = token in self.stop_token_ids
        return is_stopping_id

    def is_stop_pattern(self, sentence):
        """Returns True if one of the stopping token patterns matches the end of the sentence.

        Args:
            sentence (string): a sentence.

        Returns:
            bool: True if one of the stopping token patterns matches the end of the sentence.
        """
        for i, pat in enumerate(self.stop_token_text_patterns):
            if pat == sentence[-len(pat) :]:
                return True, self.stop_token_patterns[i], i
        return False, None, None

    def is_role_pattern(self, sentence):
        """Returns True if one of the role token patterns matches the beginning of the sentence.

        Args:
            sentence (string): a sentence.

        Returns:
            bool: True if one of the role token patterns matches the beginning of the sentence.
        """
        # We want to only check at the very beginning of the sentence
        if self.max_role_pattern_length < len(sentence):
            return False, None
        # return True, and the stop pattern if last n characters of the sentence is a stop pattern.
        for i, pat in enumerate(self.role_token_text_patterns):
            if pat == sentence[-len(pat) :]:
                return True, self.role_token_patterns[i]
        return False, None

    def generate_next_sentence(
        self,
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

        def stop_function(tokens, logits):
            """stop the sentence generation if the which_stop_criteria class parameter is not None,
            i.e. when a stop token, pattern has been encountered or the user interrupted the agent.

            Args:
                tokens (_type_): tokens generated by LLM during current turn.
                logits (_type_): _description_

            Returns:
                bool: True if which_stop_criteria is not None.
            """
            return self.which_stop_criteria is not None

        # Define the parameters
        final_prompt = self.prompt + self.end_prompt
        # print("final_prompt = ", final_prompt)
        tokens = self.model.tokenize(final_prompt, special=True)

        last_sentence = b""
        last_sentence_nb_tokens = 0
        self.which_stop_criteria = None
        self.nb_clauses = 0

        # IMPORTANT : the stop crit is executed after the body of the for loop,
        # which means token here is seen inside the loop before being accessible in stop crit funct
        for token in self.model.generate(
            tokens,
            stopping_criteria=stop_function,
            top_k=top_k,
            top_p=top_p,
            temp=temp,
            repeat_penalty=repeat_penalty,
        ):
            # get word from token
            word_bytes = self.model.detokenize([token])
            word = word_bytes.decode("utf-8")

            # Update current generated sentence and nb tokens
            last_sentence += word_bytes
            last_sentence_nb_tokens += 1

            # Check if the sentence generation should be stopped (interruption or EOT)
            is_stop_token = self.is_stop_token(token)
            is_stop_pattern, stop_pattern, pattern_id = self.is_stop_pattern(
                last_sentence
            )
            if self.interruption:
                self.which_stop_criteria = "interruption"
            elif is_stop_pattern:
                self.which_stop_criteria = "stop_pattern_" + str(pattern_id)
            elif is_stop_token:
                self.which_stop_criteria = "stop_token"
            elif self.which_stop_criteria is None:
                is_role_pattern, role_pattern = self.is_role_pattern(last_sentence)
                is_punctuation = self.is_punctuation(word_bytes)
                self.incremental_iu_sending(
                    word,
                    is_punctuation,
                    role_pattern,
                )

        return last_sentence, last_sentence_nb_tokens

    #######
    # RETICO MODULE
    #######

    def recreate_sentence_from_um(self, msg):
        """recreate the complete user sentence from the strings contained in every COMMIT update message IU (msg).

        Args:
            msg (list): list of every COMMIT IUs contained in the UpdateMessage.

        Returns:
            string: the complete user sentence calculated by the ASR.
        """
        sentence = ""
        for iu in msg:
            # sentence += iu.get_text() + " "
            sentence += iu.payload + " "
        return sentence

    def incremental_iu_sending(
        self,
        payload,
        is_punctuation=None,
        role_pattern=None,
    ):
        """
        This function will be called by the submodule at each token generation.
        It handles the communication with the subscribed module (TTS for example),
        by updating and publishing new UpdateMessage containing the new IUS.
        IUs are :
            - ADDED in every situation (the generated words are sent to the subscribed modules)
            - COMMITTED if the last token generated is a punctuation (The TTS can start generating the voice corresponding to the clause)
            - REVOKED if the last tokens generated corresponds to a role pattern (so that the subscribed module delete the role pattern)

        Args:
            payload (string): the text corresponding to the last generated token
            is_punctuation (bool, optional): True if the last generated token correspond to a punctuation. Defaults to None.
            stop_pattern (string, optional): Text corresponding to the generated stop_pattern. Defaults to None.
        """
        # Construct UM and IU
        next_um = retico_core.abstract.UpdateMessage()
        # if len(self.current_input) > 0:
        #     output_iu = self.create_iu(self.current_input[-1])
        # else:
        #     output_iu = self.create_iu()
        # output_iu.set_data(
        #     text=payload,
        #     turn_id=len(self.utterances),
        #     clause_id=self.nb_clauses,
        # )
        last_iu = None
        if len(self.current_input) > 0:
            last_iu = self.current_input[-1]
        # print("last_iu = ", last_iu)
        output_iu = self.create_iu(
            grounded_in=last_iu,
            text=payload,
            turn_id=len(self.utterances),
            clause_id=self.nb_clauses,
        )
        self.current_output.append(output_iu)

        # ADD
        next_um.add_iu(output_iu, retico_core.UpdateType.ADD)

        # REVOKE if role patterns
        if role_pattern is not None:
            for id, token in enumerate(
                role_pattern
            ):  # take all IUs corresponding to stop pattern
                iu = self.current_output.pop(
                    -1
                )  # the IUs corresponding to the stop pattern are the last n ones where n=len(stop_pattern).
                iu.revoked = True
                next_um.add_iu(iu, retico_core.UpdateType.REVOKE)

        # COMMIT if punctuation and not role patterns and not stop_pattern
        if is_punctuation and role_pattern is None:
            self.nb_clauses += 1
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

    def process_incremental(self):
        """Function that calls the submodule LLamaCppMemoryIncremental to generates a system answer (text) using the chosen LLM.
        Incremental : Use the subprocess function as a callback function for the submodule to call to check if the current chunk
        of generated sentence has to be sent to the Children Modules (TTS for example).

        """

        if self.printing:
            print(
                "LLM : process sentence ",
                datetime.datetime.now().strftime("%T.%f")[:-3],
            )

        self.time_logs_buffer.append(
            ["Start", datetime.datetime.now().strftime("%T.%f")[:-3]]
        )

        # TODO : find a way to have only one data buffer for generated token/text. currently we have competitively IU buffer (current_output), and text buffer (agent_sentence).
        # this way, we would only have to remove from one buffer when deleting stop pattern, or role pattern.
        user_sentence = self.recreate_sentence_from_um(self.current_input)
        self.new_user_sentence(user_sentence)
        self.prepare_dialogue_history()
        last_processed_iu = self.current_input[-1]

        agent_sentence, agent_sentence_nb_tokens = self.generate_next_sentence()

        next_um = retico_core.abstract.UpdateMessage()

        if self.which_stop_criteria == "interruption":
            # REVOKE every word in interrupted clause (every IU in current_output)
            for iu in self.current_output:
                self.revoke(iu, remove_revoked=False)
                next_um.add_iu(iu, retico_core.UpdateType.REVOKE)

            # align dialogue history with last word spoken by speaker module
            if self.interrupted_speaker_iu is not None:
                self.interruption_alignment_new_agent_sentence(agent_sentence)

        elif self.which_stop_criteria.startswith("stop_pattern"):
            # remove from agent sentence every word contained in the stop pattern encountered
            pattern_id = int(self.which_stop_criteria.split("_")[-1])
            agent_sentence_reduced, nb_token_removed = self.remove_stop_patterns(
                agent_sentence, pattern_id
            )

            # REVOKE every word contained in the stop pattern encountered
            for i in range(nb_token_removed - 1):
                iu = self.current_output.pop(-1)
                iu.revoked = True
                next_um.add_iu(iu, retico_core.UpdateType.REVOKE)
            # add an IU significating that the agent turn is complete (EOT)
            # iu = self.create_iu()
            # iu.set_data(final=True)
            iu = self.create_iu(grounded_in=last_processed_iu, final=True)
            next_um.add_iu(iu, retico_core.UpdateType.COMMIT)

            # add updated agent sentence to dialogue history
            self.new_agent_sentence(
                agent_sentence_reduced, agent_sentence_nb_tokens - nb_token_removed
            )

        elif self.which_stop_criteria == "stop_token":
            # add an IU significating that the agent turn is complete (EOT)
            # iu = self.create_iu()
            # iu.set_data(final=True)
            iu = self.create_iu(grounded_in=last_processed_iu, final=True)
            next_um.add_iu(iu, retico_core.UpdateType.COMMIT)

            # add updated agent sentence to dialogue history
            self.new_agent_sentence(agent_sentence, agent_sentence_nb_tokens)

        else:
            raise NotImplementedError(
                "this which_stop_criteria has not been implemented"
            )

        self.append(next_um)

        # reset because it is end of sentence
        self.current_output = []
        # reset current_input ?? It could erase new IU from the next turn ?
        self.current_input = []

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
            elif isinstance(iu, VADTurnAudioIU):
                if ut == retico_core.UpdateType.ADD:
                    if iu.vad_state == "interruption":
                        self.interruption = True
                elif ut == retico_core.UpdateType.REVOKE:
                    continue
                elif ut == retico_core.UpdateType.COMMIT:
                    continue
            elif isinstance(iu, TextAlignedAudioIU):
                if ut == retico_core.UpdateType.ADD:
                    if iu.final:
                        # print(
                        #     "LLM : iu final, no interrupt, agent just stopped talking, ignore iu."
                        # )
                        pass
                    else:
                        # print(
                        #     f"len turns = {len(self.model_wrapper.utterances), iu.turn_id}"
                        # )
                        # print(f"utterances = {self.model_wrapper.utterances}")
                        # print(
                        #     f"grounded_word, word_id, iu.char_id, iu.turn_id, iu.clause.id = {iu.grounded_word, iu.word_id, iu.char_id, iu.turn_id, iu.clause_id}"
                        # )
                        self.interruption_alignment_last_agent_sentence(iu)
                elif ut == retico_core.UpdateType.REVOKE:
                    continue
                elif ut == retico_core.UpdateType.COMMIT:
                    continue

        if len(msg) > 0:
            self.current_input.extend(msg)
            self.full_sentence = True
            self.interruption = False

    def _llm_thread(self):
        """
        function running the LLM, executed ina separated thread so that the generation can be interrupted,
        if the user starts talking (the reception of an interruption VADTurnAudioIU).
        """
        while self.thread_active:
            time.sleep(0.01)
            if self.full_sentence:
                self.terminal_logger.info("start_process")
                self.file_logger.info("start_process")
                self.process_incremental()
                self.full_sentence = False

    def setup(self, **kwargs):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L402

        Instantiate the model with the given model info, if insufficient info given, raise an NotImplementedError.
        Init the prompt with the initialize_prompt function.
        Calculates the stopping with the init_stop_criteria function.
        """
        super().setup(**kwargs)

        if self.model_path is not None:
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.context_size,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.printing,
            )

        elif self.model_repo is not None and self.model_name is not None:
            self.model = Llama.from_pretrained(
                repo_id=self.model_repo,
                filename=self.model_name,
                device_map=self.device,
                n_ctx=self.context_size,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.printing,
            )

        else:
            raise NotImplementedError(
                "Please, when creating the module, you must give a model_path or model_repo and model_name"
            )

        self.initialize_prompt()
        self.init_stop_criteria()

    def prepare_run(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L808
        """
        super().prepare_run()
        self.thread_active = True
        threading.Thread(target=self._llm_thread).start()
        print("LLM started")

    def shutdown(self):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L819
        """
        super().shutdown()
        # write_logs(self.log_file, self.time_logs_buffer)
        self.thread_active = False
