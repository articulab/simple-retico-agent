"""
SpeakerInterruptionModule
==================

A retico module that outputs through the computer's speakers the audio contained in
TextAlignedAudioIUs. The module stops the speakers if it receives the information that the user
started talking (user barge-in/interruption of agent turn).
The interruption information is recognized by an VADTurnAudioIU with a parameter
vad_state="interruption".

The modules sends TextAlignedAudioIUs in 2 cases :
- When it is agent end of turn : when it consumes the last TextAlignedAudioIU of an agent turn
(with the parameter final=True), it sends back that IU to indicate to other modules that the
agent has stopped talking.
- When the agent turn is interrupted by the user : when it receives an VADTurnAudioIU with a
parameter vad_state="interruption", it sends the last TextAlignedAudioIU consumed (ie. the last
audio that has been spoken by the agent in the interrupted turn). It is useful to align the
dialogue history with the last spoken words.

Inputs : TextAlignedAudioIU, VADTurnAudioIU

Outputs : TextAlignedAudioIU
"""

import datetime
import platform
import pyaudio
import retico_core
import retico_core.abstract

from utils import *


class SpeakerInterruptionModule(retico_core.AbstractModule):
    """A retico module that outputs through the computer's speakers the audio contained in
    TextAlignedAudioIUs. The module stops the speakers if it receives the information that the user
    started talking (user barge-in/interruption of agent turn).
    The interruption information is recognized by an VADTurnAudioIU with a parameter
    vad_state="interruption".

    The modules sends TextAlignedAudioIUs in 2 cases :
    - When it is agent end of turn : when it consumes the last TextAlignedAudioIU of an agent turn
    (with the parameter final=True), it sends back that IU to indicate to other modules that the
    agent has stopped talking.
    - When the agent turn is interrupted by the user : when it receives an VADTurnAudioIU with a
    parameter vad_state="interruption", it sends the last TextAlignedAudioIU consumed (ie. the last
    audio that has been spoken by the agent in the interrupted turn). It is useful to align the
    dialogue history with the last spoken words.

    Inputs : TextAlignedAudioIU, VADTurnAudioIU

    Outputs : TextAlignedAudioIU
    """

    @staticmethod
    def name():
        return "Speaker Interruption Module"

    @staticmethod
    def description():
        return "A module that plays audio to speakers and stops playing audio if the user starts speaking."

    @staticmethod
    def input_ius():
        return [TextAlignedAudioIU, VADTurnAudioIU]

    @staticmethod
    def output_iu():
        return TextAlignedAudioIU

    def __init__(
        self,
        rate=44100,
        frame_length=0.2,
        channels=1,
        sample_width=2,
        use_speaker="both",
        device_index=None,
        # log_file="speaker_interruption.csv",
        # log_folder="logs/test/16k/Recording (1)/demo",
        **kwargs,
    ):
        """
        Initializes the SpeakerInterruption Module.

        Args:
            whisper_model (string): name of the desired model, has to correspond to a model in the faster_whisper library.
            device (string): wether the model will be executed on cpu or gpu (using "cuda").
            language (string): language of the desired model, has to be contained in the constant LANGUAGE_MAPPING.
            speaker_wav (string): path to a wav file containing the desired voice to copy (for voice cloning models).
            rate (int): framerate of the played audio chunks.
            frame_length (float): duration of the played audio chunks.
            channels (int): number of channels (1=mono, 2=stereo) of the received VADTurnAudioIUs.
            sample_width (int):sample width (number of bits used to encode each frame) of the received VADTurnAudioIUs.
            use_speaker (string): wether the audio should be played in the right, left or both speakers.
            device_index(string):
            printing (bool, optional): You can choose to print some running info on the terminal. Defaults to False.
        """
        super().__init__(**kwargs)
        self.rate = rate
        self.sample_width = sample_width
        self.use_speaker = use_speaker

        self._p = pyaudio.PyAudio()

        if device_index is None:
            device_index = self._p.get_default_output_device_info()["index"]
        self.device_index = device_index

        self.stream = None
        self.time = None

        self.channels = channels
        # self.log_file = manage_log_folder(log_folder, log_file)
        self.time_logs_buffer = []
        self.first_time = True

        # interruptions parameters
        self.audio_iu_buffer = []
        self.latest_processed_iu = None
        self.interrupted_iu = None
        self.frame_length = frame_length

    def process_update(self, update_message):
        """overrides SpeakerModule : https://github.com/retico-team/retico-core/blob/main/retico_core/audio.py#L282

        overrides SpeakerModule's process_update to save logs.
        """
        # TODO: replace this method by an actual way of knowing what is the starting and ending time where the speaker is active (actually outputs time, and not receive messages).
        if self.first_time:
            self.time_logs_buffer.append(
                ["Start", datetime.datetime.now().strftime("%T.%f")[:-3]]
            )
            self.first_time = False
        else:
            self.time_logs_buffer.append(
                ["Stop", datetime.datetime.now().strftime("%T.%f")[:-3]]
            )
        for iu, ut in update_message:
            if isinstance(iu, VADTurnAudioIU):
                if ut == retico_core.UpdateType.ADD:
                    if iu.vad_state == "interruption":
                        if self.latest_processed_iu is not None:
                            # word, word_id, char_id, turn_id, clause_id = (
                            #     self.latest_processed_iu.grounded_word,
                            #     self.latest_processed_iu.word_id,
                            #     self.latest_processed_iu.char_id,
                            #     self.latest_processed_iu.turn_id,
                            #     self.latest_processed_iu.clause_id,
                            # )
                            # print(
                            #     f"PARAMS INTER SENT TO LLM = {word, word_id, char_id, turn_id, clause_id}"
                            # )

                            um = retico_core.UpdateMessage.from_iu(
                                self.latest_processed_iu, retico_core.UpdateType.ADD
                            )
                            self.append(um)
                            self.interrupted_iu = self.latest_processed_iu
                            # remove all audio in audio_buffer
                            self.audio_iu_buffer = []
                            self.latest_processed_iu = None
                        else:
                            print("SPEAKER : self.latest_processed_iu = None")

            elif isinstance(iu, TextAlignedAudioIU):
                if ut == retico_core.UpdateType.ADD:
                    if self.interrupted_iu is not None:
                        # if, after an interrupted turn, an IU from a new turn has been received
                        if not iu.final and self.interrupted_iu.turn_id != iu.turn_id:
                            self.interrupted_iu = None
                            self.audio_iu_buffer.append(iu)
                    else:
                        self.audio_iu_buffer.append(iu)

            else:
                raise TypeError("Unknown IU type " + iu)

        return None

    def callback(self, in_data, frame_count, time_info, status):
        """callback function given to the pyaudio stream that will output audio to the computer speakers.
        This function returns an audio chunk that will be written in the stream.
        It is called everytime the last chunk has been fully consumed.

        Args:
            in_data (_type_): _description_
            frame_count (_type_): number of frames in an audio chunk written in the stream
            time_info (_type_): _description_
            status (_type_): _description_

        Returns:
            (bytes, pyaudio type): the tuple containing the audio chunks (bytes)
            and the pyaudio type informing wether the stream should continue or stop.
        """
        if len(self.audio_iu_buffer) == 0:
            silence_bytes = b"\x00" * frame_count * self.channels * self.sample_width
            return (silence_bytes, pyaudio.paContinue)

        iu = self.audio_iu_buffer.pop(0)
        #
        if iu.final:
            um = retico_core.UpdateMessage.from_iu(iu, retico_core.UpdateType.ADD)
            self.append(um)
            silence_bytes = b"\x00" * frame_count * self.channels * self.sample_width
            return (silence_bytes, pyaudio.paContinue)
        else:
            data = bytes(iu.raw_audio)
            self.latest_processed_iu = iu
            return (data, pyaudio.paContinue)

    def prepare_run(self):
        """overrides SpeakerModule : https://github.com/retico-team/retico-core/blob/main/retico_core/audio.py#L288"""

        p = self._p

        if platform.system() == "Darwin":
            if self.use_speaker == "left":
                stream_info = pyaudio.PaMacCoreStreamInfo(channel_map=(0, -1))
            elif self.use_speaker == "right":
                stream_info = pyaudio.PaMacCoreStreamInfo(channel_map=(-1, 0))
            else:
                stream_info = pyaudio.PaMacCoreStreamInfo(channel_map=(0, 0))
        else:
            stream_info = None

        # Adding the stream_callback parameter should make the stream.write() function non blocking,
        # which would make it possible to run in parallel of the reception of update messages (and the uptdate of vad_state)
        self.stream = p.open(
            format=p.get_format_from_width(self.sample_width),
            channels=self.channels,
            rate=self.rate,
            input=False,
            output_host_api_specific_stream_info=stream_info,
            output=True,
            output_device_index=self.device_index,
            stream_callback=self.callback,
            frames_per_buffer=int(self.rate * self.frame_length),
        )

        self.stream.start_stream()

    def shutdown(self):
        """overrides SpeakerModule : https://github.com/retico-team/retico-core/blob/main/retico_core/audio.py#L312

        Write logs and close the audio stream."""
        # write_logs(self.log_file, self.time_logs_buffer)
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
