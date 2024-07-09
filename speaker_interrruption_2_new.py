"""
SpeakerModule_Interruption
==================

This module outputs the audio signal contained in the AudioIUs by the computer's speakers,
and can interrupt this audio streaming if the user starts speaking (with the reception of VADStateIUs).
"""

import datetime
import pyaudio
import retico_core
import platform
import retico_core.abstract

from utils import *


class SpeakerInterruptionModule(retico_core.AbstractModule):
    """A module that consumes AudioIUs of arbitrary size and outputs them to the
    speakers of the machine. It stops outputting audio if the user starts speaking,
    because it also receives information trough the VADStateIU.
    When a new IU is incoming, the module blocks as long as the current IU is being played.
    """

    @staticmethod
    def name():
        return "Speaker Interruption Module"

    @staticmethod
    def description():
        return "A consuming module that plays audio from speakers and stops playing audio if the user starts speaking (it receives a VADStateIU == True)"

    @staticmethod
    def input_ius():
        return [TurnAudioIU, AudioVADIU]

    @staticmethod
    def output_iu():
        return TurnAudioIU

    def __init__(
        self,
        rate=44100,
        frame_length=0.2,
        sample_width=2,
        channels=1,
        use_speaker="both",
        device_index=None,
        log_file="speaker_interruption.csv",
        log_folder="logs/test/16k/Recording (1)/demo",
        **kwargs,
    ):
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
        self.log_file = manage_log_folder(log_folder, log_file)
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
            if isinstance(iu, AudioVADIU):
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

            elif isinstance(iu, TurnAudioIU):
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

    def setup(self):
        """overrides SpeakerModule : https://github.com/retico-team/retico-core/blob/main/retico_core/audio.py#L288"""
        return

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
        write_logs(
            self.log_file,
            self.time_logs_buffer,
        )
        self.stream.stop_stream()
        self.stream.close()
        self.stream = None
