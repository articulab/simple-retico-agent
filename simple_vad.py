import pydub
import webrtcvad

import retico_core
from retico_core.audio import AudioIU
from retico_core.text import TextIU
from additional_IUs import VADIU


class SimpleVADModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "VAD Simple Module"

    @staticmethod
    def description():
        return "a module enhancing AudioIUs with VA activation for both user (using webrtcvad's VAD) and agent (using IUs received from Speaker Module)."

    @staticmethod
    def input_ius():
        return [AudioIU, TextIU]

    @staticmethod
    def output_iu():
        return VADIU

    def __init__(
        self,
        printing=False,
        target_framerate=16000,
        input_framerate=44100,
        channels=1,
        sample_width=2,
        vad_aggressiveness=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.printing = printing
        self.target_framerate = target_framerate
        self.input_framerate = input_framerate
        self.channels = channels
        self.sample_width = sample_width
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.VA_agent = False

    def resample_audio(self, audio):
        """Resample the audio's frame_rate to correspond to self.target_framerate.

        Args:
            audio (bytes): the audio received from the microphone that could need resampling.

        Returns:
            bytes: the resampled audio chunk.
        """
        if self.input_framerate != self.target_framerate:
            s = pydub.AudioSegment(
                audio,
                sample_width=self.sample_width,
                channels=self.channels,
                frame_rate=self.input_framerate,
            )
            s = s.set_frame_rate(self.target_framerate)
            return s._data
        return audio

    def process_update(self, update_message):
        """
        overrides AbstractModule : https://github.com/retico-team/retico-core/blob/main/retico_core/abstract.py#L402

        Args:
            update_message (UpdateType): UpdateMessage that contains new IUs, if the IUs are ADD,
            they are added to the audio_buffer.
        """
        for iu, ut in update_message:
            # IUs from SpeakerModule, can be either agent BOT or EOT
            if isinstance(iu, TextIU):
                if ut == retico_core.UpdateType.ADD:
                    # agent BOT
                    if iu.payload == "agent_BOT":
                        self.VA_agent = True
                    # agent EOT
                    elif iu.payload == "agent_EOT":
                        self.VA_agent = False
            elif isinstance(iu, AudioIU):
                if ut == retico_core.UpdateType.ADD:
                    if self.input_framerate != iu.rate:
                        raise ValueError(
                            f"input framerate differs from iu framerate : {self.input_framerate} vs {iu.rate}"
                        )
                    audio = self.resample_audio(iu.raw_audio)
                    VA_user = self.vad.is_speech(audio, self.target_framerate)
                    output_iu = self.create_iu(
                        grounded_in=iu,
                        audio=audio,
                        nframes=iu.nframes,
                        rate=self.input_framerate,
                        sample_width=self.sample_width,
                        va_user=VA_user,
                        va_agent=self.VA_agent,
                    )
                    um = retico_core.UpdateMessage.from_iu(
                        output_iu, retico_core.UpdateType.ADD
                    )
                    self.append(um)

                    # something for logging
                    if self.VA_agent:
                        if VA_user:
                            event = "VA_overlap"
                        else:
                            event = "VA_agent"
                    else:
                        if VA_user:
                            event = "VA_user"
                        else:
                            event = "VA_silence"
                    self.file_logger.info(event)
