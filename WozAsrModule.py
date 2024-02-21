import asyncio
import time
import wave
import retico_core

from retico_core.audio import AudioIU
from retico_core.text import SpeechRecognitionIU

# from audio import AudioIU, SpeechIU


class WozAsrModule(retico_core.AbstractProducingModule):
    """A module that produces fake SpeechRecognitionIUs containing text captured from the text written by the user on the terminal."""

    @staticmethod
    def name():
        return "woz asr Module"

    @staticmethod
    def description():
        return (
            "A producing module that produce audio from terminal user input."
        )

    @staticmethod
    def output_iu():
        return SpeechRecognitionIU
        # return SpeechIU

    def __init__(self, **kwargs):
        """
        Initialize the WozAsrModule.

        Args:
        """
        super().__init__(**kwargs)
        self.sentence = None

    async def gather_sentence(self, question):
        # result = await self.llama.generate(sentence)
        print("before input")
        result = input(question)
        self.sentence = result
        # print("after input :")
        # output_iu = self.create_iu()
        # # we will not use the folowing
        # predictions = []
        # stability = 0.5
        # confidence = 0.9
        # final = True
        # output_iu.set_asr_results(
        #     predictions, result, stability, confidence, final
        # )
        # self.sentence = retico_core.UpdateMessage.from_iu(
        #     output_iu, retico_core.UpdateType.ADD
        # )
        # return retico_core.UpdateMessage.from_iu(
        #     output_iu, retico_core.UpdateType.ADD
        # )

        # question = "write your sentence : "
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # coroutine = gather_sentence(question)
        # loop.run_until_complete(coroutine)

    async def async_update(self):
        question = "write your sentence : "
        um = await self.gather_sentence(question)
        return um

    def process_update(self, _):
        print("update")
        question = "write your sentence : "
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        coroutine = self.gather_sentence(question)
        loop.run_until_complete(coroutine)
        # um = self.async_update()
        # self.append(um)
        print("sentence = " + str(self.sentence))
        print("after input :")
        output_iu = self.create_iu()
        # we will not use the folowing
        predictions = []
        stability = 0.5
        confidence = 0.9
        final = True
        output_iu.set_asr_results(
            predictions, self.sentence, stability, confidence, final
        )
        um = retico_core.UpdateMessage.from_iu(
            output_iu, retico_core.UpdateType.ADD
        )
        self.append(um)
        print("after process_update :")

    def setup(self):
        pass

    def prepare_run(self):
        # print("prepare_run")
        # question = "write your sentence : "
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # coroutine = self.gather_sentence(question)
        # loop.run_until_complete(coroutine)
        pass

    def shutdown(self):
        pass
