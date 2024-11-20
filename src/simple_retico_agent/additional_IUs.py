import retico_core


class TextFinalIU(retico_core.text.TextIU):
    """TextIU with an additional final attribute."""

    @staticmethod
    def type():
        return "Text Final IU"

    def __init__(self, final=False, **kwargs):
        super().__init__(
            **kwargs,
        )
        self.final = final


class AudioFinalIU(retico_core.audio.AudioIU):
    """AudioIU with an additional final attribute."""

    @staticmethod
    def type():
        return "Audio Final IU"

    def __init__(self, final=False, **kwargs):
        super().__init__(
            **kwargs,
        )
        self.final = final


class VADIU(retico_core.audio.AudioIU):
    """AudioIU enhanced by VADModule with VA for both user and agent.

    Attributes:
        va_user (bool): user VA activation, True means voice recognized,
            False means no voice recognized.
        va_agent (bool): agent VA activation, True means audio outputted
            by the agent, False means no audio outputted by the agent.
    """

    @staticmethod
    def type():
        return "VAD IU"

    def __init__(
        self,
        va_user=None,
        va_agent=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.va_user = va_user
        self.va_agent = va_agent
