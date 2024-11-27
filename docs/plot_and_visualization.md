# System Execution Visualization

The retico-core package that is used to build this simple retico agent provides us with a basic logging and plotting system. The plotting system is simple : load the log messages stored in the log file, sort them according to the plot configuration file (here `src/simple-retico-agent/configs/plot_config_simple.json`) and use them to create a plot showing all log messages from all modules accross time. This system can easily be used in real time, as it can generate a plot repeatidly while the system is running.

For further information concerning the configuration of the plotting system, check the documentation : [https://retico-core-sos.readthedocs.io/en/latest/logs.html#configurate-plotting](https://retico-core-sos.readthedocs.io/en/latest/logs.html#configurate-plotting)

## Explanation of the `plot_config_simple.json` file and its visualization

```{figure} img/plot_example.png
:align: center

*A plot generated from the log file of a retico system's execution*

```

Here is a list of all general log messages and their signification :

- `create_iu` : log every time the module call its  `create_iu` function to create a new `Incremental Unit`.
- `append_UM` : log every time the module call its  `append` function to send a new `Update Message` to all modules subscribed to it.
- `process update` : log every time the module will enter `process_update` function (log called from the `_run` function), and process a new `Update Message` received.
- `start_answer_generation` : for modules that take appart in the generation of the agent's answer (`LLM` & `TTS`), log when they receive the `Update Message` indicating that they should start the generation.
- `send_clause` : for modules that sends their IU at a `clause` level of incrementality, once they generated or gathered enough word to constitute a full `clause`, they send all corresponding IUs in one `UM`, and log after sending the `UM`.
- `EOT` : for modules that are after the `ASR` module in the pipeline (`LLM`, `TTS` and `Speaker`), log once they've generated or consumed all the IUs corresponding to the agent full turn (EOT = End-Of-Turn).

Here is a list of all module-specific log messages and there signification :

- `Speaker_output_audio` : log whenever the `Speaker Module` consumes a `IU` containing speech (from `TTS`) and outputs its raw_audio through the speakers.
- `Speaker_output_silence` : log whenever the `Speaker Module` outputs silence raw audio through the speakers (during silences or user turn).
- `TTS_before_synthesize` : log just before calling the TTS model to start the voice synthesis (for a full `clause`, as it is the TTS incrementality level).
- `TTS_after_synthesize` : log just after having synthesized the voice for a full `clause` (as it is the TTS incrementality level).
- `ASR_predict` : log whenever an ASR prediciton is made from the IU received from the `VAD` during a user turn (`IUs` containing audio + voice activation).
- `VAD_VA_overlap` : log whenever the `VAD` detects an overlap between the agent and the user.
- `VAD_VA_agent` : log whenever the `VAD` detects that the agent is speaking (agent Voice Activity). It can detects this because it receives `IUs` from the `Speaker Module` (not through the analysis of audio captured by `Microphone`).
- `VAD_VA_user` : log whenever the `VAD` detects that the user is speaking (user Voice Activity), from the audio captured by the `Microphone`.
- `VAD_VA_silence` : log whenever the `VAD` detects neither agent's nor user's voice activation.
