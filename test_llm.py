import datetime
import time
import torch
from llama_cpp import Llama

from dialogue_manager import DialogueHistory

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"

model = Llama(
    model_path=model_path,
    n_ctx=2000,
    n_gpu_layers=100,
    verbose=True,
)

prompt = b"[INST] <<SYS>>\
This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
<</SYS>>\
\
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
Child : Hello ! \
Teacher : Hi! How are your today ? \
Child : I am fine, and I can't wait to learn mathematics ! \
    This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
    This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
    This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
    This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
    This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
    This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
    This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
    This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
    This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
    This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
    This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
    This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
    This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
    This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
    This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation : \
[/INST]"

# print(prompt)

# starttime = time.time()
# startdate = datetime.datetime.now()
# tokens = model.tokenize(prompt, add_bos=False)
# endtime = time.time()
# enddate = datetime.datetime.now()
# print("time = ", endtime - starttime)
# print("datetime = ", enddate - startdate)
# print("len(tokens) = ", len(tokens))

# starttime = time.time()
# startdate = datetime.datetime.now()
# tokens = model.tokenize(prompt, add_bos=False)
# endtime = time.time()
# enddate = datetime.datetime.now()
# print("time = ", endtime - starttime)
# print("datetime = ", enddate - startdate)
# print("len(tokens) = ", len(tokens))

# starttime = time.time()
# startdate = datetime.datetime.now()
# tokens = model.tokenize(prompt, add_bos=False)
# endtime = time.time()
# enddate = datetime.datetime.now()
# print("time = ", endtime - starttime)
# print("datetime = ", enddate - startdate)
# print("len(tokens) = ", len(tokens))

pre = b""
role = b" Child :"
suf = b"\n\n"
prompt_0 = b"Hello, what's your name"
prompt_1 = b" How are you ?"
prompt_2 = b" Hello, what's your name "

concat_0 = pre + role + prompt_0 + suf
concat_1 = pre + role + prompt_1 + suf
concat_2 = pre + role + prompt_2 + suf

nb_tokens_pre = len(model.tokenize(pre, add_bos=False))
nb_tokens_role = len(model.tokenize(role, add_bos=False))
nb_tokens_suf = len(model.tokenize(suf, add_bos=False))

nb_tokens_prompt_0 = len(model.tokenize(prompt_0, add_bos=False))
nb_tokens_prompt_1 = len(model.tokenize(prompt_1, add_bos=False))
nb_tokens_prompt_2 = len(model.tokenize(prompt_2, add_bos=False))

nb_tokens_concat_0 = len(model.tokenize(concat_0, add_bos=False))
nb_tokens_concat_1 = len(model.tokenize(concat_1, add_bos=False))
nb_tokens_concat_2 = len(model.tokenize(concat_2, add_bos=False))

print(
    f"nb tokens concat 0 : {concat_0} : {nb_tokens_pre} + {nb_tokens_role} + {nb_tokens_prompt_0} + {nb_tokens_suf} = {nb_tokens_pre+nb_tokens_role+nb_tokens_prompt_0+nb_tokens_suf} = {nb_tokens_concat_0} "
)

print(
    f"nb tokens concat 1 : {concat_1} : {nb_tokens_pre} + {nb_tokens_role} + {nb_tokens_prompt_1} + {nb_tokens_suf} = {nb_tokens_pre+nb_tokens_role+nb_tokens_prompt_1+nb_tokens_suf} = {nb_tokens_concat_1} "
)

print(
    f"nb tokens concat 0 : {concat_2} : {nb_tokens_pre} + {nb_tokens_role} + {nb_tokens_prompt_2} + {nb_tokens_suf} = {nb_tokens_pre+nb_tokens_role+nb_tokens_prompt_2+nb_tokens_suf} = {nb_tokens_concat_2} "
)

print(model.tokenize(pre, add_bos=False, special=True))
print(model.tokenize(role, add_bos=False, special=True))
print(model.tokenize(prompt_1, add_bos=False, special=True))
print(model.tokenize(suf, add_bos=False, special=True))
print(model.tokenize(concat_1, add_bos=False, special=True))


prompt = "[INST] <<SYS>>\
This is a spoken dialog scenario between a teacher and a 8 years old child student.\
The teacher is teaching mathematics to the child student.\
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation :\
<</SYS>>\
Child : Hello !\n\n\
Teacher : Hi! How are your today ?\n\n\
Child : I am fine, and I can't wait to learn mathematics !\n\n\
[/INST]"

sentence_0 = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation. \
You play the role of a teacher. Here is the beginning of the conversation :"
sentence_1 = "Hello ! How are you to..."
sentence_2 = "Hi!"
sentence_3 = "Hello ! How are you today ? I can't wait to teach you mathematics !"


dialogue_history = DialogueHistory(
    "prompt_format_config.json", initial_system_prompt=sentence_0, context_size=100
)
dialogue_history.append_utterance(
    {"turn_id": 1, "text": sentence_1, "speaker": "agent"}
)
dialogue_history.append_utterance({"turn_id": 2, "text": sentence_2, "speaker": "user"})
# dialogue_history.append_utterance({"turn_id": 3, "text": sentence_3, "speaker": "user"})
dialogue_history.prepare_dialogue_history(model.tokenize)


###########

### REPEAT

print("\n\nREPEAT\n\n")

print(dialogue_history.get_prompt())

repeat_system_prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. \
You play the role of a teacher, and your last sentence has been interrupted by the child, please repeat the last teacher sentence. \
Here is the beginning of the conversation :"
system_prompt = dialogue_history.change_system_prompt(repeat_system_prompt)
prompt = dialogue_history.get_prompt()
print(prompt)

# prompt = "[INST] <<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. \
# You play the role of a teacher, and your last sentence has been interrupted by the child. Before the interruption you were supposed to say : 'Hello ! How are you today ? I can't wait to teach you mathematics !'. Please provide the next teacher sentence by intergrating the interrupted sentence into your next sentence. \
# Here is the beginning of the conversation, before the interruption:<</SYS>>\
# \n\nTeacher: Hello ! How are you to...\
# \n\nChild: Hi![/INST]"


# prompt_tokens = model.tokenize(bytes(prompt, encoding="utf-8"))
# print(prompt)

# pattern = bytes("\n\n", encoding="utf-8")
# pattern_tokens = model.tokenize(pattern, add_bos=False)
# print(pattern_tokens)
# print(model.tokenize(bytes("\n\nChild:", encoding="utf-8"), add_bos=False))
# print(model.tokenize(bytes("Child:", encoding="utf-8"), add_bos=False))
# print(model.tokenize(bytes("Child: ", encoding="utf-8"), add_bos=False))

# # prompt, prompt_tokens = dialogue_history.prepare_dialogue_history(model.tokenize)
# # print(prompt)


# sentence_tokens = []
# try:
#     for t in model.generate(
#         prompt_tokens,
#         top_k=40,
#         top_p=0.95,
#         temp=1.0,
#         repeat_penalty=1.1,
#     ):
#         sentence_tokens.append(t)
#         sentence = model.detokenize(sentence_tokens)
#         if sentence_tokens[-4:-2] == [13, 13]:
#             print(len(sentence_tokens))
#             print(-len(pattern_tokens))
#             print(len(sentence))
#             print(-len(pattern))
#             print(sentence_tokens[-len(pattern_tokens) :])
#             print(sentence[-len(pattern) :])
#         if pattern_tokens == sentence_tokens[-len(pattern_tokens) :]:
#             break
#         if pattern == sentence[-len(pattern) :]:
#             break
# except Exception as e:
#     print(e)

# print(sentence_tokens)
# sentence = model.detokenize(sentence_tokens)

# print(sentence)
# dialogue_history.append_utterance(
#     {
#         "turn_id": 5,
#         "speaker": "agent",
#         "text": sentence.decode("utf-8"),
#     }
# )
# dialogue_history.change_system_prompt(system_prompt)
# prompt = dialogue_history.get_prompt()
# # prompt, prompt_tokens = dialogue_history.prepare_dialogue_history(model.tokenize)

# print(prompt)


### RELANCE
# prompt = "[INST] <<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
# The teacher is teaching mathematics to the child student. \
# As the student is a child, the teacher needs to stay gentle all the time. \
# You play the role of a teacher, and your last sentence 'Great! Are you ready to learn mathematics ?' had no answer from the child. Please provide a next teacher sentence that would re-engage the child in the conversation. \
# Here is the beginning of the conversation : <</SYS>>\
# \n\nTeacher: Hello ! How are you today ?\
# \n\nChild: Hi! I am fine.\
# \n\nTeacher: Great! Are you ready to learn mathematics ? [/INST]"
prompt = "[INST] <<SYS>>This is a spoken dialog scenario between a teacher and a 8 years old child student. \
The teacher is teaching mathematics to the child student. \
As the student is a child, the teacher needs to stay gentle all the time. \
You play the role of a teacher, and your last sentence 'Great, so what is the addition of two and two ?' had no answer from the child. Please provide a next teacher sentence that would re-engage the child in the conversation. \
Here is the beginning of the conversation : <</SYS>>\
\n\nTeacher: Addition is when you add two number together to have a bigger number. For example, the addition of one and two is three. Do you understand ?\
\n\nChild: Yes I think I do.\
\n\nTeacher: Great, so what is the addition of two and two ? \
\n\nChild: ... [/INST]"
prompt_tokens = model.tokenize(bytes(prompt, encoding="utf-8"))
pattern = bytes("\n\n", encoding="utf-8")
pattern_tokens = model.tokenize(pattern, add_bos=False)

sentence_tokens = []
try:
    for t in model.generate(
        prompt_tokens,
        top_k=40,
        top_p=0.95,
        temp=1.0,
        repeat_penalty=1.1,
    ):
        sentence_tokens.append(t)
        sentence = model.detokenize(sentence_tokens)
        if pattern_tokens == sentence_tokens[-len(pattern_tokens) :]:
            break
        if pattern == sentence[-len(pattern) :]:
            break
except Exception as e:
    print(e)

sentence = model.detokenize(sentence_tokens)
print(sentence)
