import datetime
import time
import torch
from llama_cpp import Llama

from vad_turn_2 import DialogueHistory

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
As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the following conversation.\
You play the role of a teacher. Here is the beginning of the conversation :"
sentence_1 = "Hello !"
sentence_2 = "Hi! How are your today ?"
sentence_3 = "I am fine, and I can't wait to learn mathematics !"


dialogue_history = DialogueHistory(
    "prompt_format_config.json", initial_system_prompt=sentence_0, context_size=100
)

print(dialogue_history.get_dialogue_history())
print(dialogue_history.get_prompt())
print(dialogue_history.get_stop_patterns())

dialogue_history.append_utterance({"turn_id": 1, "text": sentence_1, "speaker": "user"})

print(dialogue_history.get_dialogue_history())
print(dialogue_history.get_prompt())

dialogue_history.append_utterance(
    {"turn_id": 2, "text": sentence_2, "speaker": "agent"}
)

print(dialogue_history.get_dialogue_history())
print(dialogue_history.get_prompt())

dialogue_history.append_utterance({"turn_id": 3, "text": sentence_3, "speaker": "user"})

print(dialogue_history.get_dialogue_history())
print(dialogue_history.get_prompt())

dialogue_history.prepare_dialogue_history(model.tokenize)
