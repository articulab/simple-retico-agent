import datetime
import time
import torch
from llama_cpp import Llama

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

print(prompt)

starttime = time.time()
startdate = datetime.datetime.now()
tokens = model.tokenize(prompt, add_bos=False)
endtime = time.time()
enddate = datetime.datetime.now()
print("time = ", endtime - starttime)
print("datetime = ", enddate - startdate)
print("len(tokens) = ", len(tokens))

starttime = time.time()
startdate = datetime.datetime.now()
tokens = model.tokenize(prompt, add_bos=False)
endtime = time.time()
enddate = datetime.datetime.now()
print("time = ", endtime - starttime)
print("datetime = ", enddate - startdate)
print("len(tokens) = ", len(tokens))

starttime = time.time()
startdate = datetime.datetime.now()
tokens = model.tokenize(prompt, add_bos=False)
endtime = time.time()
enddate = datetime.datetime.now()
print("time = ", endtime - starttime)
print("datetime = ", enddate - startdate)
print("len(tokens) = ", len(tokens))
