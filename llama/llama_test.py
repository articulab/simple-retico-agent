import transformers
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# model = "meta-llama/Llama-2-7b-hf" # too big, needs 16GB GPU

# Running with 8GB GPU
# model = "Aryanne/Mistral-3B-Instruct-v0.2-init" # not pre trained
# model = "typeof/mistral-3.3B" # do not generate anything, needs to be pretrained ?
# model = "princeton-nlp/Sheared-LLaMA-1.3B" # work with auto and pipeline
model = "openlm-research/open_llama_3b_v2" # works with every thing, but what works better ?
# work with auto and none auto tokenizer. 
# work with none auto and pipeline (kinda)
# work with none auto and generate
# work with auto and generate
# work with auto and pipeline

access_token = "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"

print(torch.cuda.is_available())

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    # bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_compute_dtype=getattr(torch,"float16"),
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# device_map = {"":0}
device_map = "auto"

# NOT AUTO FOR OPENLLAMA
# tokenizer = LlamaTokenizer.from_pretrained(model, access_token=access_token, legacy=False)
# m = LlamaForCausalLM.from_pretrained(
#     model,
#     device_map=device_map,
#     quantization_config=bnb_config
#     # load_in_4bit=True,
# )

# # AUTO
tokenizer = AutoTokenizer.from_pretrained(model, access_token=access_token)
m = AutoModelForCausalLM.from_pretrained(
    model,
    device_map=device_map,
    quantization_config=bnb_config
)


# NO PIPELINE
# # # prompt = 'Q: What is the largest animal?\nA:'
# # # prompt = 'Q: who will win in a fight opposing a crocodile to a grizzly ?\nA:'
# prompt = 'Who will win in a fight opposing a crocodile to a grizzly ? Explain step by step.'
# inputs = tokenizer(prompt, return_tensors="pt")
# input_ids = inputs.input_ids
# input_ids = input_ids.to('cuda')

# generation_output = m.generate(
#     input_ids=input_ids,
#     max_new_tokens=100
# )
# print(tokenizer.decode(generation_output[0]))


# PIPELINE

# prompt = "The settings is a mathematics tutoring session between a teacher and a 8 years old child. \
# The exchange is a dialogue, you are the teacher and I am the child. \
# Stay gentle all the time. \
# Here is the beginning of the conversation, only generate the next teacher sentence : \
# Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics !"

prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
    The teacher is teaching mathemathics to the child student. \
    As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. You play the role of a teacher.\
    Here is the dialog: \
    Teacher : Hi! How are your today ? \
    Child : I am fine, and I can't wait to learn mathematics !"

# prefix = "The settings is a mathematics tutoring session between a teacher and a 8 years old child. \
# The exchange is a dialogue, you are the teacher and I am the child. \
# Stay gentle all the time."
# prefix = "This is a conversation between a teacher and a 8 years old child student. \
#     The teacher is teaching mathemathics to the child student. \
#     As the student is a child, the teacher needs to stay gentle all the time."
# prompt_without_prefix = "Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics !"

# prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'

# prompt = 'hi ! how are you today ?'

# prompt = 'Who will win in a fight opposing a crocodile to a grizzly ? Explain step by step.'

text_generation_pipeline = transformers.pipeline(
    "text-generation",
    model=m,
    torch_dtype=torch.float16,
    device_map=device_map,
    token=access_token,
    tokenizer=tokenizer
)

sequences = text_generation_pipeline(
    prompt,
    # prefix=prefix,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=1000,
    # max_length=100,
    # min_length=20,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")



# # conversational pipeline
# from transformers import Conversation

# prefix = "This is a conversation between a teacher and a 8 years old child student. \
#     The teacher is teaching mathemathics to the child student. \
#     As the student is a child, the teacher needs to stay gentle all the time."
# prompt_without_prefix = "Teacher : Hi! How are your today ? \
# Child : I am fine, and I can't wait to learn mathematics !"



# ## usage of chat template
# # chat = [
# #    {"role": "user", "content": "Hello, how are you?"},
# #    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
# #    {"role": "user", "content": "I'd like to show off how chat templating works!"},
# # ]
# # tokenizer.apply_chat_template(chat, tokenize=False)

# # usage of chat template
# messages = [
#     {
#         "role": "system",
#         "content": "You are a friendly chatbot who always responds in the style of a pirate",
#     },
#     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
#  ]
# tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
# print(tokenizer.decode(tokenized_chat[0]))
# outputs = m.generate(tokenized_chat, max_new_tokens=128) 
# print("OUT\n")
# print(tokenizer.decode(outputs[0]))

# # # usage of chat template
# # messages = [
# #     {
# #         "role": "system",
# #         "content": "You are a friendly chatbot who always responds in the style of a pirate",
# #     },
# #     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
# # ]
# # print(pipe(messages, max_new_tokens=128)[0]['generated_text'][-1])  # Print the assistant's response




# # conversational_pipeline = transformers.pipeline(
# #     "conversational",
# #     # model=model,
# #     model=m,
# #     torch_dtype=torch.float16,
# #     device_map=device_map,
# #     token=access_token,
# #     tokenizer=tokenizer
# # )

# # chatbot = conversational_pipeline(
# #     prompt_without_prefix,
# #     prefix=prefix,
# #     do_sample=True,
# #     top_k=10,
# #     num_return_sequences=1,
# #     eos_token_id=tokenizer.eos_token_id,
# #     max_new_tokens=1000,
# #     # max_length=100,
# #     # min_lengt
# #     # h=20,
# # )

# # conversation = Conversation("I'm looking for a movie - what's your favourite one?")
# # conversation = conversational_pipeline(conversation)
# # print(conversation.messages[-1]["content"])
# # conversation.add_message({"role": "user", "content": "That's interesting, why do you like action movies?"})
# # conversation = conversational_pipeline(conversation)
# # print(conversation.messages[-1]["content"])
