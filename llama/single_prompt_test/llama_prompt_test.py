"""
Conda environment to activate before running the code : llama
conda activate llama
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

DEVICE_MAP = "auto"
PROMPT_TYPE = "conversational"
# PROMPT_TYPE = "text-generation"

def main():

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        # bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_compute_dtype=getattr(torch,"float16"),
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    if PROMPT_TYPE == "text-generation":

        MODEL_PATH = "openlm-research/open_llama_3b_v2"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map=DEVICE_MAP,
            quantization_config=bnb_config
        )

        prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
        The teacher is teaching mathemathics to the child student. \
        As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. You play the role of a teacher.\
        Here is the dialog: \
        Teacher : Hi! How are your today ? \
        Child : I am fine, and I can't wait to learn mathematics !"

        time_0 = time.time()

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids
        input_ids = input_ids.to(model.device)

        output_tokens = model.generate(
            input_ids=input_ids,
            repetition_penalty=1.05,
            max_new_tokens=100
        )
        output_text = tokenizer.decode(output_tokens[0])

        time_1 = time.time()
        print("["+str(round(time_1 - time_0, 3)) + "s] " + output_text)
        # print(output_text)

    if PROMPT_TYPE == "conversational":

        MODEL_PATH = "mediocredev/open-llama-3b-v2-chat"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map=DEVICE_MAP,
            quantization_config=bnb_config
        )

        chat_history = [
            {"role": "user", "content": "Hello"}, 
            {"role": "assistant", "content": "Hello! I am your math teacher, you are a 8 years old student. This is a dialog during which I will teach you how to add two numbers together."},
            {"role": "user", "content": "Okay, I can't wait to learn about mathematics !"},
        ]
        # prompt = Conversation(chat_history)

        time_0 = time.time()

        input_ids = tokenizer.apply_chat_template(
            chat_history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        output_tokens = model.generate(
            input_ids,
            repetition_penalty=1.05,
            max_new_tokens=1000,
        )
        output_text = tokenizer.decode(
            output_tokens[0][len(input_ids[0]) :], skip_special_tokens=True
        )

        time_1 = time.time()
        print("["+str(round(time_1 - time_0, 3)) + "s] " + output_text)
        # print(output_text)

if __name__ == '__main__':
    main()


# ## usage of chat template
# # chat = [
# #    {"role": "user", "content": "Hello, how are you?"},
# #    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
# #    {"role": "user", "content": "I'd like to show off how chat templating works!"},
# # ]
# # tokenizer.apply_chat_template(chat, tokenize=False)

# # usage of chat template
# messages = [
#     {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate"},
#     {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
#  ]
# tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
# print(tokenizer.decode(tokenized_chat[0]))
# outputs = m.generate(tokenized_chat, max_new_tokens=128)
# print(tokenizer.decode(outputs[0]))


# # conversational pipeline
# from transformers import Conversation
    
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
