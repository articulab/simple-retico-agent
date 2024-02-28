"""
Conda environment to activate before running the code : llama_cpp_cuda
conda activate llama_cpp_cuda
"""

from llama_cpp import Llama
# from llama_cpp_python import Llama

# Instanciate the model
# MODEL_PATH = "./model/zephyr-7b-beta.Q4_0.gguf"
MODEL_PATH = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"

CONTEXT_SIZE = 512
N_GPU_LAYERS = 100

def generate_text_from_prompt(
    my_prompt,
    my_model,
    max_tokens = 100,
    temperature = 0.3,
    top_p = 0.1,
    echo = True,
    stop = ["Q", "\n"],
    ):

    # Define the parameters
    model_output = my_model(
        my_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
    )
    return model_output

if __name__ == "__main__":

    my_model = Llama(model_path=MODEL_PATH, n_ctx=CONTEXT_SIZE, n_gpu_layers=N_GPU_LAYERS)
   
#    my_prompt = "What do you think about the inclusion policies in Tech companies?"
#    chat_history = [
#         {"role": "user", "content": "Hello"}, 
#         {"role": "assistant", "content": "Hello! I am your math teacher, you are a 8 years old student. This is a dialog during which I will teach you how to add two numbers together."},
#     ]
    # prefix = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
    #     The teacher is teaching mathemathics to the child student. \
    #     As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. You play the role of a teacher."
    # prompt_without_prefix = "Teacher : Hi! How are your today ? \
    # Child : I am fine, and I can't wait to learn mathematics !"
    # my_prompt = "<s>[INST] This is a spoken dialog scenario between a teacher and a 8 years old child student. \
    # The teacher is teaching mathemathics to the child student. \
    # As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
    # You play the role of a teacher. Here is the beginning of the conversation : [/INST] \
    # Teacher : Hi! How are your today ? \
    # Child : I am fine, and I can't wait to learn mathematics !</s>\
    # [INST] Generate the next Teacher sentence [/INST]"
    my_prompt = "[INST]This is a spoken dialog scenario between a teacher and a 8 years old child student. \
    The teacher is teaching mathemathics to the child student. \
    As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation.\
    You play the role of a teacher. Here is the beginning of the conversation : \
    Teacher : Hi! How are your today ? \
    Child : I am fine, and I can't wait to learn mathematics ![/INST]"
    
    model_output = generate_text_from_prompt(my_prompt, my_model)
    print(model_output)
    text = model_output["choices"][0]["text"].strip()
    print(text)