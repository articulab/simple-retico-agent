"""
Conda environment to activate before running the code : llama
conda activate llama
"""

from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline

model = "./models/mistral-7b-instruct-v0.2.Q4_K_S.gguf"

m = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    model_file="mistral-7b-instruct-v0.2.Q4_K_S.gguf",
    model_type="mistral", gpu_layers=50, hf=True
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")



def generate_next_sentence(
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


prompt = "This is a spoken dialog scenario between a teacher and a 8 years old child student. \
    The teacher is teaching mathemathics to the child student. \
    As the student is a child, the teacher needs to stay gentle all the time. Please provide the next valid response for the followig conversation. You play the role of a teacher.\
    Here is the dialog: \
    Teacher : Hi! How are your today ? \
    Child : I am fine, and I can't wait to learn mathematics !"

# PROMPT WITHOUT PIPE
output = generate_next_sentence(prompt, m)
print(output)

# PIPE
# pipe = pipeline(model=m, tokenizer=tokenizer, task='text-generation')
# sequences = pipe(
#     prompt,
#     # prefix=prefix,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     # eos_token_id=tokenizer.eos_token_id,
#     max_new_tokens=1000,
#     # max_length=100,
#     # min_length=20,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")