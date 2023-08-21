from llama_cpp import Llama
# import torch

# print(torch.cuda.is_available())
# print(torch.__version__)

# from IPython.display import display, HTML
import json
import time
import pathlib

model_path = r"C:\Users\capek\pycharm_projects\Transformers_for_czech_language\llama\codeup-llama-2-13b-chat-hf.ggmlv3.q2_K.bin"
# model_path = r"C:\Users\capek\pycharm_projects\Transformers_for_czech_language\llama\codeup-llama-2-13b-chat-hf.ggmlv3.q6_K.bin"
# model_path = r"C:\Users\capek\pycharm_projects\Transformers_for_czech_language\llama\llama-2-7b-chat.ggmlv3.q8_0.bin"

MODEL_Q8_0 = Llama(
    #model_path="./llama/codeup-llama-2-13b-chat-hf.ggmlv3.q6_K.bin",
    model_path=model_path,
    n_ctx=500,
    n_gpu_layers=-1,
    use_mlock=True
)

def query(model, question):
    model_name = pathlib.Path(model.model_path).name
    time_start = time.time()
    prompt = f"Q: {question} A:"
    output = model(prompt=prompt, max_tokens=50) # if max tokens is zero, depends on n_ctx
    response = output["choices"][0]["text"]
    time_elapsed = time.time() - time_start
    # display(HTML(f'<code>{model_name} response time: {time_elapsed:.02f} sec</code>'))
    # display(HTML(f'<strong>Question:</strong> {question}'))
    # display(HTML(f'<strong>Answer:</strong> {response}'))
    print(f'<code>{model_name} response time: {time_elapsed:.02f} sec</code>')
    print(f'Question:{question}')
    print(f'Answer: {response}')
    print(json.dumps(output, indent=2))

query(MODEL_Q8_0, """Tell me if the following sentence talks about dog. Return a json format and nothing else: {'answer': True if about dogs, False otherwise}
Sentence: Včera jsem byl na procházce se psem, zatímco kočka zůstala doma.}""")