import ollama
import dotenv 
import os

from ModelClient import ModelClient

CONFIGS_PATH = "D:\\LLM_labs\\llm-26\\projects\\ti-rudinskiy\\lab1\\source\\configs"
dotenv.load_dotenv(os.path.join(CONFIGS_PATH, '.env'))
OLLAMA_URL = os.environ.get('OLLAMA_URL')
PROMPTS_PATH = os.environ.get('PROMPTS_PATH')

client = ollama.Client(OLLAMA_URL)
qwen = ModelClient('qwen2.5:3b', CONFIGS_PATH, client)
phi = ModelClient('phi3:3.8b', CONFIGS_PATH, client)
mistral = ModelClient('ministral-3:3b', CONFIGS_PATH, client)
for i, role in enumerate(['classifier', 'extractor']):
    with open(PROMPTS_PATH + '/' + os.listdir(PROMPTS_PATH)[i+1], 'r', encoding='utf-8') as file:
        txt = file.read()
        # print(f'Model: qwen, role: default, output:')
        # qwen.task = 'default'
        # print(qwen.call(txt))
        # print(f'Model: qwen, role: {role}, output:')
        # qwen.task = role
        # print(qwen.call(txt))
        print(f'Model: phi, role: default, output:')
        phi.task = 'default'
        print(phi.call(txt))
        print(f'Model: phi, role: {role}, output:')
        phi.task = role
        print(phi.call(txt))
        # print(f'Model: ministral, role: default, output:')
        # mistral.task = 'default'
        # print(mistral.call(txt))
        # print(f'Model: ministral, role: {role}, output:')
        # mistral.task = role
        # print(mistral.call(txt))
