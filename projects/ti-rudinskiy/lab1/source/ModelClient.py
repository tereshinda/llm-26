import yaml
import os
import ollama

class ModelClient:

    def __init__(self, model: str, config_path: str, client:ollama.Client):
        self.config_path = config_path
        self.model = model
        self.client = client

    @property
    def task(self):
        return self._curr_task

    @task.setter
    def task(self, model_type: str):
        if model_type == 'default':
            self.options = None
            self.system = None
            return 
        
        path = os.path.join(self.config_path, f'{model_type}.yaml')
        
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

            self.system = data['system']
            self.options = data['options']

    def call(self, message:str):
        m = [{
            'role': 'user',
            'content': message,
        }]
        
        res = self.client.chat(
            model=self.model,
            messages=m,
            stream=False,
            options=self.options
        )
        
        return {
            "answer": res.message.content,
            "prompt_eval_count": res.prompt_eval_count,
            "eval_count": res.eval_count,
            "total_duration": res.total_duration / 1e9,
            "prompt_eval_speed": res.prompt_eval_count / res.prompt_eval_duration * 1e9,
            "eval_speed": res.eval_count / res.eval_duration * 1e9,
        }