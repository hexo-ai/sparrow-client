import os
from typing import List, Tuple, Dict

from requests import Session

CONFIGURATION_LOCATION = 'SPARROW_CONFIG'


class Config:
    def __init__(self, **kwargs):
        self.api_key = kwargs.get('api_key')


class Client:
    def __init__(self, **kwargs):
        self.api_key = None
        if 'config' in kwargs:
            self.api_key = kwargs.get('config', {}).api_key
        elif CONFIGURATION_LOCATION in os.environ:
            config_file_path = os.environ[CONFIGURATION_LOCATION]
            if os.path.isfile(config_file_path):
                with open(config_file_path) as f:
                    variables = {}
                    exec(f.read(), variables)
                    self.api_key = variables.get('API_KEY')
        if self.api_key is None:
            raise RuntimeError("Missing API key")
        self.base_url = 'https://sparrow.messio.com/api/v1'
        self.session = Session()
        self.session.headers.update({
            'accept': 'application/json',
            'x-api-key': self.api_key
        })

    @staticmethod
    def check_response(res) -> Dict:
        if not res.ok:
            raise RuntimeError(f'API call failed: {res.status_code}')
        data = res.json()
        if 'error' in data:
            raise RuntimeError(f"Error: {data['error']}")
        return data

    def create_finetune_job(self, model_reference: str,
                            gender: str, image_urls: List[str], max_train_steps: int) -> str:
        payload = {
            'model_reference': model_reference,
            'image_urls': image_urls,
            'gender': gender,
            'max_train_steps': max_train_steps,
        }
        res = self.session.post(f'{self.base_url}/finetune-job', json=payload)
        data = Client.check_response(res)
        return data.get('finetune_job_id')

    def get_finetune_job_status(self, finetune_job_id: str) -> Tuple[str, float]:
        res = self.session.get(f'{self.base_url}/finetune-job-status/{finetune_job_id}')
        data = Client.check_response(res)
        return data.get('status'), data.get('progress')

    def create_inference_job(self, model_reference: str, prompt: str, negative_prompt: str, num_inference_steps: int,
                             guidance_scale: float):
        payload = {
            'model_reference': model_reference,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'num_inference_steps': num_inference_steps,
            'guidance_scale': guidance_scale,
        }
        res = self.session.post(f'{self.base_url}/inference-job', json=payload)
        data = Client.check_response(res)
        return data.get('inference_job_id')

    def get_inference_job_status(self, inference_job_id: str):
        res = self.session.get(f'{self.base_url}/inference-job-status/{inference_job_id}')
        data = Client.check_response(res)
        return data.get('status'), data.get('progress')

    def get_generated_image_urls(self, inference_job_id: str):
        res = self.session.get(f'{self.base_url}/generated-images/{inference_job_id}')
        data = Client.check_response(res)
        return data.get('image_urls', [])
