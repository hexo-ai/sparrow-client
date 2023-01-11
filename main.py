import uuid

from sparrow import Client, Config

if __name__ == '__main__':
    client = Client()
    # client = Client(config=Config(api_key='my-api-key'))
    model_reference = str(uuid.uuid4())
    finetune_job_id = client.create_finetune_job(model_reference, 'male', [
        'https://something1',
        'https://something2',
        'https://something3'
    ], 5000)
    print(finetune_job_id)
    finetune_job_status = client.get_finetune_job_status(finetune_job_id)
    print(finetune_job_status)
