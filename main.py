import uuid

from sparrow import Client

if __name__ == '__main__':
    client = Client()  # uses environment variable for config
    # client = Client(config=Config(api_key='vicky-api-key'))
    model_reference = str(uuid.uuid4())
    finetune_job_id = client.create_finetune_job(model_reference, 'male', [
        'https://something1',
        'https://something2',
        'https://something3'
    ], 5000)
    print(finetune_job_id)
    finetune_job_status, finetune_job_progress = client.get_job_status(finetune_job_id)
    print(f"Finetune job: {finetune_job_id}, status: {finetune_job_status}, progress: {finetune_job_progress}")
    inference_job_id = client.create_inference_job(
        model_reference,
        'My positive prompt', 'My negative prompt', 150, 6.5)
    print(inference_job_id)
    inference_job_status, inference_job_progress = client.get_job_status(inference_job_id)
    print(f"Inference job: {inference_job_id}, status: {inference_job_status}, progress: {inference_job_progress}")
    image_urls = client.get_generated_image_urls(inference_job_id)  # returns empty array because job progress < 1.0
    print(f"Image urls: {image_urls}")
