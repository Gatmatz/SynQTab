import os
from dotenv import load_dotenv


load_dotenv()
MINIO_ROOT_USER = os.getenv('MINIO_ROOT_USER')
MINIO_ROOT_PASSWORD = os.getenv('MINIO_ROOT_PASSWORD')
MINIO_API_MAPPED_PORT = os.getenv('MINIO_API_MAPPED_PORT')
MINIO_UI_MAPPED_PORT = os.getenv('MINIO_UI_MAPPED_PORT')
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT')
MINIO_HOST = os.getenv('MINIO_HOST')
