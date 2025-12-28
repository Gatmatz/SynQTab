import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

MINIO_ROOT_USER = os.getenv('MINIO_ROOT_USER')
MINIO_ROOT_PASSWORD = os.getenv('MINIO_ROOT_PASSWORD')
MINIO_HOST = os.getenv('MINIO_HOST')
MINIO_API_MAPPED_PORT = os.getenv('MINIO_API_MAPPED_PORT')
BUCKET_NAME = "test-bucket"
FOLDER_PATH = "example.d"

s3 = boto3.client(
    "s3",
    aws_access_key_id=MINIO_ROOT_USER,
    aws_secret_access_key=MINIO_ROOT_PASSWORD,
    endpoint_url=f"{MINIO_HOST}:{MINIO_API_MAPPED_PORT}",
)

def create_bucket(bucket_name):
    try:
        s3.create_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' created successfully.")
    except s3.exceptions.BucketAlreadyExists as e:
        print(f"Bucket already exists: {e}")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        print(f"Bucket '{bucket_name}' already owned by you.")

def upload_folder(bucket_name, folder_path):
    try:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                key = os.path.relpath(file_path, start=folder_path).replace("\\", "/")
                s3.upload_file(file_path, bucket_name, key)
                print(f"Uploaded {file_path} as {key}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except NoCredentialsError:
        print("AWS credentials not found.")
    except PartialCredentialsError:
        print("Incomplete AWS credentials provided.")

def list_bucket_contents(bucket_name):
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        if "Contents" in response:
            print(f"\nContents of bucket '{bucket_name}':")
            for obj in response["Contents"]:
                print(f" - {obj['Key']} (Size: {obj['Size']} bytes)")
        else:
            print(f"\nBucket '{bucket_name}' is empty.")
    except s3.exceptions.NoSuchBucket:
        print(f"Bucket '{bucket_name}' does not exist.")
    except NoCredentialsError:
        print("AWS credentials not found.")
    except PartialCredentialsError:
        print("Incomplete AWS credentials provided.")

if __name__ == "__main__":
    create_bucket(BUCKET_NAME)
    upload_folder(BUCKET_NAME, FOLDER_PATH)
    list_bucket_contents(BUCKET_NAME)
