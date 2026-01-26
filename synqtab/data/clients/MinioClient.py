import io
import json
import os
from typing import Any, Optional
import yaml

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd

from synqtab.environment import (
    MINIO_ROOT_USER, MINIO_ROOT_PASSWORD,
    MINIO_API_MAPPED_PORT, MINIO_HOST,
)
from synqtab.utils import get_logger


LOG = get_logger(__file__)


class SingletonMinioClient(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMinioClient, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class _MinioClient:
    _client = boto3.client(
        "s3",
        endpoint_url=f"http://{MINIO_HOST}:{MINIO_API_MAPPED_PORT}",
        aws_access_key_id=MINIO_ROOT_USER,
        aws_secret_access_key=MINIO_ROOT_PASSWORD,
    )
    

class MinioClient(_MinioClient, metaclass=SingletonMinioClient):
    
    @classmethod
    def get_existing_buckets(cls) -> list[str]:
        try:
            response = cls._client.list_buckets()
            return [bucket["Name"] for bucket in response.get("Buckets", [])]
        except (ClientError, NoCredentialsError) as e:
            LOG.error(
                f"Failed to get the existing buckets. Error {e}."
            )
            raise
        
    @classmethod
    def ensure_bucket_exists(cls, bucket_name: str) -> None:
        try:
            cls._client.head_bucket(Bucket=bucket_name)
            LOG.info(f"Bucket '{bucket_name}' exists and is accessible.")
        except ClientError:
            try:
                LOG.info(f"Bucket '{bucket_name}' does not exist or is inaccessible. Attempting creation.")
                cls._client.create_bucket(Bucket=bucket_name)
                LOG.info(f"Created bucket '{bucket_name}'.")
            except ClientError as e:
                LOG.error(f"Failed to create bucket '{bucket_name}'. {e}")
                raise
    
    @classmethod
    def list_bucket_objects(cls, bucket_name: str, prefix: str = "") -> list[dict[str, Any]]:
        try:
            response = cls._client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            contents = response.get("Contents", [])
            LOG.info(f"Found {len(contents)} objects in '{bucket_name}' with prefix '{prefix}'.")
            return contents
        except ClientError as e:
            LOG.error(f"Failed to list objects in bucket '{bucket_name}'. {e}")
            raise
        
    @classmethod
    def list_files_in_bucket_by_file_extension(
        cls,
        file_extension: str,
        bucket_name: str,
        prefix: str="",
        include_extension: bool=False,
        txt_output_file: Optional[str]=None,
    ):
        objects = cls.list_bucket_objects(bucket_name=bucket_name, prefix=prefix)
        relevant_files = [
            os.path.splitext(os.path.basename(obj['Key']))[0]
            for obj in objects
            if obj['Key'].endswith(file_extension)
        ]
        LOG.info(f"Found {len(relevant_files)} files with {file_extension} \
            extension in bucket '{bucket_name}' and prefix {prefix}.")
        
        if not include_extension:
            relevant_files = [file_name.split('.')[0] for file_name in relevant_files]
            
        if not txt_output_file:
            return relevant_files
        
        with open(txt_output_file, 'w') as f:
            for file_name in relevant_files:
                f.write(f"{file_name}\n")

        LOG.info(f"Written {len(relevant_files)} file names to '{txt_output_file}'.")
        return relevant_files
    
    @classmethod
    def delete_file_from_bucket(cls, bucket_name: str, object_key: str) -> None:
        try:
            cls._client.delete_object(Bucket=bucket_name, Key=object_key)
            LOG.info(f"Successfully deleted file '{object_key}' from bucket '{bucket_name}'")
        except (ClientError, NoCredentialsError) as e:
            LOG.error(
                f"Failed to deleted file '{object_key}' from bucket '{bucket_name}'. Error {e}."
            )
            raise
        
    
    @classmethod
    def upload_file_to_bucket(
        cls, local_file_path: str, bucket_name: str, object_name: Optional[str]
    ) -> None:
        local_file_path = str(local_file_path)
        if not object_name:
            object_name = os.path.basename(local_file_path)

        LOG.info(f"Attempting to upload file: {local_file_path} to bucket: '{bucket_name}'")
        try:
            cls._client.upload_file(local_file_path, bucket_name, object_name)
            LOG.info(f"Uploaded '{local_file_path}' to '{bucket_name}/{object_name}'.")
        except FileNotFoundError:
            LOG.error(f"The file '{local_file_path}' was not found.")
            raise
        except (ClientError, NoCredentialsError):
            LOG.error(f"Failed to upload '{local_file_path}' to '{bucket_name}'.")
            raise
        
    @classmethod
    def download_file_from_bucket(
        cls, bucket_name: str, object_name: str, local_file_path: str
    ) -> None:
        try:
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            cls._client.download_file(bucket_name, object_name, local_file_path)
            LOG.info(f"Downloaded '{bucket_name}/{object_name}' to '{local_file_path}'.")
        except (ClientError, NoCredentialsError):
            LOG.error(f"Failed to download object '{object_name}' from bucket '{bucket_name}'.")
            raise
        
    @classmethod
    def read_parquet_from_bucket(
        cls, bucket_name: str, object_name: str, **pandas_kwargs
    ) -> pd.DataFrame:
        try:
            response = cls._client.get_object(Bucket=bucket_name, Key=object_name)
            df = pd.read_parquet(io.BytesIO(response['Body'].read()), **pandas_kwargs)
            LOG.info(f"Loaded Parquet from '{bucket_name}/{object_name}' into DataFrame with shape {df.shape}.")
            return df
        except (ClientError, NoCredentialsError):
            LOG.error(f"Failed to read Parquet from bucket '{bucket_name}'.")
            raise

    @classmethod
    def read_yaml_from_bucket(
        cls, bucket_name: str, object_name: str, **yaml_kwargs
    ) -> dict[str, Any]:
        try:
            response = cls._client.get_object(Bucket=bucket_name, Key=object_name)
            content = response['Body'].read().decode('utf-8')
            data = yaml.safe_load(content, **yaml_kwargs)
            LOG.info(f"Loaded YAML from '{bucket_name}/{object_name}'.")
            return data
        except (ClientError, NoCredentialsError):
            LOG.error(f"Failed to read YAML from bucket '{bucket_name}'.")
            raise

    @classmethod
    def upload_json_to_bucket(
        cls, data: dict[str, Any], bucket_name: str, folder: Optional[str], file_name: str,
    ) -> None:
        cls.ensure_bucket_exists(bucket_name=bucket_name)
        object_key = f"{folder}/{file_name}" if folder else file_name

        try:
            json_bytes = json.dumps(data, indent=2).encode('utf-8')
            cls._client.put_object(
                Bucket=bucket_name,
                Key=object_key,
                Body=json_bytes,
                ContentType='application/json'
            )
            LOG.info(f"Uploaded JSON to '{bucket_name}/{object_key}'.")
        except ClientError:
            LOG.error(f"Failed to upload JSON to '{bucket_name}/{object_key}'.")
            raise
    
    @classmethod
    def upload_dataframe_as_parquet_to_bucket(
        cls,
        df: pd.DataFrame,
        bucket_name: str,
        object_name: str,
    ) -> None:
        temp_file_path = None
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                df.to_parquet(tmp.name, index=False)
                temp_file_path = tmp.name
                MinioClient.upload_file_to_bucket(
                    local_file_path=temp_file_path,
                    bucket_name=bucket_name,
                    object_name=object_name,
                )
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
        
