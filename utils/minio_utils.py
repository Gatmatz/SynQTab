from enum import Enum
import os
from typing import Optional, List, Dict, Any

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

from logging_utils import get_logger

LOG = get_logger(__file__)


class MinioBucket(Enum):
    REAL = 'real'
    SYNTHETIC = 'synthetic'
    
class MinioFolder(Enum):
    PERFECT = 'perfect'
    IMPERFECT = 'imperfect'
    DATA = 'data'
    METADATA = 'metadata'
    


def get_minio_client() -> BaseClient:
    """
    Create and return a boto3 s3 client configured for MinIO using environment variables.
    """
    load_dotenv()
    
    # Fetch credentials with sensible defaults or None
    endpoint = os.getenv("MINIO_HOST", "http://localhost")
    port = os.getenv("MINIO_API_MAPPED_PORT", "9000")
    access_key = os.getenv("MINIO_ROOT_USER")
    secret_key = os.getenv("MINIO_ROOT_PASSWORD")
    
    # Construct the full endpoint URL
    # Handle cases where HOST might already include http/https or port
    if not endpoint.startswith("http"):
        endpoint = f"http://{endpoint}"
    
    endpoint_url = f"{endpoint}:{port}"

    try:
        client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        return client
    except Exception:
        LOG.exception("Failed to create MinIO client.")
        raise


def get_existing_buckets(client: Optional[BaseClient] = None) -> List[str]:
    """
    Return a list of bucket names that exist in the MinIO instance.
    """
    if client is None:
        client = get_minio_client()

    try:
        response = client.list_buckets()
        buckets = [bucket["Name"] for bucket in response.get("Buckets", [])]
        LOG.info(f"Retrieved {len(buckets)} buckets.")
        return buckets
    except ClientError:
        LOG.exception("Failed to list buckets.")
        raise


def ensure_bucket_exists(bucket_name: str, client: Optional[BaseClient] = None) -> None:
    """
    Check if a bucket exists, and create it if it does not.
    """
    if client is None:
        client = get_minio_client()

    try:
        client.head_bucket(Bucket=bucket_name)
        LOG.info(f"Bucket '{bucket_name}' already exists.")
    except ClientError:
        # The bucket does not exist or we lack access; attempt creation
        try:
            client.create_bucket(Bucket=bucket_name)
            LOG.info(f"Created bucket '{bucket_name}'.")
        except ClientError:
            LOG.exception(f"Failed to create bucket '{bucket_name}'.")
            raise


def list_bucket_objects(
    bucket_name: str, 
    prefix: str = "", 
    client: Optional[BaseClient] = None
) -> List[Dict[str, Any]]:
    """
    List objects in a specific bucket. Returns a list of dictionaries containing metadata 
    (Key, Size, LastModified, etc.).
    """
    if client is None:
        client = get_minio_client()

    try:
        response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        contents = response.get("Contents", [])
        LOG.info(f"Found {len(contents)} objects in '{bucket_name}' with prefix '{prefix}'.")
        return contents
    except ClientError:
        LOG.exception(f"Failed to list objects in bucket '{bucket_name}'.")
        raise


def upload_file_to_bucket(
    local_file_path: str,
    bucket_name: str,
    object_name: Optional[str] = None,
    client: Optional[BaseClient] = None,
) -> None:
    """
    Upload a file to a MinIO bucket.

    - `local_file_path`: Path to the file to upload.
    - `object_name`: S3 object name. If not specified, the file_name is used.
    """
    if client is None:
        client = get_minio_client()
        
    local_file_path = str(local_file_path)

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(local_file_path)

    LOG.info(f"Attempting to upload file: {local_file_path}")
    try:
        client.upload_file(local_file_path, bucket_name, object_name)
        LOG.info(f"Uploaded '{local_file_path}' to '{bucket_name}/{object_name}'.")
    except FileNotFoundError:
        LOG.error(f"The file '{local_file_path}' was not found.")
        raise
    except (ClientError, NoCredentialsError):
        LOG.exception(f"Failed to upload '{local_file_path}' to '{bucket_name}'.")
        raise


def download_file_from_bucket(
    bucket_name: str,
    object_name: str,
    local_file_path: str,
    client: Optional[BaseClient] = None,
) -> None:
    """
    Download a file from a MinIO bucket to a local path.
    """
    if client is None:
        client = get_minio_client()

    try:
        # Ensure local directory exists
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        client.download_file(bucket_name, object_name, local_file_path)
        LOG.info(f"Downloaded '{bucket_name}/{object_name}' to '{local_file_path}'.")
    except ClientError:
        LOG.exception(f"Failed to download object '{object_name}' from bucket '{bucket_name}'.")
        raise
