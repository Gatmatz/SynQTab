from typing import Optional
from botocore.client import BaseClient
import os

from synqtab.utils.minio_utils import get_minio_client, list_bucket_objects, MinioBucket
from synqtab.utils.logging_utils import get_logger

LOG = get_logger(__file__)


def create_parquet_list(
        bucket_name: str | MinioBucket,
        output_file: str = "parquet_files.txt",
        prefix: str = "",
        client: Optional[BaseClient] = None,
) -> None:
    """
    List all .parquet files from a MinIO bucket and write their names to a text file.

    Args:
        bucket_name: Name of the bucket (string) or MinioBucket enum.
        output_file: Path to the output text file.
        prefix: Optional prefix to filter objects in the bucket.
        client: Optional MinIO client instance.
    """
    if client is None:
        client = get_minio_client()

    # Convert MinioBucket enum to string if needed
    bucket_str = bucket_name.value if isinstance(bucket_name, MinioBucket) else bucket_name

    # List all objects in the bucket
    objects = list_bucket_objects(bucket_str, prefix=prefix, client=client)

    # Filter only .parquet files and extract just the file name without extension
    parquet_files = [
        os.path.splitext(os.path.basename(obj['Key']))[0]
        for obj in objects
        if obj['Key'].endswith('.parquet')
    ]

    LOG.info(f"Found {len(parquet_files)} .parquet files in bucket '{bucket_str}'.")

    # Write file names to text file
    with open(output_file, 'w') as f:
        for file_name in parquet_files:
            f.write(f"{file_name}\n")

    LOG.info(f"Written {len(parquet_files)} file names to '{output_file}'.")


if __name__ == "__main__":
    # Example usage
    create_parquet_list(
        bucket_name=MinioBucket.REAL,
        output_file="../tabarena_list.txt"
    )
