import os
import tempfile
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from db_utils import write_dataframe_to_db
from minio_utils import get_minio_client, ensure_bucket_exists, upload_file_to_bucket, MinioBucket, MinioFolder
from logging_utils import get_logger
from file_utils import read_yaml_file, read_files_from_directory

# --- Configuration ---
DATASETS_DIR = "../tabarena_datasets"
DB_SCHEMA = "tabarena_label_encoded"
# -------------------

LOG = get_logger(__file__)

def _curate_marketing_campaign(marketing_campaign_df: pd.DataFrame) -> pd.DataFrame:
    timestamp_column = 'Dt_Customer'
    if timestamp_column in marketing_campaign_df.columns:
        # Convert to datetime, then to integer (Unix timestamp in seconds)
        # Using errors='coerce' will turn unparseable dates into NaT (Not a Time)
        dt_series = pd.to_datetime(marketing_campaign_df[timestamp_column], format='%Y-%m-%d', errors='coerce')
        # Convert to Unix timestamp (seconds) and fill NaNs with a placeholder like 0
        marketing_campaign_df[timestamp_column] = (dt_series.astype('int64') // 10 ** 9).fillna(0).astype(int)
        return marketing_campaign_df


def _curate_qsar_tid_11(qsar_tid_11_df: pd.DataFrame) -> pd.DataFrame:
    # Convert binary columns to integer, keep MEDIAN_PXC50 as float with reduced precision
    for col in qsar_tid_11_df.columns:
        
        if col == 'MEDIAN_PXC50':
            qsar_tid_11_df['MEDIAN_PXC50'] = qsar_tid_11_df['MEDIAN_PXC50'].round(6)
            continue
        
        unique_values = qsar_tid_11_df[col].dropna().unique()
        if set(unique_values).issubset({0.0, 1.0}):
            qsar_tid_11_df[col] = qsar_tid_11_df[col].astype('int8')

DATASET_NAME_TO_CURATION_FUNCTION = {
    'Marketing_Campaign': _curate_marketing_campaign,
    'QSAR-TID-11': _curate_qsar_tid_11,
}
DATASETS_REQUIRING_SPECIAL_CURATION = DATASET_NAME_TO_CURATION_FUNCTION.keys()


def _load_dataset_to_postgres(name: str, df: pd.DataFrame, yaml_path: str) -> None:
    metadata = read_yaml_file(yaml_path)
    
    # 1. Write processed DataFrame to DB
    # Sanitize table name for SQL
    table_name = name.replace('-', '_').replace(' ', '_').lower()
    LOG.info(f"Writing data to table `{DB_SCHEMA}.{table_name}`...")
    # write_dataframe_to_db(df, table_name=table_name, schema=DB_SCHEMA, row_by_row=True)

    # 2. Write YAML metadata to a separate DB table
    meta_df = pd.DataFrame(list(metadata.items()), columns=['meta_key', 'meta_value'])
    # Convert lists/dicts in meta_value to strings for DB compatibility
    meta_df['meta_value'] = meta_df['meta_value'].apply(str)
    meta_table_name = f"{table_name}_meta"
    LOG.info(f"Writing metadata to table `{DB_SCHEMA}.{meta_table_name}`...")
    # write_dataframe_to_db(meta_df, table_name=meta_table_name, schema=DB_SCHEMA)
    
    
def _load_dataset_to_minio(name: str, df: pd.DataFrame, yaml_path: str) -> None:
    """
    Serializes a DataFrame to a temporary Parquet file and uploads it to MinIO.
    Also, uploads the raw YAML metadata file as-is.
    """
    # Initialize client once to reuse connection
    client = get_minio_client()
    
    # 0. Ensure target buckets exist
    # access .value to get the string representation of the bucket name
    bucket = MinioBucket.REAL.value
    data_folder = f"{MinioFolder.PERFECT.value}/{MinioFolder.DATA.value}"
    metadata_folder = f"{MinioFolder.PERFECT.value}/{MinioFolder.METADATA.value}"

    ensure_bucket_exists(bucket, client=client)

    # 1. Write DataFrame to MinIO as Parquet
    # We use a temporary file to save the Parquet, upload it, then delete it.
    LOG.info("Trying to get a temporary file name")
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        df.to_parquet(tmp.name, index=False)
        tmp_path = tmp.name
    
    LOG.info("Found a temporary file name.")

    try:
        data_object_key = f"{data_folder}/{name}.parquet"
        
        LOG.info(f"Uploading data to MinIO: {bucket}/{data_object_key}")
        upload_file_to_bucket(
            local_file_path=tmp_path, 
            bucket_name=bucket, 
            object_name=data_object_key, 
            client=client
        )
        LOG.info(f"Uploading data to MinIO: {bucket}/{data_object_key}")
    finally:
        # Clean up the local temporary Parquet file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 2. Write Metadata YAML to MinIO
    # We upload the source YAML file directly to the metadata bucket
    # We ensure the object matches the dataset name for easy correlation
    meta_object_key = f"{metadata_folder}/{name}.yaml"
    
    LOG.info(f"Uploading metadata to MinIO: {bucket}/{meta_object_key}")
    upload_file_to_bucket(
        local_file_path=yaml_path,
        bucket_name=bucket, 
        object_name=meta_object_key, 
        client=client
    )
    LOG.info(f"Upload of data and metadata completed for dataset: {name}")

TARGET_TO_LOAD_FUNCTION = {
    'db': _load_dataset_to_postgres,
    'minio': _load_dataset_to_minio,
}


def process_dataset(name: str, csv_path: str, yaml_path: str) -> pd.DataFrame:
    # 1. Read YAML to get metadata and categorical features
    metadata = read_yaml_file(yaml_path)

    categorical_features = metadata.get('categorical_features', [])
    if not categorical_features:
        LOG.warning(f"No `categorical_features` found in yaml for dataset: {name}")

    if metadata.get('problem_type', None) == 'classification':
        target_feature = metadata.get('target_feature', None)
        if target_feature:
            categorical_features.append(metadata.get('target_feature', ''))
        else:
            LOG.error(f"No target feature found in yaml for classification dataset: {name}")

    # 2. Read CSV data
    df = pd.read_csv(csv_path)

    # 3. Label Encode categorical features
    if categorical_features:
        le = LabelEncoder()
        for col in categorical_features:
            if col in df.columns:
                # Ensure column is of string type before encoding to handle mixed types
                df[col] = le.fit_transform(df[col].astype(str))
            else:
                LOG.error(f"Column `{col}` from yaml not found in csv for dataset: {name}")
      
    # 4. If needed, perform dataset-specific, additional curation steps          
    if name in DATASETS_REQUIRING_SPECIAL_CURATION:
        df = DATASET_NAME_TO_CURATION_FUNCTION[name](df)
        
    return df


def process_and_load_datasets(target="db"):
    """
    Finds dataset pairs (.csv, .yaml) in DATASETS_DIR, processes them,
    and loads them into the database.
    
    :param target: 'db' for writing (loading) to Postgres, 'minio' for minio; case insensitive.
    """
    LOG.info(f"Starting dataset processing from directory: `{DATASETS_DIR}`")
    try:
        files = read_files_from_directory(DATASETS_DIR)
    except FileNotFoundError:
        LOG.error(f"Not found directory: `{DATASETS_DIR}`. Please create it and add your datasets and yaml files.")
        return
    
    # Find all unique dataset names (without extension)
    dataset_names = sorted(list(set([os.path.splitext(f)[0] for f in files])))
    
    target = target.lower()
    if target not in TARGET_TO_LOAD_FUNCTION:
        LOG.error(f"Unknown target: {target}. Was expecting one of 'db' or 'minio' (case insensitive). Falling back to 'db'.")
    load_function = TARGET_TO_LOAD_FUNCTION.get(target, TARGET_TO_LOAD_FUNCTION['db'])

    for name in dataset_names:
        csv_path = os.path.join(DATASETS_DIR, f"{name}.csv")
        yaml_path = os.path.join(DATASETS_DIR, f"{name}.yaml")

        # Check if both files for a dataset exist
        if not (os.path.exists(csv_path) and os.path.exists(yaml_path)):
            LOG.warning(f"Skipping dataset: {name}. Missing .csv or .yaml file.")
            continue

        LOG.info(f"Starting processing dataset: {name}")
        try:
            df = process_dataset(name, csv_path, yaml_path)
            load_function(name, df, yaml_path)
            LOG.info(f"Finished processing dataset: {name}")

        except Exception as e:
            LOG.error(f"Failed to process and load dataset: {name}. Error: {type(e)}, {str(e)}")

    LOG.info("Finished processing of all datasets.")


if __name__ == "__main__":
    # Note: Ensure your .env file is configured with your database credentials.
    # The script assumes a schema named 'tabarena' exists in your database.
    # You might need to create it first: CREATE SCHEMA tabarena;
    process_and_load_datasets('minio')