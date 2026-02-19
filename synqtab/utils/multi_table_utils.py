import subprocess
import tempfile
import os
from synqtab.configs.MinioSettings import MinioBucket
from synqtab.data.clients.MinioClient import MinioClient
import pandas as pd

MULTI_DATASETS = [
    'airbnb-recruiting-new-user-bookings',
    'rossmann-store-sales'
]


def download_kaggle_competition(competition_name: str) -> str:
    """
    Download a Kaggle competition dataset and save to a temporary directory.
    
    Args:
        competition_name: The Kaggle competition name (e.g., 'rossmann-store-sales')
    
    Returns:
        str: Path to the temporary directory containing downloaded files
    """
    temp_dir = tempfile.mkdtemp()
    
    try:
        subprocess.run(
            ['kaggle', 'competitions', 'download', '-c', competition_name],
            cwd=temp_dir,
            check=True,
            capture_output=True
        )
        return temp_dir
    except subprocess.CalledProcessError as e:
        print(f"Error downloading competition: {e.stderr.decode()}")
        raise
    except FileNotFoundError:
        print("Kaggle CLI not found. Please install it with: pip install kaggle")
        raise

def extract_zip_file(zip_file_path: str, extract_to: str) -> None:
    """
    Extract a zip file to a specified directory.
    
    Args:
        zip_file_path: Path to the zip file
        extract_to: Directory where the contents should be extracted
    """
    import zipfile
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    os.remove(zip_file_path)

def clean_up_temp_dir(temp_dir: str) -> None:
    """
    Remove the temporary directory and its contents.
    
    Args:
        temp_dir: Path to the temporary directory to be removed
    """
    import shutil
    
    shutil.rmtree(temp_dir)

def process_rossman(file_path: str):
    minio_client = MinioClient()

    for file_name in ['store.csv', 'test.csv', 'train.csv']:
        csv_path = os.path.join(file_path, file_name)
        df = pd.read_csv(csv_path)
        
        # Fix mixed type columns - convert StateHoliday to string
        if 'StateHoliday' in df.columns:
            df['StateHoliday'] = df['StateHoliday'].astype(str)

        minio_client.upload_dataframe_as_parquet_to_bucket(
            df = df,
            bucket_name='multitable',
            object_name=f'data/rossmann/{file_name.replace(".csv", ".parquet")}'
        )

def process_airbnb(file_path: str):
    minio_client = MinioClient()

    airbnb_files = ['age_gender_bkts.csv.zip', 'train_users_2.csv.zip', 'sessions.csv.zip', 'countries.csv.zip']

    for zip_file in airbnb_files:
        zip_path = os.path.join(file_path, zip_file)
        extract_zip_file(zip_path, file_path)
        
        csv_file = zip_file.replace('.zip', '')
        csv_path = os.path.join(file_path, csv_file)
        df = pd.read_csv(csv_path)
        
        minio_client.upload_dataframe_as_parquet_to_bucket(
            df=df,
            bucket_name=MinioBucket.MULTI.value,
            object_name=f'data/airbnb/{csv_file.replace(".csv", ".parquet")}'
        )
        
        os.remove(csv_path)

def upload_data_pipeline():
    for dataset in MULTI_DATASETS:
        print(f"Processing dataset: {dataset}")
        temp_dir = download_kaggle_competition(dataset)
        
        zip_file_path = os.path.join(temp_dir, f"{dataset}.zip")
        extracted_path = os.path.join(temp_dir, dataset)

        extract_zip_file(zip_file_path, extracted_path)
        
        if dataset == 'rossmann-store-sales':
            process_rossman(extracted_path)
        elif dataset == 'airbnb-recruiting-new-user-bookings':
            process_airbnb(extracted_path)

        clean_up_temp_dir(temp_dir)

def get_data_dict(dataset:str):
    if dataset not in MULTI_DATASETS:
        raise ValueError(f"Dataset '{dataset}' is not supported. Supported datasets: {MULTI_DATASETS}")

    minio_client = MinioClient()

    if dataset == 'rossmann-store-sales':
        store_dataframe = pd.DataFrame()
        train_dataframe = pd.DataFrame()
            
        for file in minio_client.list_bucket_objects(MinioBucket.MULTI.value, "data/rossmann"):
            file_name = file.get("Key")
                
            if "store" in file_name:
                store_dataframe = minio_client.read_parquet_from_bucket(MinioBucket.MULTI.value, file_name)

            if "train" in file_name:
                train_dataframe = minio_client.read_parquet_from_bucket(MinioBucket.MULTI.value, file_name)

        data = {
            'store': store_dataframe,
            'train': train_dataframe
        }
        
        return data

    elif dataset == 'airbnb-recruiting-new-user-bookings':
        age_gender_dataframe = pd.DataFrame()
        train_dataframe = pd.DataFrame()
        sessions_dataframe = pd.DataFrame()
        countries_dataframe = pd.DataFrame()

        for file in minio_client.list_bucket_objects(MinioBucket.MULTI.value, "data/airbnb"):
            file_name = file.get("Key")
                
            if "age_gender" in file_name:
                age_gender_dataframe = minio_client.read_parquet_from_bucket(MinioBucket.MULTI.value, file_name)

            if "countries" in file_name:
                countries_dataframe = minio_client.read_parquet_from_bucket(MinioBucket.MULTI.value, file_name)
                
            if "sessions" in file_name:
                sessions_dataframe = minio_client.read_parquet_from_bucket(MinioBucket.MULTI.value, file_name)

            if "train" in file_name:
                train_dataframe = minio_client.read_parquet_from_bucket(MinioBucket.MULTI.value, file_name)
            
        data = {
            "age_gender": age_gender_dataframe,
            "countries": countries_dataframe,
            "sessions": sessions_dataframe,
            "train": train_dataframe
        }
        
        return data



def create_metadata():
    from sdv.metadata import Metadata
    for dataset in MULTI_DATASETS:
        data_dict = get_data_dict(dataset)
        metadata = Metadata.detect_from_dataframes(data = data_dict)
        metadata.save_to_json(f"{dataset}_metadata.json")


def upload_metadata_pipeline():
    import json
    minio_client = MinioClient()
    for dataset in MULTI_DATASETS:
        with open(f"{dataset}_metadata.json", 'r') as f:
            metadata_dict = json.load(f)
        
        # Upload the dict to MinIO with the correct filename
        minio_client.upload_json_to_bucket(
            data=metadata_dict,
            bucket_name=MinioBucket.MULTI.value,
            folder="metadata",
            file_name=f"{dataset}.json"
        )


def fetch_metadata(dataset: str):
    from sdv.metadata import Metadata
    if dataset not in MULTI_DATASETS:
        raise ValueError(f"Dataset '{dataset}' is not supported. Supported datasets: {MULTI_DATASETS}")

    minio_client = MinioClient()
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, f"{dataset}_metadata.json")
        
    minio_client.download_file_from_bucket(
        bucket_name=MinioBucket.MULTI.value,
        object_name=f"metadata/{dataset}.json",
        local_file_path=temp_file_path
    )
        
    # Load the metadata as an SDV Metadata object
    metadata = Metadata.load_from_json(temp_file_path)
        
    clean_up_temp_dir(temp_dir)
        
    return metadata


def get_synthetic_data(path: str) -> dict:
    """
    Read synthetic dataframes from MinIO based on a path string.
    
    Args:
        path: A path string like 'MFK#rossmann-store-sales#16840#PERF#hma'
              which will be converted to 'data/MFK/rossmann-store-sales/16840/PERF/hma'
    
    Returns:
        dict: A dictionary where keys are table names and values are DataFrames
    """
    minio_client = MinioClient()
    
    # Convert path: replace # with / and prepend data/
    minio_path = "data/" + path.replace("#", "/")

    data = {}
    
    for file in minio_client.list_bucket_objects(MinioBucket.SYNTHETIC.value, minio_path):
        file_name = file.get("Key")
        print(file_name)
        # Extract table name from file path (e.g., 'store' from '.../store.parquet')
        table_name = os.path.basename(file_name).replace(".parquet", "")
        data[table_name] = minio_client.read_parquet_from_bucket(
            MinioBucket.SYNTHETIC.value, 
            file_name
        )            
    
    return data