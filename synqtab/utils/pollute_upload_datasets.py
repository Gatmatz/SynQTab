from synqtab.data.clients.MinioClient import MinioClient
from synqtab.environment.experiment import RANDOM_SEEDS
from synqtab.configs.MinioSettings import MinioBucket, MinioFolder


real_data_bucket = MinioBucket.REAL.value
perfect_real_data_folder = MinioFolder.create_path(
    MinioFolder.PERFECT, MinioFolder.DATA
)

real_data_files = MinioClient.list_bucket_objects(real_data_bucket, perfect_real_data_folder)

for real_data_file in real_data_files:
    file_key = real_data_file.get('Key')
    # print(real_data_file.get('Key'))
    df = MinioClient.read_parquet_from_bucket(real_data_bucket, file_key)
    print(df.head(2))
    print()



