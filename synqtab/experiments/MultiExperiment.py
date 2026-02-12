from synqtab.utils import get_logger

LOG = get_logger(__file__)


class MultiExperiment():
    _NULL: str = 'NULL'
    _delimiter: str = '#'
    
    def __init__(
        self,
        dataset: str,
        generator,
        drop_unknown_references: bool,
        data_error_type,
        data_error_rate,
        data_perfectness,
        evaluation_methods
    ):
        self.dataset = dataset
        self.generator = generator
        self.drop_unknown_references = drop_unknown_references
        self.data_error = data_error_type
        self.data_error_rate = data_error_rate
        self.data_perfectness = data_perfectness
        self.evaluation_methods = evaluation_methods
    
    @classmethod
    def short_name(cls):
        from synqtab.enums import ExperimentType
        return str(ExperimentType.MULTI)

    def _get_experiment_id_parts(self):
        from synqtab.reproducibility.ReproducibleOperations import ReproducibleOperations
        return [
            str(self.short_name()),  # Experiment shortname, e.g., 'NOR' for Normal Experiment
            str(self.dataset),  # Dataset name, e.g., 'anneal'
            str(ReproducibleOperations.get_current_random_seed()),  # Random seed
            str(self.data_perfectness), # Data perfectness level, e.g., 'PERF' for perfect
            str(self.data_error) if self.data_error else self._NULL,    # Data error type, e.g., 'OUT' for outliers
            str(int(self.data_error_rate * 100)) if self.data_error_rate else self._NULL, # Data error rate multiplied by 100, e.g., 0.2 -> 20 -> '20'
            str(self.generator),   # Generator type, e.g., 'tabpfn' 
        ]
    
    def __str__(self):
        experiment_id_parts = self._get_experiment_id_parts()
        return self._delimiter.join(experiment_id_parts)
    
    def minio_path(self):
        from synqtab.enums import MinioFolder
        
        experiment_id_parts = self._get_experiment_id_parts()
        return MinioFolder.create_prefix(
            MinioFolder.DATA, *experiment_id_parts, ignore=self._NULL
        )

    def compute_training_size(self, df_dict):
        training_size = {table_name: len(df) for table_name, df in df_dict.items()}
        return training_size

    def _run(self) -> None:
        from synqtab.data import PostgresClient, MinioClient
        from synqtab.enums import MinioBucket
        from synqtab.mappings.mappings import GENERATOR_MODEL_TO_GENERATOR_INSTANCE
        from synqtab.reproducibility.ReproducibleOperations import ReproducibleOperations
        from synqtab.utils import timed_computation
        from synqtab.utils.multi_table_utils import get_data_dict, fetch_metadata
        from sdv.utils import drop_unknown_references

        LOG.info(f"Entering the _run() function of Multi Experiment {str(self)}")
        
        real_perfect_df = get_data_dict(dataset = self.dataset)
        metadata = fetch_metadata(dataset=self.dataset)

        # Set the primary key column for the dataset, which is needed for the data corruption step.
        if self.dataset == 'rossmann-store-sales':
            self.pk_column = 'Store'

        # todo: if we need to sample the data we should use sdv and poc

        # If we are in pollution version then start with perfect data and then we will add errors
        if not self.drop_unknown_references:
            real_perfect_df = drop_unknown_references(real_perfect_df, metadata, drop_missing_values=True)
        

        corrupted_rows = corrupted_cols = []
        if self.data_error:
            if self.data_error_rate:
                data_error_instance =  self.data_error.get_class()(row_fraction=self.data_error_rate, columns= [self.pk_column])
                corrupted_df = {}
                for table_name, table_df in real_perfect_df.items():
                    if table_name == 'store':
                        corrupted_df[table_name] = table_df
                    else:
                        corrupted_table, corrupted_rows, corrupted_cols = data_error_instance.corrupt(table_df)
                        corrupted_df[table_name] = corrupted_table
                LOG.info(f"Data Corruption was completed successfully for experiment {str(self)}")
                
        
        df = drop_unknown_references(corrupted_df if self.data_error else real_perfect_df, metadata, drop_missing_values=self.drop_unknown_references)
        LOG.info(f"Foreign key cleaning completed for experiment {str(self)}")

        LOG.info(f"Initializing {self.generator} generator for experiment {str(self)}")

        generator_instance = GENERATOR_MODEL_TO_GENERATOR_INSTANCE.get(self.generator)

        synthetic_df, elapsed_time = timed_computation(
            computation=generator_instance.generate,
            params={
                'data': df,
                'metadata': metadata,
                'scale': 1.0
            }
        )
        LOG.info(f"Generation for experiment {str(self)} was completed in {elapsed_time} seconds.")
        
        for table_name, table_df in synthetic_df.items():
            # Action 1: Write the Synthetic data to MinIO for asynchronous evaluation
            MinioClient.upload_dataframe_as_parquet_to_bucket(
                df=table_df,
                bucket_name=MinioBucket.SYNTHETIC,
                object_name=f"{self.minio_path()}/{table_name}"
            )
        LOG.info(f"Successfully wrote the synthetic data of experiment {str(self)} to MinIO '{self.minio_path()}'.")
        
        # Action 2: Write experiment metadata to Postgres for offline analysis
        PostgresClient.write_experiment(
            experiment_id=str(self),
            experiment_type=self.short_name(),
            dataset_name=self.dataset,
            random_seed=str(ReproducibleOperations.get_current_random_seed()),
            data_perfectness=str(self.data_perfectness),
            data_error=str(self.data_error) if self.data_error else None,
            error_rate=str(int(self.data_error_rate * 100)) if self.data_error_rate else None,
            generator=str(self.generator),
            training_size=str(self.compute_training_size(real_perfect_df)),
            synthetic_size=str(self.compute_training_size(synthetic_df)),
            corrupted_rows=corrupted_rows,
            corrupted_cols=corrupted_cols,
            execution_time=elapsed_time,
        )
        LOG.info(f"Successfully wrote the metadata of experiment {str(self)} to Postgres.")
