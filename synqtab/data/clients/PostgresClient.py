from typing import Any, Optional

from sqlalchemy import create_engine

from synqtab.environment.postgres import (
    POSTGRES_USER, POSTGRES_PASSWORD,
    POSTGRES_MAPPED_PORT, POSTGRES_HOST, POSTGRES_DB
)
from synqtab.utils.logging_utils import get_logger


LOG = get_logger(__file__)


class SingletonPostgresClient(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonPostgresClient, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class _PostgresClient:
    _engine = create_engine(
        url = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_MAPPED_PORT}/{POSTGRES_DB}",
        echo=False,
        pool_pre_ping=True,
    )
    

class PostgresClient(_PostgresClient, metaclass=SingletonPostgresClient):
    
    @classmethod
    def execute_insert_query(
        cls,
        table_name: str,
        query_params: dict[str, Any],
    ):
        from sqlalchemy import text
        from synqtab.environment import EXECUTION_PROFILE
        
        query_params['execution_profile'] = EXECUTION_PROFILE
        field_names_list = list(query_params.keys())
        value_indicators_list = [':' + field_name for field_name in field_names_list]
        
        field_names = ', '.join(field_names_list)
        value_indicators = ', '.join(value_indicators_list)
        
        query = text(f"""INSERT INTO {table_name} ({field_names}) VALUES ({value_indicators})""")
        with cls._engine.connect() as connection:
            connection.execute(query, query_params)
            connection.commit()
            
    @classmethod
    def write_skipped_computation(
        cls,
        computation_id: str,
        reason: str,
        skipped_computations_table_name: str='skipped_computations',
    ) -> None:
        try:
            query_params = {
                "computation_id": computation_id,
                "reason": reason,
            }
            cls.execute_insert_query(table_name=skipped_computations_table_name, query_params=query_params)
            LOG.info(f"Wrote skipped computation {computation_id} in '{skipped_computations_table_name}'")
        except Exception as e:
            LOG.error(f"Failed to write skipped computation {computation_id}. Error: {e}")
            raise
    
    @classmethod
    def write_runtime_error(
        cls,
        experiment_id: str,
        file_path: str,
        error_message: str,
        errors_table_name: str = 'errors'
    ):
        try:
            query_params = {
                "experiment_id": experiment_id,
                "file_path": file_path,
                "error_message": error_message
            }
            cls.execute_insert_query(table_name=errors_table_name, query_params=query_params)
            LOG.info(f"Wrote runtime error for experiment {experiment_id} in '{errors_table_name}'")
        except Exception as e:
            LOG.error(f"Failed to write runtime error for experiment {experiment_id}. Error: {e}")
            raise
        
    @classmethod
    def write_experiment(
        cls,
        experiment_id: str,
        experiment_type: str,
        dataset_name: str,
        random_seed: str,
        data_perfectness: str,
        data_error: Optional[str],
        error_rate: Optional[str],
        generator: str,
        training_size: int,
        synthetic_size: int,
        execution_time: float,
        corrupted_rows: list = [],
        corrupted_cols: list = [],
        experiment_results_table_name: str = 'experiments',
    ):
        try:
            query_params = {
                'experiment_id': experiment_id,
                'experiment_type': experiment_type,
                'dataset_name': dataset_name,
                'random_seed': random_seed,
                'data_perfectness': data_perfectness,
                'data_error': data_error,
                'error_rate': error_rate,
                'generator': generator,
                'training_size': training_size,
                'synthetic_size': synthetic_size,
                'execution_time': execution_time,
                'corrupted_rows': corrupted_rows,
                'corrupted_cols': corrupted_cols,   
            }
            cls.execute_insert_query(table_name=experiment_results_table_name, query_params=query_params)
            LOG.info(f"Wrote experiment {experiment_id} in '{experiment_results_table_name}'")
        except Exception as e:
            LOG.error(f"Failed to write experiment {experiment_id}. Error: {e}")
            raise
        
        
    @classmethod
    def write_evaluation_result(
        cls,
        evaluation_id: str,
        first_target: str,
        second_target: str,
        result: int | float,
        execution_time: float,
        notes: Optional[dict[str, Any]] = None,
        evaluation_results_table_name: str = 'evaluations'
    ):
        try:
            query_params = {
                "evaluation_id": evaluation_id,
                "first_target": first_target,
                "second_target": second_target,
                "result": result,
                "execution_time": execution_time,
                "notes": notes if notes else None,
            }
            cls.execute_insert_query(table_name=evaluation_results_table_name, query_params=query_params)
            LOG.info(f"Wrote evaluation result {evaluation_id} in '{evaluation_results_table_name}'")
        except Exception as e:
            LOG.exception(f"Failed to write evaluation result for experiment {evaluation_id}. Error: {e}")
            raise
    
    @classmethod
    def evaluation_result_exists(
        cls, 
        evaluation_id: str, 
        evaluation_results_table_name: str = 'evaluation_results'
    ) -> bool:
        """Checks if an evaluation result with the specific evaluation id exists.

        Args:
            experiment_id (str): The evaluation id to check for existence.

        Returns:
            bool: True if it exists, else False.
        """
        from sqlalchemy import text
        try:
            query = text(f"""
                SELECT 1 FROM {evaluation_results_table_name} \
                WHERE experiment_id = :experiment_id \
                LIMIT 1 
            """)
            with cls._engine.connect() as connection:
                result = connection.execute(query, {"experiment_id": evaluation_id})
                exists = result.scalar() is not None
                LOG.info(f"Checked existence of evaluation {evaluation_id}: {exists}")
                return exists 
        except Exception as e:
            LOG.exception(
                f"Failed to check existence of evaluation {evaluation_id}. Error: {e}")
            raise
        
    @classmethod
    def experiment_exists(
        cls, 
        experiment_id: str, 
        experiments_table_name: str = 'experiments',
        experiment_id_column_name: str = 'experiment_id',
    ) -> bool:
        """Checks if an experiment with the specific experiment id exists.

        Args:
            experiment_id (str): The experiment id to check for existence.

        Returns:
            bool: True if it exists, else False.
        """
        from sqlalchemy import text
        try:
            query = text(f"""
                SELECT 1 FROM {experiments_table_name} \
                WHERE {experiment_id_column_name} = :experiment_id \
                LIMIT 1 
            """)
            with cls._engine.connect() as connection:
                result = connection.execute(query, {"experiment_id": experiment_id})
                exists = result.scalar() is not None
                LOG.info(f"Checked existence of evaluation {experiment_id}: {exists}")
                return exists 
        except Exception as e:
            LOG.error(f"Failed to check existence of experiment {experiment_id}. Error: {e}")
            raise
        
    @classmethod
    def evaluation_exists(
        cls,
        evaluation_id: str,
        experiment_id: str, 
        evaluations_table_name: str = 'evaluations',
    ) -> bool:
        """Checks if an experiment with the specific experiment id exists.

        Args:
            experiment_id (str): The experiment id to check for existence.

        Returns:
            bool: True if it exists, else False.
        """
        from sqlalchemy import text
        try:
            query = text(f"""
                SELECT 1 FROM {evaluations_table_name} \
                WHERE evaluation_id = :evaluation_id AND experiment_id = :experiment_id \
                LIMIT 1 
            """)
            with cls._engine.connect() as connection:
                result = connection.execute(query, {"experiment_id": experiment_id, "evaluation_id": evaluation_id})
                exists = result.scalar() is not None
                LOG.info(f"Checked existence of evaluation {evaluation_id} for experiment {experiment_id}: {exists}")
                return exists 
        except Exception as e:
            LOG.error(f"Failed to check existence of evaluation {evaluation_id} for experiment {experiment_id}. Error: {e}")
            raise
