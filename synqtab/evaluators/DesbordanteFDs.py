from synqtab.evaluators.Evaluator import Evaluator
from synqtab.utils import get_logger


LOG = get_logger(__name__)


class DesbordanteFDs(Evaluator):
    """ Desbordante Functional Dependency Discovery. Leverages
    https://github.com/Desbordante/desbordante-core. Parameters:
        - [*required*] `'data'`: the data to perform FD discovery on
        - [*optional*] `'notes'`: True/False on whether to include notes in the result or not.
        If absent, defaults to False.
    """
    
    def short_name(self):
        from synqtab.enums import EvaluationMethod
        return str(EvaluationMethod.DFD)
    
    def full_name(self):
        return "Desbordante Functional Dependencies Discovery"
    
    def compute_result(self):
        import os
        import desbordante as db
        
        try:
            data = self.params.get('data')
            if len(data.columns) > 100:
                return None

            # Load data from pandas DataFrame
            pyro_alg = db.fd.algorithms.Default()
            pyro_alg.load_data(table=data)

            LOG.info("Data loaded into FD discovery algorithm.")

            pyro_alg.execute()
            LOG.info("Executed FD discovery algorithm.")

            # Collect functional dependencies
            fds = [str(fd) for fd in pyro_alg.get_fds()]
            
            if self.params.get('notes', False):
                return len(fds), {'FDs': fds}
            return len(fds)

        finally:
            # Remove log file created by desbordante if it exists
            if os.path.exists('myeasylog.log'):
                os.remove('myeasylog.log')
