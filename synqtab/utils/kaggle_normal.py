from synqtab.utils.kaggle_utils import run_kaggle_scripts_multi_profile

result = run_kaggle_scripts_multi_profile(
    yaml_path='../configs/kaggle_config_multi_profile.yaml',
    max_retries=3,
    check_interval=30
)

from pprint import pp
pp(result)
