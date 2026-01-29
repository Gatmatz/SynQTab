# Kaggle Utils Manual

## Overview

The `kaggle_utils.py` module provides utilities for executing Python scripts on Kaggle's platform, with support for multiple profiles and concurrent execution.


## Quick Start

### 0. Prerequisites
`uv pip install kaggle` (or `pip install kaggle` if you are not using uv)

### 1. Single Script Execution

Execute a single Python script on Kaggle:

```python
from synqtab.utils.kaggle_utils import execute_single_script

execute_single_script(
    script_path="my_script.py",
    username="your_kaggle_username",
    title="My Experiment",
    enable_gpu=True,
    enable_internet=True,
    is_private=True
)
```

### 2. Batch Execution (Single Profile)

**YAML Configuration** (`kaggle_config.yaml`):
```yaml
username: your_kaggle_username
title_prefix: experiment
enable_gpu: true
enable_internet: true
is_private: true

scripts:
  - scripts/experiment1.py
  - scripts/experiment2.py
  - scripts/experiment3.py
```

**Note**: scripts can be anywhere in our project structure. No need to be on folder `synqtab/utils/`.
**Python Code**:

```python
from synqtab.utils.kaggle_utils import run_kaggle_scripts_from_yaml

result = run_kaggle_scripts_from_yaml(
    yaml_path='kaggle_config.yaml',
    max_concurrent=5,      # Run 5 scripts simultaneously
    max_retries=3,         # Retry failed scripts 3 times
    check_interval=30      # Check status every 30 seconds
)
```

### 3. Multi-Profile Execution (Recommended) (Not tested yet)

**YAML Configuration** (`kaggle_config_multi.yaml`):

```yaml
profiles:
  - name: profile1
    credential_name: bilpapster  # Must match the username in bilpapster.json
    max_concurrent: 5
    scripts:
      - scripts/experiment1.py
      - scripts/experiment2.py
  
  - name: profile2
    credential_name: georgeatmatzidis  # Must match the username in georgeatmatzidis.json
    max_concurrent: 3
    scripts:
      - scripts/experiment3.py
      - scripts/experiment4.py

common_settings:
  title_prefix: experiment
  enable_gpu: true
  enable_internet: true
  is_private: true
```

**Python Code**:

```python
from synqtab.utils.kaggle_utils import run_kaggle_scripts_multi_profile

result = run_kaggle_scripts_multi_profile(
    yaml_path='kaggle_config_multi.yaml',
    max_retries=3,
    check_interval=30
)
```

---

## Credential Management

### Setting Up Credentials

1. Create credentials directory:
   ```bash
   mkdir -p synqtab/utils/kaggle_credentials/
   ```

2. Add credential files:
   ```
   synqtab/utils/kaggle_credentials/
   ├── bilpapster.json
   └── georgeatmatzidis.json
   ```
   
   **IMPORTANT**: The credential file names **must match** the Kaggle username in the file. 
   For example, if the username is "bilpapster", the file must be named `bilpapster.json`.

3. Each credential file should contain:
   ```json
   {
     "username": "your_kaggle_username",
     "key": "your_kaggle_api_key"
   }
   ```

**Note**: When you generate a new API key, select "Create Legacy API Key" to get the correct .json format.

### Switching Credentials

```python
from synqtab.utils.kaggle_utils import set_kaggle_credentials

# Switch to different account
set_kaggle_credentials("bilpapster")
```
**Note**: For multi-profile execution, credentials are switched automatically.

---

## Status Monitoring

### Check Kernel Status

```python
from synqtab.utils.kaggle_utils import get_kaggle_kernel_status

status = get_kaggle_kernel_status("experiment-profile1-0", "your_kaggle_username")
print(status)
# {
#     'kernel_ref': 'username/experiment-profile1-0',
#     'raw_output': '...',
#     'is_complete': False,
#     'has_error': False,
#     'is_running': True
# }
```

### Discord Notifications

The system automatically sends Discord notifications for:
- Script completion
- Script failures (with retry info)
- Batch summary

Configure Discord webhook in `.env`:
```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

---

## Troubleshooting

### Common Issues

- **Kaggle Weekly Limits**: Because I could not replicate the exact message when surpassing the limits, there is no specific handling for it.

## API Reference

### Core Functions

#### `execute_single_script(script_path, username, title, enable_gpu, enable_internet, is_private)`
Execute a single script on Kaggle.

#### `run_kaggle_scripts_from_yaml(yaml_path, max_concurrent, max_retries, check_interval)`
**DEPRECATED**: Use `run_kaggle_scripts_multi_profile` instead.

#### `run_kaggle_scripts_multi_profile(yaml_path, max_retries, check_interval)`
Execute scripts across multiple Kaggle profiles with concurrent execution.

#### `get_kaggle_kernel_status(kernel_slug, username)`
Get the current status of a running Kaggle kernel.

#### `set_kaggle_credentials(credential_name)`
Switch to a different set of Kaggle credentials. The `credential_name` must match the Kaggle username.

## Environment Variables

Required in `.env`:

```bash
# Discord notifications (optional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

---

