"""
Configuration management utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import copy


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load main configuration file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_experiments(experiments_path: str = "configs/experiments.yaml") -> Dict[str, Any]:
    """
    Load experiments configuration file.
    
    Args:
        experiments_path: Path to experiments YAML file
        
    Returns:
        Experiments dictionary
    """
    experiments_path = Path(experiments_path)
    
    if not experiments_path.exists():
        raise FileNotFoundError(f"Experiments file not found: {experiments_path}")
    
    with open(experiments_path, 'r') as f:
        experiments = yaml.safe_load(f)
    
    return experiments


def get_experiment_config(
    experiment_id: str,
    config_path: str = "configs/config.yaml",
    experiments_path: str = "configs/experiments.yaml"
) -> Dict[str, Any]:
    """
    Get complete configuration for a specific experiment.
    Merges base config with experiment-specific settings.
    
    Args:
        experiment_id: Experiment ID (e.g., 'exp01_densenet121_weighted_ce')
        config_path: Path to base config file
        experiments_path: Path to experiments config file
        
    Returns:
        Complete experiment configuration
    """
    # Load base config and experiments
    base_config = load_config(config_path)
    experiments = load_experiments(experiments_path)
    
    # Check if experiment exists
    if experiment_id not in experiments['experiments']:
        available = list(experiments['experiments'].keys())
        raise ValueError(
            f"Experiment '{experiment_id}' not found. "
            f"Available experiments: {available}"
        )
    
    # Get experiment settings
    exp_config = experiments['experiments'][experiment_id]
    model_name = exp_config['model']
    loss_name = exp_config['loss']
    
    # Deep copy base config
    merged_config = copy.deepcopy(base_config)
    
    # Add experiment metadata
    merged_config['experiment'] = {
        'id': experiment_id,
        'name': exp_config['name'],
        'model': model_name,
        'loss': loss_name,
        'description': exp_config.get('description', ''),
        'priority': exp_config.get('priority', 999)
    }
    
    # Merge model-specific settings
    if model_name in base_config['models']:
        model_config = base_config['models'][model_name]
        
        # Override batch_size if specified
        if 'batch_size' in model_config:
            merged_config['training']['batch_size'] = model_config['batch_size']
        
        # Override optimizer settings if specified
        if 'optimizer' in model_config:
            for key, value in model_config['optimizer'].items():
                merged_config['training']['optimizer'][key] = value
        
        # Add model-specific settings
        merged_config['model_config'] = model_config
    
    return merged_config


def get_model_and_loss(
    model_name: str,
    loss_name: str,
    config_path: str = "configs/config.yaml"
) -> Dict[str, Any]:
    """
    Get configuration for a specific model and loss combination.
    (Alternative to using experiment_id)
    
    Args:
        model_name: Model name (e.g., 'densenet121')
        loss_name: Loss name (e.g., 'focal')
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    base_config = load_config(config_path)
    merged_config = copy.deepcopy(base_config)
    
    # Add experiment info
    merged_config['experiment'] = {
        'model': model_name,
        'loss': loss_name,
        'name': f"{model_name} + {loss_name}"
    }
    
    # Merge model-specific settings
    if model_name in base_config['models']:
        model_config = base_config['models'][model_name]
        
        if 'batch_size' in model_config:
            merged_config['training']['batch_size'] = model_config['batch_size']
        
        if 'optimizer' in model_config:
            for key, value in model_config['optimizer'].items():
                merged_config['training']['optimizer'][key] = value
        
        merged_config['model_config'] = model_config
    
    return merged_config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def print_config(config: Dict[str, Any]):
    """
    Pretty print configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*70)
    print("EXPERIMENT CONFIGURATION")
    print("="*70)
    
    if 'experiment' in config:
        exp = config['experiment']
        print(f"\nExperiment: {exp.get('name', 'N/A')}")
        print(f"  ID: {exp.get('id', 'N/A')}")
        print(f"  Model: {exp.get('model', 'N/A')}")
        print(f"  Loss: {exp.get('loss', 'N/A')}")
        if 'description' in exp:
            print(f"  Description: {exp['description']}")
    
    print(f"\nData:")
    print(f"  Directory: {config['data']['data_dir']}")
    print(f"  Image size: {config['data']['image_size']}")
    print(f"  Num workers: {config['data']['num_workers']}")
    
    print(f"\nTraining:")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Optimizer: {config['training']['optimizer']['type']}")
    print(f"  Learning rate: {config['training']['optimizer']['lr']}")
    print(f"  Weight decay: {config['training']['optimizer']['weight_decay']}")
    print(f"  Scheduler: {config['training']['scheduler']['type']}")
    print(f"  Early stopping: {config['training']['early_stopping_patience']} epochs")
    
    if 'model_config' in config:
        print(f"\nModel Config:")
        mc = config['model_config']
        if 'pretrained' in mc:
            print(f"  Pretrained: {mc['pretrained']}")
        if 'dropout' in mc:
            print(f"  Dropout: {mc['dropout']}")
    
    print("="*70 + "\n")


# Test functions
if __name__ == "__main__":
    print("Testing config loading...\n")
    
    # Test 1: Load base config
    print("1. Loading base config...")
    config = load_config()
    print(f"   ✓ Loaded {len(config)} sections")
    
    # Test 2: Load experiments
    print("\n2. Loading experiments...")
    experiments = load_experiments()
    exp_list = list(experiments['experiments'].keys())
    print(f"   ✓ Loaded {len(exp_list)} experiments")
    print(f"   Experiments: {exp_list}")
    
    # Test 3: Get experiment config
    print("\n3. Getting experiment config...")
    exp_id = 'exp01_densenet121_weighted_ce'
    exp_config = get_experiment_config(exp_id)
    print(f"   ✓ Loaded config for {exp_id}")
    print_config(exp_config)
    
    print("✓ All tests passed!")