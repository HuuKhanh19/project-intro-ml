import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    """
    Load và quản lý YAML configs
    """
    
    def __init__(self, project_root=None):
        if project_root is None:
            # Tự động tìm project root
            current_file = os.path.abspath(__file__)
            src_dir = os.path.dirname(os.path.dirname(current_file))
            self.project_root = os.path.dirname(src_dir)
        else:
            self.project_root = project_root
        
        self.config_dir = os.path.join(self.project_root, 'configs')
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load config từ file YAML
        
        Args:
            config_name: Tên file config (với hoặc không .yaml)
        
        Returns:
            Dictionary chứa config
        """
        if not config_name.endswith('.yaml'):
            config_name += '.yaml'
        
        config_path = os.path.join(self.config_dir, config_name)
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Resolve paths relative to project root
        config = self._resolve_paths(config)
        
        return config
    
    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Chuyển relative paths thành absolute paths"""
        if 'data' in config and 'root_dir' in config['data']:
            config['data']['root_dir'] = os.path.join(
                self.project_root, 
                config['data']['root_dir']
            )
        
        if 'checkpoint' in config and 'save_dir' in config['checkpoint']:
            config['checkpoint']['save_dir'] = os.path.join(
                self.project_root,
                config['checkpoint']['save_dir']
            )
        
        return config
    
    def load_data_config(self) -> Dict[str, Any]:
        """Load data config"""
        return self.load_config('data_config')
    
    def load_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Load model config
        
        Args:
            model_name: 'lenet', 'resnet', 'densenet', 'vit'
        """
        config_map = {
            'lenet': 'lenet_config',
            'resnet': 'resnet_config',
            'resnet50': 'resnet_config',
            'densenet': 'densenet_config',
            'densenet121': 'densenet_config',
            'vit': 'vit_config'
        }
        
        config_name = config_map.get(model_name.lower())
        if config_name is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        return self.load_config(config_name)
    
    def get_absolute_path(self, relative_path: str) -> str:
        """Chuyển relative path thành absolute"""
        return os.path.join(self.project_root, relative_path)


# Convenience function
def load_config(config_name: str, project_root=None) -> Dict[str, Any]:
    """Quick function to load config"""
    loader = ConfigLoader(project_root)
    return loader.load_config(config_name)