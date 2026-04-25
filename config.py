import os
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class EnvironmentalContext:
    """Captured state of the world/agent at a specific point in time."""
    user_id: str = "USER"
    agent_id: str = "PRIMARY"
    session_id: str = ""
    
    # Resource Constraints
    remaining_tokens: int = 32768
    total_context: int = 32768
    
    # Environmental Metadata (Location, weather, environment)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal
    timestamp: datetime = field(default_factory=datetime.utcnow)

class ReverieConfig:
    """Central configuration manager for ReverieCore."""
    
    @staticmethod
    def get_hermes_config() -> Dict[str, Any]:
        """Loads Hermes config if available."""
        try:
            # Try to import from hermes_constants if in hermes environment
            try:
                from hermes_constants import get_hermes_home
                hermes_home = get_hermes_home()
            except ImportError:
                # Fallback to standard location
                hermes_home = Path.home() / ".hermes"
            
            config_path = hermes_home / "config.yaml"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    # We use safe_load to prevent arbitrary code execution from untrusted configs
                    return yaml.safe_load(f) or {}
        except Exception as e:
            logger.debug(f"Failed to load Hermes config: {e}")
        return {}

    @staticmethod
    def get_reverie_config_path() -> Path:
        """
        Determines the path to reveriecore.yaml based on priority:
        1. Hermes config.yaml (memory.reveriecore_cfg)
        2. Environment variable REVERIECORE_CONFIG
        3. Default ~/.reveriecore.yaml
        """
        # 1. Check Hermes config
        hermes_cfg = ReverieConfig.get_hermes_config()
        memory_cfg = hermes_cfg.get("memory", {})
        if isinstance(memory_cfg, dict):
            cfg_pointer = memory_cfg.get("reveriecore_cfg")
            if cfg_pointer:
                return Path(cfg_pointer).expanduser()
        
        # 2. Check Environment Variable
        env_path = os.environ.get("REVERIECORE_CONFIG")
        if env_path:
            return Path(env_path).expanduser()
            
        # 3. Default to home directory
        return Path.home() / ".reveriecore.yaml"

    @staticmethod
    def load_config() -> Dict[str, Any]:
        """Loads the reveriecore.yaml configuration."""
        config_path = ReverieConfig.get_reverie_config_path()
        
        if not config_path.exists():
            logger.info(f"ReverieCore config not found at {config_path}. Using defaults.")
            return {}
            
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded ReverieCore config from {config_path}")
                return config or {}
        except Exception as e:
            logger.error(f"Failed to load ReverieCore config from {config_path}: {e}")
            return {}

def load_reverie_config() -> Dict[str, Any]:
    """Helper function to load config without instantiating the class."""
    return ReverieConfig.load_config()
