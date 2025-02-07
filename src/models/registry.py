"""Registry for managing relationship types and patterns."""

import logging
from typing import Dict, Any, Optional, Set
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class RelationshipRegistry:
    """Track and manage relationship types and their patterns."""
    
    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or Path("data/relationship_registry.json")
        
        # Core relationship types
        self.types = {
            'has_element': {
                'display': 'Has Element',
                'inverse': 'element_of',
                'confidence_threshold': 0.8,
                'source': 'core',
                'created_at': datetime.now().isoformat()
            },
            'evolves_from': {
                'display': 'Evolves From',
                'inverse': 'evolves_into',
                'confidence_threshold': 0.9,
                'source': 'core',
                'created_at': datetime.now().isoformat()
            },
            'member_of': {
                'display': 'Member Of',
                'inverse': 'has_member',
                'confidence_threshold': 0.8,
                'source': 'core',
                'created_at': datetime.now().isoformat()
            }
        }
        
        # Load any saved types
        self._load_registry()
        
    def register_type(
        self,
        rel_type: str,
        config: Dict[str, Any],
        source: str = 'llm'
    ) -> bool:
        """Register a new relationship type."""
        # Don't override core types
        if rel_type in self.types and self.types[rel_type]['source'] == 'core':
            return False
            
        # Validate config
        required_fields = {'display', 'inverse', 'confidence_threshold'}
        if not all(field in config for field in required_fields):
            logger.error(f"Missing required fields for relationship type {rel_type}")
            return False
            
        # Add metadata
        config.update({
            'source': source,
            'created_at': datetime.now().isoformat()
        })
        
        # Register type
        self.types[rel_type] = config
        logger.info(f"Registered new relationship type: {rel_type}")
        
        # Save registry
        self._save_registry()
        return True
        
    def get_type(self, rel_type: str) -> Optional[Dict[str, Any]]:
        """Get relationship type configuration."""
        return self.types.get(rel_type)
        
    def is_valid_type(self, rel_type: str) -> bool:
        """Check if a relationship type is registered."""
        return rel_type in self.types
        
    def get_inverse(self, rel_type: str) -> Optional[str]:
        """Get inverse relationship type."""
        if rel_type in self.types:
            return self.types[rel_type]['inverse']
        return None
        
    def get_confidence_threshold(self, rel_type: str) -> float:
        """Get confidence threshold for a relationship type."""
        if rel_type in self.types:
            return self.types[rel_type]['confidence_threshold']
        return 0.9  # Default high threshold for unknown types
        
    def get_all_types(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered relationship types."""
        return self.types.copy()
        
    def get_learned_types(self) -> Dict[str, Dict[str, Any]]:
        """Get all learned (non-core) relationship types."""
        return {
            rel_type: config 
            for rel_type, config in self.types.items()
            if config['source'] != 'core'
        }
        
    def _load_registry(self):
        """Load registry from file."""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, 'r') as f:
                    saved_types = json.load(f)
                    
                # Only load non-core types
                for rel_type, config in saved_types.items():
                    if config['source'] != 'core':
                        self.types[rel_type] = config
                        
                logger.info(f"Loaded {len(saved_types)} relationship types from registry")
                
        except Exception as e:
            logger.error(f"Error loading relationship registry: {str(e)}")
            
    def _save_registry(self):
        """Save registry to file."""
        try:
            # Create directory if needed
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.registry_path, 'w') as f:
                json.dump(self.types, f, indent=2)
                
            logger.info(f"Saved {len(self.types)} relationship types to registry")
            
        except Exception as e:
            logger.error(f"Error saving relationship registry: {str(e)}")
            
    def merge_registry(self, other_registry: 'RelationshipRegistry'):
        """Merge another registry into this one."""
        for rel_type, config in other_registry.get_learned_types().items():
            if rel_type not in self.types or self.types[rel_type]['source'] != 'core':
                self.register_type(rel_type, config, source=config['source']) 