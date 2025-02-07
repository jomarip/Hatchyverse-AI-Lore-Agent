"""Data cleaning utilities for Hatchyverse data processing."""

import re
import pandas as pd
from typing import Any, Dict, Optional

class DataCleaner:
    """Clean and normalize input data for the knowledge graph."""
    
    @staticmethod
    def clean_political_field(value: str) -> str:
        """Clean political conflict fields."""
        if pd.isna(value):
            return ""
        
        # Remove non-conflict phrases
        value = re.sub(
            r'\b(and|vs|versus|between)\b', 
            '', 
            str(value).split('(')[0].strip()
        )
        
        # Remove trailing punctuation and whitespace
        return re.sub(r'[.,;]$', '', value).strip()
    
    @staticmethod
    def clean_faction_name(name: str) -> str:
        """Clean faction names from CSV data."""
        if pd.isna(name):
            return ""
            
        # Convert to string and strip
        name = str(name).strip()
            
        # Remove explanatory clauses and normalize
        name = re.sub(
            r'\s*(?:who|and|,).*|\(.*?\)', 
            '', 
            name
        ).strip()
        
        # Remove leading articles with word boundary
        name = re.sub(r'^\s*(?:the|a|an)\b\s+', '', name, flags=re.IGNORECASE)
        
        # Title case and final cleanup
        return ' '.join(word.capitalize() for word in name.split())
    
    @staticmethod
    def clean_element_name(element: str) -> str:
        """Clean and normalize element names."""
        if pd.isna(element):
            return ""
            
        # Convert to string, strip, and get first word
        element = str(element).strip().lower()
        element = element.split()[0]  # Take first word only
        
        # Map variations to standard names
        element_map = {
            'fire': 'Fire',
            'water': 'Water',
            'plant': 'Plant',
            'dark': 'Dark',
            'light': 'Light',
            'void': 'Void',
            'both': 'Both',  # Special case
            'lunar': 'Dark',  # Map lunar to dark
            'solar': 'Light'  # Map solar to light
        }
        
        return element_map.get(element, element.capitalize())
    
    @staticmethod
    def clean_entity_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean all fields in entity data."""
        cleaned = {}
        
        for key, value in data.items():
            if pd.isna(value):
                continue
                
            # Clean based on field type
            if key.lower() in ['faction', 'group', 'nation']:
                cleaned[key] = DataCleaner.clean_faction_name(value)
            elif key.lower() == 'element':
                cleaned[key] = DataCleaner.clean_element_name(value)
            elif key.lower() in ['political_conflict', 'political_tensions']:
                cleaned[key] = DataCleaner.clean_political_field(value)
            elif isinstance(value, str):
                # Basic string cleaning for other fields
                cleaned[key] = value.strip()
            else:
                cleaned[key] = value
        
        return cleaned
    
    @staticmethod
    def clean_relationship_data(rel_type: str, target: str) -> tuple[str, str]:
        """Clean relationship type and target."""
        # Normalize relationship type
        rel_type = rel_type.lower().replace(' ', '_')
        
        # Clean target based on relationship type
        if rel_type in ['has_element', 'element_of']:
            target = DataCleaner.clean_element_name(target)
        elif rel_type in ['member_of', 'belongs_to', 'leads']:
            target = DataCleaner.clean_faction_name(target)
            
        return rel_type, target 