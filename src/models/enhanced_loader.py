from typing import Dict, List, Any, Optional
import pandas as pd
import os
import uuid
import re
import logging
import csv
import json
from pathlib import Path
from .knowledge_graph import HatchyKnowledgeGraph

logger = logging.getLogger(__name__)

class TextProcessor:
    """Process text content to extract semantic information."""
    
    def __init__(self):
        self.relationship_patterns = {
            'evolves_from': r'evolves?\s+from\s+(\w+)',
            'habitat': r'found\s+in\s+(\w+)',
            'ability': r'can\s+([\w\s]+)',
            'size': r'(large|huge|massive|giant)',
            'mountable': r'(can\s+be\s+ridden|mountable|rideable)'
        }
    
    def extract_relationships(self, text: str) -> List[Dict[str, str]]:
        """Extract relationships from text."""
        relationships = []
        
        for rel_type, pattern in self.relationship_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                relationships.append({
                    'type': rel_type,
                    'value': match.group(1),
                    'context': text[max(0, match.start()-50):min(len(text), match.end()+50)]
                })
        
        return relationships

class RelationshipExtractor:
    """Extract relationships between entities."""
    
    def __init__(self):
        self.text_processor = TextProcessor()
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships from text."""
        relationships = []
        
        # Extract text-based relationships
        text_relationships = self.text_processor.extract_relationships(text)
        relationships.extend(text_relationships)
        
        # Add metadata about extraction
        for rel in relationships:
            rel['confidence'] = self._calculate_confidence(rel)
            rel['extracted_at'] = pd.Timestamp.now().isoformat()
        
        return relationships
    
    def _calculate_confidence(self, relationship: Dict[str, Any]) -> float:
        """Calculate confidence score for extracted relationship."""
        # Basic confidence scoring
        confidence = 0.7  # Base confidence
        
        # Adjust based on relationship type
        if relationship['type'] in ['evolves_from', 'mountable']:
            confidence += 0.2  # Higher confidence for explicit relationships
        
        # Adjust based on context length
        if len(relationship.get('context', '')) > 100:
            confidence += 0.1  # More context increases confidence
        
        return min(1.0, confidence)

class EnhancedDataLoader:
    """Enhanced data loader with relationship extraction and validation."""
    
    def __init__(self, knowledge_graph: HatchyKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
        self.logger = logging.getLogger(__name__)
        
    def _convert_numeric(self, value: Any, field: str) -> Optional[float]:
        """Convert numeric values safely."""
        try:
            if pd.isna(value) or value is None or value == '':
                return None
            # Handle percentage values
            if isinstance(value, str):
                if '%' in value:
                    # Convert percentage to decimal
                    return float(value.strip('%')) / 100
                # Handle fractions like '107/0'
                if '/' in value:
                    num, denom = value.split('/')
                    if float(denom) == 0:
                        return None
                    return float(num) / float(denom)
                # Handle text values that can't be converted
                if not value.replace('.', '').replace('-', '').isdigit():
                    return None
            return float(value) if value is not None else None
        except (ValueError, TypeError, ZeroDivisionError):
            logger.warning(f"Could not convert {field} value {value} to number")
            return None

    def _convert_string(self, value: Any) -> Optional[str]:
        """Convert values to strings safely."""
        try:
            if pd.isna(value) or value is None or value == '':
                return None
            # Convert float/int to string without decimal places if possible
            if isinstance(value, (float, int)):
                if float(value).is_integer():
                    return str(int(value))
                return str(value)
            # Handle special string cases
            if isinstance(value, str):
                # Clean up the string
                cleaned = value.strip()
                if not cleaned:
                    return None
                if cleaned.lower() in ['nan', 'none', 'null']:
                    return None
                return cleaned
            return str(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert value {value} to string")
            return None

    def load_csv_data(self, file_path: str, entity_type: str, relationship_mapping: Optional[Dict[str, str]] = None):
        """Load entity data from CSV with relationship extraction."""
        try:
            df = pd.read_csv(file_path)
            loaded_count = 0
            
            # Clean up problematic column names
            df.columns = [col.strip() for col in df.columns]
            
            # Remove problematic columns that contain ratios or fractions
            columns_to_drop = []
            for col in df.columns:
                if '/' in col or col.startswith('Unnamed:'):
                    columns_to_drop.append(col)
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
            
            for _, row in df.iterrows():
                try:
                    # Convert row to dict and extract core fields
                    data = row.to_dict()
                    
                    # Clean up data keys
                    data = {k.strip(): v for k, v in data.items() if isinstance(k, str)}
                    
                    # Convert numeric fields
                    numeric_fields = ['Monster ID', 'monster_id', 'Height', 'Weight']
                    for field in numeric_fields:
                        if field in data:
                            data[field] = self._convert_numeric(data[field], field)
                    
                    # Convert string fields
                    string_fields = ['Name', 'name', 'Description', 'description', 'sound', 'Element', 'element', 'Image', 'image', 'egg']
                    for field in string_fields:
                        if field in data:
                            data[field] = self._convert_string(data[field])
                    
                    # Clean up data by removing None values, empty strings, and problematic fields
                    data = {
                        k: v for k, v in data.items() 
                        if v is not None and v != '' and '/' not in k and not k.startswith('Unnamed:')
                    }
                    
                    # Add entity with converted data
                    entity_id = self.knowledge_graph.add_entity(
                        name=data.get('name', data.get('Name', f"Entity_{uuid.uuid4().hex[:8]}")),
                        entity_type=entity_type,
                        attributes=data,
                        metadata={'source_file': file_path},
                        source=file_path
                    )
                    
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing row: {str(e)}")
                    continue
                    
            logger.info(f"Loaded {loaded_count} entities from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            
    def load_text_data(self, file_path: str, chunk_size: int = 1000):
        """Load text data with chunking and entity extraction."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Split into chunks
            chunks = self._split_text(text, chunk_size)
            
            # Create unique prefix for chunks from this file
            file_prefix = Path(file_path).stem.replace(" ", "_")
            chunk_count = 0
            
            for i, chunk in enumerate(chunks):
                try:
                    # Create unique chunk name
                    chunk_name = f"{file_prefix}_{uuid.uuid4().hex[:8]}"
                    
                    # Create entity for chunk
                    self.knowledge_graph.add_entity(
                        name=chunk_name,
                        entity_type="text_chunk",
                        attributes={
                            'content': chunk,
                            'position': i,
                            'source_file': file_path,
                            'chunk_index': i
                        },
                        metadata={'source': file_path},
                        source=file_path
                    )
                    chunk_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {str(e)}")
                    continue
                    
            logger.info(f"Loaded {chunk_count} chunks from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            
    def _get_or_create_entity(self, name: str, entity_type: str) -> Optional[str]:
        """Get entity by name or create if it doesn't exist."""
        # Search for existing entity
        existing = self.knowledge_graph.get_entity_by_name(name)
        if existing:
            return existing['id']
        
        # Create new entity
        try:
            return self.knowledge_graph.add_entity(
                name=name,
                entity_type=entity_type,
                attributes={},
                metadata={'source': 'auto_created'},
                source='auto_created'
            )
        except Exception as e:
            self.logger.error(f"Error creating entity {name}: {str(e)}")
            return None
            
    def _split_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks of roughly equal size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # Add 1 for space
            if current_size + word_size > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks
    
    def load_json_data(
        self,
        file_path: str,
        entity_mapping: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """Load entity data from JSON file."""
        try:
            entity_ids = []
            file_path = Path(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both single object and array of objects
            if isinstance(data, dict):
                data = [data]
            
            for item in data:
                try:
                    # Map JSON fields to entity attributes
                    if entity_mapping:
                        mapped_item = {}
                        for entity_field, json_field in entity_mapping.items():
                            if json_field in item:
                                mapped_item[entity_field] = item[json_field]
                        item = mapped_item
                    
                    # Create entity with individual parameters
                    entity_id = self.knowledge_graph.add_entity(
                        name=item.get('name', f"Entity_{len(entity_ids)}"),
                        entity_type=item.get('type', file_path.stem),
                        attributes={
                            k: v for k, v in item.items()
                            if k not in ['id', 'name', 'type']
                        },
                        metadata={'source_file': str(file_path)},
                        source=str(file_path)
                    )
                    
                    entity_ids.append(entity_id)
                    
                except Exception as e:
                    logger.error(f"Error processing JSON item: {str(e)}")
                    continue
            
            logger.info(f"Loaded {len(entity_ids)} entities from {file_path}")
            return entity_ids
            
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {str(e)}")
            return []
    
    def load_directory(
        self,
        directory_path: str,
        file_types: Optional[List[str]] = None,
        recursive: bool = True
    ) -> Dict[str, List[str]]:
        """Load all supported files from a directory."""
        results = {
            'csv': [],
            'json': [],
            'text': [],
            'errors': []
        }
        
        try:
            directory = Path(directory_path)
            pattern = '**/*' if recursive else '*'
            
            for file_path in directory.glob(pattern):
                if not file_path.is_file():
                    continue
                    
                ext = file_path.suffix.lower()
                if file_types and ext[1:] not in file_types:
                    continue
                
                try:
                    if ext == '.csv':
                        entity_ids = self.load_csv_data(str(file_path))
                        results['csv'].extend(entity_ids)
                    elif ext == '.json':
                        entity_ids = self.load_json_data(str(file_path))
                        results['json'].extend(entity_ids)
                    elif ext in ['.txt', '.md']:
                        entity_ids = self.load_text_data(str(file_path))
                        results['text'].extend(entity_ids)
                        
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    results['errors'].append(str(file_path))
            
            # Log summary
            total = sum(len(ids) for ids in results.values())
            logger.info(
                f"Loaded {total} entities from {directory_path} "
                f"({len(results['errors'])} errors)"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            return results 