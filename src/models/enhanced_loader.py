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
        
    def load_csv_data(self, file_path: str, entity_type: str, relationship_mapping: Optional[Dict[str, str]] = None):
        """Load entity data from CSV with relationship extraction."""
        try:
            df = pd.read_csv(file_path)
            loaded_count = 0
            
            for _, row in df.iterrows():
                try:
                    # Convert row to dict and extract core fields
                    data = row.to_dict()
                    name = data.pop('name', f"Entity_{uuid.uuid4().hex[:8]}")
                    
                    # Extract relationship fields
                    relationship_data = {}
                    if relationship_mapping:
                        for source_field, rel_type in relationship_mapping.items():
                            if source_field in data and pd.notna(data[source_field]):
                                relationship_data[rel_type] = data.pop(source_field)
                    
                    # Add entity with individual parameters
                    entity_id = self.knowledge_graph.add_entity(
                        name=name,
                        entity_type=entity_type,
                        attributes=data,
                        metadata={'source_file': file_path},
                        source=file_path
                    )
                    
                    # Process relationships
                    for rel_type, target_name in relationship_data.items():
                        # Create target entity if it doesn't exist
                        target_id = self._get_or_create_entity(target_name, entity_type)
                        if target_id:
                            self.knowledge_graph.add_relationship(entity_id, target_id, rel_type)
                    
                    loaded_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing row: {str(e)}")
                    continue
                    
            self.logger.info(f"Loaded {loaded_count} entities from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            
    def load_text_data(self, file_path: str, chunk_size: int = 1000):
        """Load text data with chunking and entity extraction."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Split into chunks
            chunks = self._split_text(text, chunk_size)
            
            for i, chunk in enumerate(chunks):
                try:
                    # Create entity for chunk
                    self.knowledge_graph.add_entity(
                        name=f"Chunk_{i}",
                        entity_type="text_chunk",
                        attributes={'content': chunk, 'position': i},
                        source=file_path
                    )
                except Exception as e:
                    self.logger.error(f"Error processing chunk {i}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error loading text file {file_path}: {str(e)}")
            
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