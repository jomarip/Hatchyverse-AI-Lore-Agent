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
    """Enhanced data loader for processing various file formats."""
    
    def __init__(self, knowledge_graph: HatchyKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
    
    def load_csv_data(
        self,
        file_path: str,
        entity_type: Optional[str] = None,
        relationship_mapping: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """Load entity data from CSV file."""
        try:
            entity_ids = []
            file_path = Path(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        # Process entity attributes
                        attributes = {}
                        for key, value in row.items():
                            if key.lower() not in ['id', 'name', 'type'] and value:
                                attributes[key] = value
                        
                        # Determine entity type
                        if not entity_type:
                            if 'type' in row:
                                entity_type = row['type']
                            else:
                                entity_type = file_path.stem
                        
                        # Create entity
                        entity_id = self.knowledge_graph.add_entity(
                            name=row.get('name', f"Entity_{len(entity_ids)}"),
                            entity_type=entity_type,
                            attributes=attributes,
                            metadata={'source_file': str(file_path)}
                        )
                        
                        entity_ids.append(entity_id)
                        
                        # Process relationships if mapping provided
                        if relationship_mapping:
                            self._process_relationships(
                                entity_id,
                                row,
                                relationship_mapping
                            )
                            
                    except Exception as e:
                        logger.error(f"Error processing row: {str(e)}")
                        continue
            
            logger.info(f"Loaded {len(entity_ids)} entities from {file_path}")
            return entity_ids
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            return []
    
    def load_text_data(
        self,
        file_path: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[str]:
        """Load and chunk text data."""
        try:
            entity_ids = []
            file_path = Path(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split text into chunks
            chunks = self._chunk_text(text, chunk_size, overlap)
            
            # Create text entities
            for i, chunk in enumerate(chunks):
                entity_id = self.knowledge_graph.add_entity(
                    name=f"{file_path.stem}_chunk_{i}",
                    entity_type='text_chunk',
                    attributes={
                        'content': chunk,
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    },
                    metadata={
                        'source_file': str(file_path),
                        'chunk_size': chunk_size,
                        'overlap': overlap
                    }
                )
                
                entity_ids.append(entity_id)
                
                # Link chunks sequentially
                if i > 0:
                    self.knowledge_graph.add_relationship(
                        entity_ids[i-1],
                        entity_id,
                        'next_chunk'
                    )
            
            logger.info(f"Created {len(entity_ids)} text chunks from {file_path}")
            return entity_ids
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            return []
    
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
                    
                    # Create entity
                    entity_id = self.knowledge_graph.add_entity(
                        name=item.get('name', f"Entity_{len(entity_ids)}"),
                        entity_type=item.get('type', file_path.stem),
                        attributes={
                            k: v for k, v in item.items()
                            if k not in ['id', 'name', 'type']
                        },
                        metadata={'source_file': str(file_path)}
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
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk with overlap
            end = start + chunk_size
            chunk = text[start:end]
            
            # Adjust chunk boundaries to avoid splitting words
            if end < len(text):
                # Find last space within chunk
                last_space = chunk.rfind(' ')
                if last_space != -1:
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk.strip())
            
            # Move start position, accounting for overlap
            start = end - overlap
            
            # Ensure we don't get stuck in small text
            if start >= len(text) - overlap:
                break
        
        return chunks
    
    def _process_relationships(
        self,
        entity_id: str,
        row: Dict[str, str],
        relationship_mapping: Dict[str, str]
    ) -> None:
        """Process relationships based on mapping."""
        for field, rel_type in relationship_mapping.items():
            if field in row and row[field]:
                # Handle multiple relationships in one field
                targets = row[field].split(',')
                
                for target in targets:
                    target = target.strip()
                    if not target:
                        continue
                    
                    # Try to find target entity
                    target_entity = self.knowledge_graph.get_entity_by_name(target)
                    if target_entity:
                        self.knowledge_graph.add_relationship(
                            entity_id,
                            target_entity['id'],
                            rel_type
                        )
                    else:
                        logger.warning(
                            f"Target entity '{target}' not found for relationship '{rel_type}'"
                        )
    
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