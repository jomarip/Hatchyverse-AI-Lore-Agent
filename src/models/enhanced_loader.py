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
            'mountable': r'(can\s+be\s+ridden|mountable|rideable)',
            # Add new patterns for factions and politics
            'operates_in': r'operates?\s+in\s+(\w+)',
            'affiliated_with': r'affiliated?\s+with\s+(\w+)',
            'has_conflict_with': r'conflicts?\s+with\s+(\w+)',
            'leads': r'leads?\s+(\w+)',
            'member_of': r'member\s+of\s+(\w+)',
            'allied_with': r'allied?\s+with\s+(\w+)'
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
        self.relationship_extractor = RelationshipExtractor()
        self.data_dir = None
        
    def set_data_directory(self, data_dir: Path):
        """Set the data directory for loading files."""
        self.data_dir = data_dir
        
    def load_all_data(self):
        """Load all data sources with appropriate mappings."""
        if not self.data_dir:
            raise ValueError("Data directory not set. Call set_data_directory first.")
            
        # Initialize core elements first
        core_elements = ['fire', 'water', 'plant', 'void', 'light', 'dark', 'electric', 'earth']
        logger.info("Initializing core elements...")
        for element in core_elements:
            try:
                logger.debug(f"Creating core element: {element}")
                self.knowledge_graph.add_entity(
                    name=element.capitalize(),
                    entity_type='element',
                    attributes={'name': element.capitalize()},
                    metadata={'core_element': True}
                )
                logger.debug(f"Successfully created core element: {element}")
            except Exception as e:
                logger.warning(f"Core element {element} already exists or couldn't be created: {e}")
            
        csv_mappings = [
            ('Hatchy - Monster Data - gen 1.csv', 'monster', {
                'Evolves From': 'evolves_from',
                'Habitat': 'lives_in',
                'Element': 'has_element',
                'Generation': 'belongs_to_generation'
            }),
            ('Hatchy - Monster Data - gen 2.csv', 'monster', {
                'Evolves From': 'evolves_from',
                'Habitat': 'lives_in',
                'Element': 'has_element',
                'Generation': 'belongs_to_generation',
                'egg': 'hatches_from'
            }),
            ('Hatchipedia - Factions and groups.csv', 'faction', {
                'Leader': 'has_leader',
                'Allies': 'allied_with',
                'Element': 'has_element',
                'Location': 'operates_in'
            }),
            ('Hatchipedia - famous champions.csv', 'character', {
                'Nation/Group': 'belongs_to',
                'Element': 'has_element',
                'Rival': 'rival_of',
                'Mentor': 'mentor_to'
            }),
            ('Hatchipedia - nations and politics.csv', 'nation', {
                'Trading Partners': 'trades_with',
                'Borders': 'borders',
                'Element': 'has_element',
                'Capital': 'has_capital'
            })
        ]
        
        loaded_entities = {}
        for file_name, entity_type, rel_map in csv_mappings:
            full_path = self.data_dir / file_name
            if full_path.exists():
                entities = self.load_csv_data(str(full_path), entity_type, rel_map)
                loaded_entities[entity_type] = len(entities)
                self.logger.info(f"Loaded {len(entities)} {entity_type} entities from {file_name}")
            else:
                self.logger.warning(f"Missing data file: {file_name}")
                
        return loaded_entities

    def _process_relationships(self, entity_id: str, entity_data: Dict[str, Any]):
        """Process relationships for an entity."""
        relationships_created = []
        
        try:
            logger.debug(f"Processing relationships for entity {entity_id}")
            logger.debug(f"Entity data: {entity_data}")
            
            # Handle element relationships
            if 'Element' in entity_data:
                element_value = str(entity_data['Element']).strip()
                logger.debug(f"Found Element value: {element_value}")
                if element_value.lower() not in ['none', 'n/a', '-', '', 'nan']:
                    # Handle special cases
                    if element_value.lower() == 'both':
                        elements = ['Light', 'Dark']
                    elif element_value.lower() == 'lunar':
                        elements = ['Light']  # Map Lunar to Light element
                    else:
                        elements = [element_value]
                    
                    logger.debug(f"Processing elements: {elements}")
                    for element in elements:
                        element_entity = self.knowledge_graph.find_entity_by_name(element.capitalize())
                        if element_entity:
                            try:
                                logger.debug(f"Adding element relationship: {entity_id} -> {element_entity['id']} (has_element)")
                                self.knowledge_graph.add_relationship(
                                    entity_id,
                                    element_entity['id'],
                                    'has_element'
                                )
                                relationships_created.append(('has_element', element))
                                logger.debug(f"Successfully added element relationship for {element}")
                            except Exception as e:
                                logger.error(f"Failed to create element relationship with {element}: {str(e)}")
                        else:
                            logger.warning(f"Element entity not found: {element}")
            
            # Handle evolution relationships
            if 'Evolves From' in entity_data:
                evolves_from = str(entity_data['Evolves From']).strip()
                logger.debug(f"Found Evolves From value: {evolves_from}")
                if evolves_from.lower() not in ['none', 'n/a', '-', '', 'nan']:
                    # Handle comma-separated values
                    for parent_name in [p.strip() for p in evolves_from.split(',')]:
                        if parent_name:
                            parent = self.knowledge_graph.find_entity_by_name(parent_name)
                            if parent:
                                try:
                                    logger.debug(f"Adding evolution relationship: {entity_id} -> {parent['id']} (evolves_from)")
                                    self.knowledge_graph.add_relationship(
                                        entity_id,
                                        parent['id'],
                                        'evolves_from'
                                    )
                                    relationships_created.append(('evolves_from', parent_name))
                                    logger.debug(f"Successfully added evolution relationship for {parent_name}")
                                except Exception as e:
                                    logger.error(f"Failed to create evolution relationship with {parent_name}: {str(e)}")
                            else:
                                logger.warning(f"Parent entity not found: {parent_name}")
            
            # Handle habitat relationships
            if 'Habitat' in entity_data:
                habitat = str(entity_data['Habitat']).strip()
                logger.debug(f"Found Habitat value: {habitat}")
                if habitat.lower() not in ['none', 'n/a', '-', '', 'nan']:
                    # Handle comma-separated values
                    for location in [loc.strip() for loc in habitat.split(',')]:
                        if location:
                            # Create location entity if it doesn't exist
                            location_entity = self.knowledge_graph.find_entity_by_name(location)
                            if not location_entity:
                                try:
                                    logger.debug(f"Creating new location entity: {location}")
                                    location_id = self.knowledge_graph.add_entity(
                                        name=location,
                                        entity_type='location',
                                        attributes={'name': location}
                                    )
                                    location_entity = self.knowledge_graph.get_entity(location_id)
                                    logger.debug(f"Successfully created location entity: {location}")
                                except Exception as e:
                                    logger.error(f"Failed to create location entity {location}: {str(e)}")
                                    continue
                            
                            if location_entity:
                                try:
                                    logger.debug(f"Adding habitat relationship: {entity_id} -> {location_entity['id']} (lives_in)")
                                    self.knowledge_graph.add_relationship(
                                        entity_id,
                                        location_entity['id'],
                                        'lives_in'
                                    )
                                    relationships_created.append(('lives_in', location))
                                    logger.debug(f"Successfully added habitat relationship for {location}")
                                except Exception as e:
                                    logger.error(f"Failed to create habitat relationship with {location}: {str(e)}")
            
            # Log successful relationships
            if relationships_created:
                entity_name = self.knowledge_graph.get_entity(entity_id)['name']
                logger.info(f"Created {len(relationships_created)} relationships for {entity_name}:")
                for rel_type, target in relationships_created:
                    logger.info(f"  - {entity_name} -{rel_type}-> {target}")
            else:
                logger.debug("No relationships created for this entity")
            
            return relationships_created
        
        except Exception as e:
            logger.error(f"Error processing relationships for entity {entity_id}: {str(e)}")
            return []

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
                # For ID fields, keep as string
                if field.lower() in ['id', 'monster id', 'monster_id']:
                    return value
                # For text fields, keep as string
                if not value.replace('.', '').replace('-', '').isdigit():
                    return value
            return float(value) if value is not None else None
        except (ValueError, TypeError, ZeroDivisionError):
            # For fields that should remain as strings, return the original value
            if field.lower() in ['id', 'monster id', 'monster_id', 'nation name', 'faction', 'groups', 'symbol', 'note', 'themes', 'character description', 'subplot and relationship to main plot', 'political tensions', 'hatchy culture', 'conflict leading to corruption', 'story']:
                return value
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

    def load_csv_data(
        self,
        file_path: str,
        entity_type: str,
        relationship_mapping: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Load entity data from CSV with relationship extraction."""
        try:
            # Read CSV with string type for all columns
            df = pd.read_csv(file_path, dtype=str)
            loaded_count = 0
            loaded_entities = []
            relationships_created = 0
            
            # Clean up problematic column names
            df.columns = [col.strip() for col in df.columns]
            
            # Remove problematic columns that contain ratios or fractions
            columns_to_drop = []
            for col in df.columns:
                if '/' in col or col.startswith('Unnamed:'):
                    columns_to_drop.append(col)
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
            
            # Track entities for relationship creation
            entity_map = {}
            
            for _, row in df.iterrows():
                try:
                    # Convert row to dict and extract core fields
                    data = row.to_dict()
                    
                    # Clean up data keys
                    data = {k.strip(): v for k, v in data.items() if isinstance(k, str)}
                    
                    # Convert numeric fields only for specific fields
                    numeric_fields = ['Height', 'Weight']
                    for field in numeric_fields:
                        if field in data:
                            data[field] = self._convert_numeric(data[field], field)
                    
                    # Convert string fields and ensure proper element handling
                    string_fields = ['Name', 'name', 'Description', 'description', 'sound', 'Element', 'element', 'Image', 'image', 'egg']
                    for field in string_fields:
                        if field in data:
                            data[field] = self._convert_string(data[field])
                    
                    # Ensure element is properly handled
                    element = None
                    for element_field in ['Element', 'element']:
                        if element_field in data and data[element_field]:
                            element = data[element_field].lower().strip()
                            break
                    
                    # Clean up data by removing None values and empty strings
                    data = {
                        k: v for k, v in data.items() 
                        if v is not None and v != ''
                    }
                    
                    # Ensure element is included in attributes
                    if element:
                        data['element'] = element
                    
                    # Extract generation from filename or data
                    generation = None
                    gen_match = re.search(r'gen(?:eration)?[\s\-_]*(\d+)', file_path, re.IGNORECASE)
                    if gen_match:
                        generation = gen_match.group(1)
                    elif 'generation' in data:
                        generation = str(data['generation'])
                    
                    if generation:
                        data['generation'] = generation
                    
                    # Separate relationship data
                    relationship_data = {}
                    if relationship_mapping:
                        for source_field, rel_type in relationship_mapping.items():
                            if source_field in data and data[source_field]:
                                relationship_data[rel_type] = data.pop(source_field)
                    
                    # Add entity with converted data
                    entity_id = self.knowledge_graph.add_entity(
                        name=data.get('name', data.get('Name', f"Entity_{uuid.uuid4().hex[:8]}")),
                        entity_type=entity_type,
                        attributes=data,
                        metadata={'source_file': file_path},
                        source=file_path
                    )
                    
                    # Store entity for relationship creation
                    entity_map[entity_id] = {
                        'data': data,
                        'relationships': relationship_data
                    }
                    
                    # Get the added entity and append to list
                    entity = self.knowledge_graph.get_entity_by_id(entity_id)
                    if entity:
                        loaded_entities.append(entity)
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing row: {str(e)}")
                    continue
            
            # Create relationships after all entities are loaded
            total_relationships = 0
            for entity_id, entity_info in entity_map.items():
                try:
                    # Process both standard and mapped relationships
                    created_relationships = []
                    
                    # Process standard relationships (Element, Evolves From, Habitat)
                    standard_rels = self._process_relationships(entity_id, entity_info['data'])
                    if standard_rels:
                        created_relationships.extend(standard_rels)
                    
                    # Process mapped relationships
                    if entity_info['relationships']:
                        logger.debug(f"Processing mapped relationships for entity {entity_id}: {entity_info['relationships']}")
                        for rel_type, target_value in entity_info['relationships'].items():
                            if target_value and str(target_value).lower() not in ['none', 'n/a', '-', '', 'nan']:
                                # Handle comma-separated values
                                for target_name in [t.strip() for t in str(target_value).split(',')]:
                                    if target_name:
                                        try:
                                            # Find or create target entity
                                            target_entity = self.knowledge_graph.find_entity_by_name(target_name)
                                            if not target_entity:
                                                logger.debug(f"Creating new target entity: {target_name}")
                                                target_id = self.knowledge_graph.add_entity(
                                                    name=target_name,
                                                    entity_type=rel_type.split('_')[-1],  # e.g., 'has_leader' -> 'leader'
                                                    attributes={'name': target_name}
                                                )
                                                target_entity = self.knowledge_graph.get_entity_by_id(target_id)
                                            
                                            if target_entity:
                                                logger.debug(f"Adding relationship: {entity_id} -{rel_type}-> {target_entity['id']}")
                                                self.knowledge_graph.add_relationship(
                                                    entity_id,
                                                    target_entity['id'],
                                                    rel_type
                                                )
                                                created_relationships.append((rel_type, target_name))
                                        except Exception as e:
                                            logger.error(f"Failed to create relationship {rel_type} with {target_name}: {str(e)}")
                    
                    total_relationships += len(created_relationships)
                    if created_relationships:
                        entity_name = self.knowledge_graph.get_entity_by_id(entity_id)['name']
                        logger.info(f"Created {len(created_relationships)} relationships for {entity_name}:")
                        for rel_type, target in created_relationships:
                            logger.info(f"  - {entity_name} -{rel_type}-> {target}")
                except Exception as e:
                    logger.error(f"Error processing relationships for entity {entity_id}: {str(e)}")
            
            logger.info(f"Loaded {loaded_count} entities and created {total_relationships} relationships from {file_path}")
            return loaded_entities
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            return []

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
            loaded_chunks = []
            
            for i, chunk in enumerate(chunks):
                try:
                    # Create unique chunk name
                    chunk_name = f"{file_prefix}_{uuid.uuid4().hex[:8]}"
                    
                    # Create entity for chunk
                    entity_id = self.knowledge_graph.add_entity(
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
                    
                    # Get the added entity and append to list
                    entity = self.knowledge_graph.get_entity_by_id(entity_id)
                    if entity:
                        loaded_chunks.append(entity)
                    chunk_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {str(e)}")
                    continue
                    
            logger.info(f"Loaded {chunk_count} chunks from {file_path}")
            return loaded_chunks
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            return []

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