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
from .relationship_extractor import AdaptiveRelationshipExtractor
from ..data.cleaners import DataCleaner

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
    
    def __init__(self, knowledge_graph: HatchyKnowledgeGraph, llm_client=None):
        self.knowledge_graph = knowledge_graph
        self.cleaner = DataCleaner()
        self.logger = logging.getLogger(__name__)
        self.relationship_extractor = AdaptiveRelationshipExtractor(llm_client)
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

    def _resolve_or_create_faction(self, faction_name: str) -> str:
        """Ensure faction entity exists and return its ID."""
        try:
            # Clean and validate faction name
            faction_name = str(faction_name).strip()
            if not faction_name or faction_name.lower() in ['none', 'n/a', '-', '', 'nan']:
                return None
            
            # Try to find existing faction
            existing = self.knowledge_graph.find_entity_by_name(faction_name)
            if existing and existing.get('entity_type') == 'faction':
                logger.debug(f"Found existing faction: {faction_name}")
                return existing['id']
            
            # Create new faction entity
            logger.debug(f"Creating new faction entity: {faction_name}")
            faction_id = self.knowledge_graph.add_entity(
                name=faction_name,
                entity_type='faction',
                attributes={
                    'name': faction_name,
                    'faction_type': 'political',
                    'description': f"Political faction known as {faction_name}"
                },
                metadata={'source': 'auto-generated'},
                source='auto-generated'
            )
            logger.info(f"Created new faction entity: {faction_name} ({faction_id})")
            return faction_id
        except Exception as e:
            logger.error(f"Error resolving faction {faction_name}: {str(e)}")
            return None

    def _process_relationships(self, entity_id: str, entity_data: dict) -> List[Dict[str, Any]]:
        """Process and store entity relationships with improved handling."""
        try:
            relationships_created = []
            
            # Clean entity data first
            cleaned_data = self.cleaner.clean_entity_data(entity_data)
            
            # Process standard relationships (Element, Faction, etc)
            if 'element' in cleaned_data and cleaned_data['element']:
                element_name = cleaned_data['element']
                element_id = self.knowledge_graph.resolve_or_create_entity(
                    name=element_name,
                    entity_type='element',
                    attributes={'name': element_name}
                )
                if element_id:
                    rel_id = self.knowledge_graph.add_relationship(
                        source_id=entity_id,
                        target_id=element_id,
                        relationship_type='has_element',
                        metadata={'confidence': 1.0}
                    )
                    if rel_id:
                        relationships_created.append(('has_element', element_name))
                        self.logger.debug(f"Created element relationship: {entity_id} -has_element-> {element_name}")

            # Process egg/hatching relationships
            if 'egg' in cleaned_data and cleaned_data['egg']:
                egg_type = cleaned_data['egg']
                egg_id = self.knowledge_graph.resolve_or_create_entity(
                    name=egg_type,
                    entity_type='egg_type',
                    attributes={'name': egg_type}
                )
                if egg_id:
                    rel_id = self.knowledge_graph.add_relationship(
                        source_id=entity_id,
                        target_id=egg_id,
                        relationship_type='hatches_from',
                        metadata={'confidence': 0.9}
                    )
                    if rel_id:
                        relationships_created.append(('hatches_from', egg_type))

            # Process faction relationships
            if 'faction' in cleaned_data and cleaned_data['faction']:
                faction_name = cleaned_data['faction']
                faction_id = self.knowledge_graph.resolve_or_create_entity(
                    name=faction_name,
                    entity_type='faction',
                    attributes={'name': faction_name}
                )
                if faction_id:
                    rel_id = self.knowledge_graph.add_relationship(
                        source_id=entity_id,
                        target_id=faction_id,
                        relationship_type='member_of',
                        metadata={'confidence': 0.9}
                    )
                    if rel_id:
                        relationships_created.append(('member_of', faction_name))

            # Process evolution relationships
            if 'evolves_from' in cleaned_data and cleaned_data['evolves_from']:
                target_name = cleaned_data['evolves_from']
                target_id = self.knowledge_graph.resolve_or_create_entity(
                    name=target_name,
                    entity_type='monster',
                    attributes={'name': target_name}
                )
                if target_id:
                    rel_id = self.knowledge_graph.add_relationship(
                        source_id=entity_id,
                        target_id=target_id,
                        relationship_type='evolves_from',
                        metadata={'confidence': 0.95}
                    )
                    if rel_id:
                        relationships_created.append(('evolves_from', target_name))

            # Process political conflicts and tensions
            political_fields = ['Political Conflict', 'Political Tensions', 'Conflict leading to Corruption']
            for field in political_fields:
                if field in cleaned_data and cleaned_data[field]:
                    # Extract relationships from political text
                    extracted = self._extract_political_relationships(cleaned_data[field])
                    for rel in extracted:
                        target_id = self.knowledge_graph.resolve_or_create_entity(
                            name=rel['target'],
                            entity_type='faction',
                            attributes={'name': rel['target']}
                        )
                        if target_id:
                            rel_id = self.knowledge_graph.add_relationship(
                                source_id=entity_id,
                                target_id=target_id,
                                relationship_type=rel['type'],
                                metadata={'confidence': rel.get('confidence', 0.8)}
                            )
                            if rel_id:
                                relationships_created.append((rel['type'], rel['target']))

            # Process character relationships from descriptions
            if 'Character Description' in cleaned_data:
                desc = cleaned_data['Character Description']
                subplot = cleaned_data.get('Subplot and Relationship to Main Plot', '')
                extracted = self._extract_character_relationships(desc + ' ' + subplot)
                for rel in extracted:
                    target_id = self.knowledge_graph.resolve_or_create_entity(
                        name=rel['target'],
                        entity_type='faction',
                        attributes={'name': rel['target']}
                    )
                    if target_id:
                        rel_id = self.knowledge_graph.add_relationship(
                            source_id=entity_id,
                            target_id=target_id,
                            relationship_type=rel['type'],
                            metadata={'confidence': rel.get('confidence', 0.8)}
                        )
                        if rel_id:
                            relationships_created.append((rel['type'], rel['target']))

            # Log relationship creation results meaningfully
            if relationships_created:
                entity_name = self.knowledge_graph.get_entity(entity_id)['name']
                self.logger.info(f"Created {len(relationships_created)} relationships for {entity_name}:")
                for rel_type, target in relationships_created:
                    self.logger.info(f"  - {entity_name} -{rel_type}-> {target}")
            elif self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug("No relationships created - no valid relationship data found")

            return relationships_created

        except Exception as e:
            self.logger.error(f"Error processing relationships for entity {entity_id}: {str(e)}")
            return []

    def _extract_political_relationships(self, text: str) -> List[Dict[str, str]]:
        """Extract relationships from political text with improved accuracy."""
        relationships = []
        
        # Define political relationship patterns with confidence scores
        patterns = [
            (r'(opposes|conflicts? with|at war with) (?:the )?([^,.]+)', 'opposes', 0.9),
            (r'(allied with|supports?) (?:the )?([^,.]+)', 'allied_with', 0.9),
            (r'(controls?|rules?|governs?) (?:the )?([^,.]+)', 'controls', 0.8),
            (r'(member of|part of|belongs to) (?:the )?([^,.]+)', 'member_of', 0.9),
            (r'(leads?|commands?) (?:the )?([^,.]+)', 'leads', 0.9),
            (r'(rebels? against|resistance to) (?:the )?([^,.]+)', 'rebels_against', 0.8)
        ]
        
        for pattern, rel_type, confidence in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                target = match.group(2).strip()
                if target and target.lower() not in ['none', 'n/a', '-', '', 'nan']:
                    # Clean target name
                    target = self.cleaner.clean_faction_name(target)
                    if target:
                        relationships.append({
                            'type': rel_type,
                            'target': target,
                            'confidence': confidence
                        })
        
        return relationships

    def _extract_character_relationships(self, text: str) -> List[Dict[str, str]]:
        """Extract relationships from character descriptions with improved accuracy."""
        relationships = []
        
        # Define character relationship patterns with confidence scores
        patterns = [
            (r'leader of (?:the )?([^,.]+)', 'leads', 0.9),
            (r'member of (?:the )?([^,.]+)', 'member_of', 0.9),
            (r'allied with (?:the )?([^,.]+)', 'allied_with', 0.8),
            (r'opposes (?:the )?([^,.]+)', 'opposes', 0.8),
            (r'serves (?:the )?([^,.]+)', 'serves', 0.9),
            (r'commands (?:the )?([^,.]+)', 'commands', 0.9)
        ]
        
        for pattern, rel_type, confidence in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                target = match.group(1).strip()
                if target and target.lower() not in ['none', 'n/a', '-', '', 'nan']:
                    # Clean target name
                    target = self.cleaner.clean_faction_name(target)
                    if target:
                        relationships.append({
                            'type': rel_type,
                            'target': target,
                            'confidence': confidence
                        })
        
        return relationships

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

    def _clean_element_data(self, element: str) -> Optional[str]:
        """Clean and normalize element data."""
        if pd.isna(element) or element is None or element == '':
            return None
            
        element = str(element).strip().lower()
        
        # Map variations to standard names
        element_map = {
            'fire': 'Fire',
            'water': 'Water',
            'plant': 'Plant',
            'dark': 'Dark',
            'light': 'Light',
            'void': 'Void',
            'both': 'Both',
            'lunar': 'Lunar',
            'solar': 'Solar'
        }
        
        return element_map.get(element, element.capitalize())

    def _clean_egg_data(self, egg: str) -> Optional[str]:
        """Clean and normalize egg type data."""
        if pd.isna(egg) or egg is None or egg == '':
            return None
            
        egg = str(egg).strip().lower()
        
        # Map variations to standard names
        egg_map = {
            'both': 'Both',
            'lunar': 'Lunar',
            'solar': 'Solar'
        }
        
        return egg_map.get(egg, egg.capitalize())

    def load_csv_data(
        self,
        file_path: str,
        entity_type: str = 'monster',
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
                        'relationships': {}
                    }
                    
                    # Process relationships immediately after entity creation
                    relationships = self._process_relationships(entity_id, data)
                    if relationships:
                        relationships_created += len(relationships)
                    
                    # Get the added entity and append to list
                    entity = self.knowledge_graph.get_entity_by_id(entity_id)
                    if entity:
                        loaded_entities.append(entity)
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing row: {str(e)}")
                    continue
            
            logger.info(f"Loaded {loaded_count} entities and created {relationships_created} relationships from {file_path}")
            logger.info(f"Loaded {loaded_count} {entity_type} entities from {os.path.basename(file_path)}")
            
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