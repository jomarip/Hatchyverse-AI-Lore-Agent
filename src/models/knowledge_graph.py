from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import uuid
import re
from collections import defaultdict
import logging
import networkx as nx
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

class EntityData(BaseModel):
    """Validation model for entity data."""
    name: str = Field(..., min_length=1)
    entity_type: str = Field(..., min_length=1)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None
    source: Optional[str] = None

    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be empty or just whitespace")
        return v.strip()

    @validator('entity_type')
    def validate_type(cls, v):
        if not v.strip():
            raise ValueError("Entity type cannot be empty or just whitespace")
        return v.strip()

class HatchyKnowledgeGraph:
    """Core knowledge representation for the Hatchyverse."""
    
    def __init__(self):
        """Initialize the knowledge graph with proper relationship tracking."""
        self.graph = nx.MultiDiGraph()
        self.entities = {}  # Entity storage
        self.relationship_types = set()  # Track unique relationship types
        self.source_registry = defaultdict(list)  # Track entities by source
        self.generation_cache = {}  # Cache for generation lookups
        
        # Initialize relationship storage
        self.relationships = {}  # Store all relationships by ID
        
        # Initialize relationship registry with core types
        self.relationship_registry = {
            'has_element': {'inverse': 'element_of', 'confidence_threshold': 0.9},
            'evolves_from': {'inverse': 'evolves_into', 'confidence_threshold': 0.9},
            'member_of': {'inverse': 'has_member', 'confidence_threshold': 0.8},
            'allied_with': {'inverse': 'allied_with', 'confidence_threshold': 0.8},  # Symmetric
            'opposes': {'inverse': 'opposed_by', 'confidence_threshold': 0.8},
            'controls': {'inverse': 'controlled_by', 'confidence_threshold': 0.8},
            'commands': {'inverse': 'commanded_by', 'confidence_threshold': 0.8},
            'hatches_from': {'inverse': 'hatches_into', 'confidence_threshold': 0.9},  # Added for egg hatching
            'lives_in': {'inverse': 'habitat_of', 'confidence_threshold': 0.8},
            'belongs_to': {'inverse': 'owns', 'confidence_threshold': 0.8},
            'trades_with': {'inverse': 'trades_with', 'confidence_threshold': 0.8},  # Symmetric
            'borders': {'inverse': 'borders', 'confidence_threshold': 0.9},  # Symmetric
            'has_capital': {'inverse': 'capital_of', 'confidence_threshold': 0.9},
            'leads': {'inverse': 'led_by', 'confidence_threshold': 0.9},  # Added for leadership
            'serves': {'inverse': 'served_by', 'confidence_threshold': 0.8},  # Added for service relationships
            'mentors': {'inverse': 'mentored_by', 'confidence_threshold': 0.9}  # Added for mentorship
        }
        
        # Add indexes
        self._name_index = {}  # For fast name lookups
        self._type_index = defaultdict(set)  # For fast type-based lookups
        self._attribute_index = defaultdict(lambda: defaultdict(set))  # For attribute-based lookups
        self._relationship_index = defaultdict(lambda: defaultdict(list))  # For relationship lookups by type and source
        self._reverse_relationship_index = defaultdict(lambda: defaultdict(list))  # For reverse relationship lookups
        
        # Initialize statistics tracking
        self._stats = {
            'total_entities': 0,
            'total_relationships': 0,
            'relationship_counts': defaultdict(int),  # Count by type
            'entity_type_counts': defaultdict(int),   # Count by entity type
            'element_counts': defaultdict(int)        # Count by element
        }
        
        # Predefine core elements
        self.core_elements = {
            'fire', 'water', 'plant', 'void', 'light', 'dark',
            'electric', 'earth', 'air', 'metal', 'chaos', 'order',
            'both', 'lunar', 'solar'  # Added special element types
        }
        self._init_core_entities()
        
    def _init_core_entities(self):
        """Create base entities for core game concepts"""
        for element in self.core_elements:
            try:
                element_name = element.capitalize()
                # Check if element already exists
                existing = None
                for ent in self.entities.values():
                    if (ent['name'].lower() == element_name.lower() and 
                        ent['entity_type'] == 'element'):
                        existing = ent
                        break
                
                if not existing:
                    logger.debug(f"Creating core element: {element_name}")
                    
                    # Special handling for dual/special elements
                    attributes = {
                        "name": element_name,
                        "symbol": self._get_element_symbol(element),
                        "is_core_element": True
                    }
                    
                    if element == 'both':
                        attributes.update({
                            "combines": ["Light", "Dark"],
                            "is_dual_element": True
                        })
                    elif element == 'lunar':
                        attributes.update({
                            "related_to": "Dark",
                            "is_special_element": True
                        })
                    elif element == 'solar':
                        attributes.update({
                            "related_to": "Light",
                            "is_special_element": True
                        })
                    
                    self.add_entity(
                        name=element_name,
                        entity_type="element",
                        attributes=attributes,
                        metadata={'core_element': True}
                    )
                    logger.debug(f"Successfully created core element: {element_name}")
            except Exception as e:
                logger.warning(f"Error creating core element {element}: {str(e)}")

    def _get_element_symbol(self, element: str) -> str:
        """Maps elements to display symbols"""
        symbols = {
            'fire': '🔥', 'water': '💧', 'plant': '🌿',
            'void': '🌌', 'light': '✨', 'dark': '🌑',
            'electric': '⚡', 'earth': '🌍', 'air': '🌪️',
            'metal': '⚙️', 'chaos': '🌀', 'order': '⚖️',
            'both': '☯️', 'lunar': '🌙', 'solar': '☀️'  # Added special element symbols
        }
        return symbols.get(element.lower(), '❓')

    def _validate_entity_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate entity data before insertion."""
        errors = []
        
        # Check required fields
        required_fields = ['name', 'entity_type']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif not data[field]:
                errors.append(f"Field {field} cannot be empty")
        
        # Validate name format
        if 'name' in data and data['name']:
            if not isinstance(data['name'], str):
                errors.append("Name must be a string")
            elif len(data['name'].strip()) == 0:
                errors.append("Name cannot be empty or just whitespace")
        
        # Validate entity_type
        if 'entity_type' in data and data['entity_type']:
            if not isinstance(data['entity_type'], str):
                errors.append("Entity type must be a string")
            elif len(data['entity_type'].strip()) == 0:
                errors.append("Entity type cannot be empty or just whitespace")
        
        # Validate attributes
        if 'attributes' in data:
            if not isinstance(data['attributes'], dict):
                errors.append("Attributes must be a dictionary")
        
        return len(errors) == 0, errors

    def _check_data_consistency(self, entity_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check data consistency before insertion."""
        issues = []
        
        # Check for duplicate names - modified to be more lenient
        if entity_data['name'] in self._name_index:
            # Instead of error, modify name to make unique
            base_name = entity_data['name']
            counter = 1
            while f"{base_name}_{counter}" in self._name_index:
                counter += 1
            entity_data['name'] = f"{base_name}_{counter}"
        
        # Clean up attributes to remove problematic fields and handle type conversions
        if 'attributes' in entity_data:
            cleaned_attributes = {}
            for k, v in entity_data['attributes'].items():
                # Skip problematic keys
                if not isinstance(k, str) or '/' in k or k.startswith('Unnamed:'):
                    continue
                    
                # Handle special fields that should remain as strings
                if k.lower() in ['id', 'monster id', 'monster_id', 'nation name', 'faction', 'groups', 
                               'symbol', 'note', 'themes', 'character description', 
                               'subplot and relationship to main plot', 'political tensions', 
                               'hatchy culture', 'conflict leading to corruption', 'story']:
                    cleaned_attributes[k] = str(v) if v is not None else None
                    continue
                    
                # Handle numeric fields
                if k.lower() in ['height', 'weight']:
                    try:
                        cleaned_attributes[k] = float(v) if v is not None else None
                    except (ValueError, TypeError):
                        cleaned_attributes[k] = None
                    continue
                    
                # Keep other fields as is
                cleaned_attributes[k] = v
                
            entity_data['attributes'] = cleaned_attributes
        
        return len(issues) == 0, issues

    def _update_indexes(self, entity_id: str, entity_data: Dict[str, Any]):
        """Update all indexes with new entity data."""
        try:
            # Handle both dictionary and sequence types
            if isinstance(entity_data, (list, tuple)):
                name = str(entity_data[0]) if len(entity_data) > 0 else None
                entity_type = str(entity_data[1]) if len(entity_data) > 1 else None
                attributes = entity_data[2] if len(entity_data) > 2 else {}
            else:
                name = entity_data.get('name')
                entity_type = entity_data.get('entity_type')
                attributes = entity_data.get('attributes', {})

            if name:
                self._name_index[name.lower()] = entity_id
            
            if entity_type:
                if entity_type not in self._type_index:
                    self._type_index[entity_type] = set()
                self._type_index[entity_type].add(entity_id)
            
            # Update attribute index
            if isinstance(attributes, dict):
                for attr, value in attributes.items():
                    if isinstance(value, (str, int, float, bool)):
                        self._attribute_index[attr][str(value)].add(entity_id)
                        
        except Exception as e:
            logger.error(f"Error updating indexes for entity {entity_id}: {str(e)}")

    def _remove_from_indexes(self, entity_id: str):
        """Remove entity from all indexes."""
        entity = self.entities.get(entity_id)
        if not entity:
            return
            
        # Remove from name index
        self._name_index.pop(entity['name'], None)
        
        # Remove from type index
        self._type_index[entity['entity_type']].discard(entity_id)
        
        # Remove from attribute index
        for attr, value in entity.get('attributes', {}).items():
            if isinstance(value, (str, int, float, bool)):
                self._attribute_index[attr][str(value)].discard(entity_id)

    def _extract_generation(self, source: Optional[str], entity: Dict[str, Any]) -> Optional[str]:
        """Extract generation info from source or entity data with enhanced fallbacks."""
        try:
            # Priority 1: Direct generation attribute
            if 'generation' in entity.get('attributes', {}):
                return str(entity['attributes']['generation'])
            
            # Priority 2: Generation in metadata
            if '_metadata' in entity and 'generation' in entity['_metadata']:
                return str(entity['_metadata']['generation'])
            if 'metadata' in entity and 'generation' in entity['metadata']:
                return str(entity['metadata']['generation'])
            
            # Priority 3: Extract from source filename
            if source:
                # Try different generation patterns
                patterns = [
                    r'gen(?:eration)?[\s\-_]*(\d+)',
                    r'(?:gen|generation)\s*(\d+)',
                    r'gen-?(\d+)',
                    r'_g(\d+)_'
                ]
                for pattern in patterns:
                    if match := re.search(pattern, source, re.IGNORECASE):
                        return match.group(1)
            
            # Priority 4: Look for generation in description
            if 'Description' in entity:
                desc = entity['Description'].lower()
                if 'generation 1' in desc or 'gen 1' in desc or 'gen-1' in desc:
                    return '1'
                elif 'generation 2' in desc or 'gen 2' in desc or 'gen-2' in desc:
                    return '2'
                elif 'generation 3' in desc or 'gen 3' in desc or 'gen-3' in desc:
                    return '3'
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting generation: {str(e)}")
            return None

    def add_entity(
        self,
        name: str,
        entity_type: str,
        attributes: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> str:
        """Add an entity to the graph."""
        entity_id = str(uuid.uuid4())
        
        # Clean and validate name
        name = str(name).strip()
        if not name:
            raise ValueError("Entity name cannot be empty")
            
        # Store entity
        entity = {
            'id': entity_id,
            'name': name,
            'entity_type': entity_type,
            'attributes': attributes or {},
            'metadata': metadata or {},
            'source': source,
            'created_at': datetime.now().isoformat()
        }
        self.entities[entity_id] = entity
        
        # Update indexes
        self._name_index[name.lower()] = entity_id
        self._type_index.setdefault(entity_type, set()).add(entity_id)
        
        # Log entity creation
        logger.debug(f"Created entity: {name} ({entity_id}) of type {entity_type}")
        
        return entity_id

    def batch_add_entities(
        self,
        entities: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Add multiple entities in batch with validation.
        
        Returns:
            Tuple of (successful_ids, failed_entities)
        """
        successful_ids = []
        failed_entities = []
        
        for entity_data in entities:
            try:
                entity_id = self.add_entity(**entity_data)
                successful_ids.append(entity_id)
            except Exception as e:
                failed_entities.append({
                    'data': entity_data,
                    'error': str(e)
                })
        
        return successful_ids, failed_entities

    def get_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get entity by name."""
        entity_id = self._name_index.get(name.lower())
        if entity_id:
            return self.get_entity(entity_id)
        return None

    def find_entity_by_name(
        self,
        name: str,
        fuzzy_match: bool = True,
        entity_type: Optional[str] = None
    ) -> Optional[str]:
        """Find entity ID by name with improved matching."""
        if not name:
            return None
            
        name_lower = name.lower()
        
        # Try exact match first
        for eid, entity in self.entities.items():
            if entity['name'].lower() == name_lower:
                if not entity_type or entity['entity_type'] == entity_type:
                    return eid
        
        # Try fuzzy matching if enabled
        if fuzzy_match:
            best_match = None
            best_score = 0
            
            for eid, entity in self.entities.items():
                if entity_type and entity['entity_type'] != entity_type:
                    continue
                    
                # Calculate similarity score
                score = self._calculate_name_similarity(name_lower, entity['name'].lower())
                if score > best_score and score > 0.8:  # 80% similarity threshold
                    best_match = eid
                    best_score = score
            
            if best_match:
                logger.debug(f"Fuzzy matched '{name}' to '{self.entities[best_match]['name']}'")
                return best_match
        
        return None

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two names."""
        # Simple word overlap for now
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if not words1 or not words2:
            return 0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        file_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search entities using indexes for better performance."""
        results = set()
        query = query.lower()
        
        # Use type index if specified
        candidate_ids = (
            self._type_index.get(entity_type, set())
            if entity_type
            else set(self.entities.keys())
        )
        
        # Apply filters using attribute index
        if filters:
            for attr, value in filters.items():
                if attr in self._attribute_index:
                    matching_ids = self._attribute_index[attr].get(str(value), set())
                    candidate_ids &= matching_ids
        
        # Search through candidates
        for entity_id in candidate_ids:
            entity = self.entities[entity_id]
            
            # Check if entity matches file filter
            if file_filter and entity.get('source') != file_filter:
                continue
            
            # Check name match
            if query in entity['name'].lower():
                results.add(entity_id)
                if len(results) >= limit:
                    break
            
            # Check attributes
            if len(results) < limit:
                for value in entity.get('attributes', {}).values():
                    if isinstance(value, str) and query in value.lower():
                        results.add(entity_id)
                        if len(results) >= limit:
                            break
        
        return [self._prepare_entity_for_output(self.entities[eid]) for eid in results]

    def check_data_integrity(self) -> Dict[str, Any]:
        """Perform comprehensive data integrity check."""
        issues = {
            'missing_required_fields': [],
            'invalid_relationships': [],
            'orphaned_entities': [],
            'index_inconsistencies': [],
            'type_inconsistencies': []
        }
        
        # Check entities
        for entity_id, entity in self.entities.items():
            # Check required fields
            for field in ['name', 'entity_type', 'attributes']:
                if field not in entity:
                    issues['missing_required_fields'].append(
                        f"Entity {entity_id} missing required field: {field}"
                    )
            
            # Check relationship validity
            for _, target, _ in self.graph.edges(entity_id, data=True):
                if target not in self.entities:
                    issues['invalid_relationships'].append(
                        f"Entity {entity_id} has relationship to non-existent entity {target}"
                    )
            
            # Check index consistency
            if entity['name'] not in self._name_index:
                issues['index_inconsistencies'].append(
                    f"Entity {entity_id} missing from name index"
                )
            if entity_id not in self._type_index[entity['entity_type']]:
                issues['index_inconsistencies'].append(
                    f"Entity {entity_id} missing from type index"
                )
        
        # Check for orphaned entities in indexes
        for name, entity_id in self._name_index.items():
            if entity_id not in self.entities:
                issues['orphaned_entities'].append(
                    f"Name index contains orphaned reference: {name} -> {entity_id}"
                )
        
        return issues

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        attributes: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Add a relationship between two entities with improved type handling."""
        try:
            # Validate relationship type
            if not self.is_valid_type(relationship_type):
                logger.warning(f"Invalid relationship type: {relationship_type}")
                return None

            # Handle special cases where target_id is a string name instead of an ID
            if relationship_type in ['has_element', 'hatches_from']:
                # For elements and hatching, resolve target by name
                target_entity = self.resolve_or_create_entity(
                    name=target_id,  # In this case target_id is actually the name
                    entity_type='element' if relationship_type == 'has_element' else 'egg_type',
                    attributes={
                        'name': target_id,
                        'is_core_element': target_id.lower() in self.core_elements
                    } if relationship_type == 'has_element' else {'name': target_id}
                )
                target_id = target_entity  # Use the resolved/created entity ID

            # Validate and resolve source entity
            source_entity = self.get_entity(source_id)  # Use get_entity which handles type conversion
            if not source_entity:
                logger.warning(f"Source entity {source_id} not found, attempting resolution")
                source_id = self.resolve_or_create_entity(
                    name=source_id,
                    entity_type='auto_source',
                    attributes={'auto_created': True},
                    metadata={'source': 'relationship_resolution'}
                )

            # Get or create target entity (for non-special cases)
            target_entity = self.get_entity(target_id)  # Use get_entity which handles type conversion
            if not target_entity:
                logger.warning(f"Target entity {target_id} not found, attempting resolution")
                target_id = self.resolve_or_create_entity(
                    name=target_id,
                    entity_type='auto_target',
                    attributes={'auto_created': True},
                    metadata={'source': 'relationship_resolution'}
                )

            # Generate relationship ID
            relationship_id = str(uuid.uuid4())
            
            # Create relationship data with proper confidence
            relationship_data = {
                'id': relationship_id,
                'source_id': source_id,
                'target_id': target_id,
                'type': relationship_type,
                'attributes': attributes or {},
                'metadata': {
                    **(metadata or {}),
                    'created_at': datetime.now().isoformat(),
                    'confidence': self._calculate_confidence(relationship_type, metadata)
                }
            }

            # Store relationship
            self.relationships[relationship_id] = relationship_data

            # Add to graph
            self.graph.add_edge(
                source_id,
                target_id,
                key=relationship_id,
                **relationship_data
            )

            # Update indexes
            self._relationship_index[relationship_type][source_id].append(relationship_data)
            self._reverse_relationship_index[relationship_type][target_id].append(relationship_data)
            self.relationship_types.add(relationship_type)
            
            # Update statistics
            self._stats['total_relationships'] += 1
            self._stats['relationship_counts'][relationship_type] = \
                self._stats['relationship_counts'].get(relationship_type, 0) + 1

            # Log relationship creation with entity names for clarity
            source_name = self.get_entity_name(source_id)
            target_name = self.get_entity_name(target_id)
            logger.info(f"Created relationship: {source_name} -{relationship_type}-> {target_name}")
            
            return relationship_id

        except Exception as e:
            logger.error(f"Error creating relationship: {str(e)}")
            return None

    def _calculate_confidence(self, relationship_type: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        """Calculate confidence score for a relationship."""
        base_confidence = self.relationship_registry.get(relationship_type, {}).get('confidence_threshold', 0.5)
        
        # If metadata contains explicit confidence, use that
        if metadata and 'confidence' in metadata:
            return float(metadata['confidence'])
            
        # Special case for mentorship relationships
        if relationship_type == 'mentors':
            return 0.9
            
        return base_confidence

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID with safe type handling."""
        try:
            entity = self.entities.get(entity_id)
            if entity is None:
                return None
                
            # Handle case where entity is a sequence instead of dict
            if isinstance(entity, (list, tuple)):
                # Convert sequence to dictionary format
                return {
                    'id': entity_id,
                    'name': str(entity[0]) if len(entity) > 0 else 'Unknown',
                    'entity_type': str(entity[1]) if len(entity) > 1 else 'unknown',
                    'attributes': entity[2] if len(entity) > 2 else {},
                    'metadata': entity[3] if len(entity) > 3 else {}
                }
            elif isinstance(entity, dict):
                return entity
            else:
                # Handle unexpected type
                logger.warning(f"Unexpected entity type for {entity_id}: {type(entity)}")
                return {
                    'id': entity_id,
                    'name': str(entity),
                    'entity_type': 'unknown',
                    'attributes': {},
                    'metadata': {}
                }
        except Exception as e:
            logger.error(f"Error getting entity {entity_id}: {str(e)}")
            return None

    def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID (alias for get_entity)."""
        return self.get_entity(entity_id)
    
    def get_entity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get entity by name."""
        name_lower = name.lower()
        for entity in self.entities.values():
            if entity['name'].lower() == name_lower:
                return entity
        return None
    
    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        file_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search entities by name or attributes."""
        results = []
        query = query.lower()
        
        for entity_id, entity in self.entities.items():
            try:
                # Skip if entity_type doesn't match
                if entity_type and entity.get('entity_type') != entity_type:
                    continue
                    
                # Apply filters if provided
                if filters:
                    skip = False
                    for key, value in filters.items():
                        entity_value = entity.get(key) or entity.get('attributes', {}).get(key)
                        if not entity_value or str(entity_value).lower() != str(value).lower():
                            skip = True
                            break
                    if skip:
                        continue
                
                # Check if entity matches file filter
                if file_filter and entity.get('source') != file_filter:
                    continue
                
                # Check name match
                name = entity.get('name', '')
                if name and query in name.lower():
                    results.append(self._prepare_entity_for_output(entity))
                    continue
                    
                # Check attributes
                attributes = entity.get('attributes', {})
                if any(
                    isinstance(v, str) and query in str(v).lower()
                    for v in attributes.values()
                ):
                    results.append(self._prepare_entity_for_output(entity))
                    
                if len(results) >= limit:
                    break
                    
            except Exception as e:
                logger.error(f"Error processing entity {entity_id}: {str(e)}")
                continue
        
        return results
    
    def _prepare_entity_for_output(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare entity for output by flattening attributes."""
        try:
            # Handle both dictionary and sequence types
            if isinstance(entity, (list, tuple)):
                return {
                    'id': entity[0],
                    'name': str(entity[1]) if len(entity) > 1 else 'Unknown',
                    'entity_type': str(entity[2]) if len(entity) > 2 else 'unknown',
                    'attributes': entity[3] if len(entity) > 3 else {},
                    'metadata': entity[4] if len(entity) > 4 else {}
                }
            
            # For dictionary type, ensure all fields exist
            output = {
                'id': entity.get('id', str(uuid.uuid4())),
                'name': entity.get('name', 'Unknown'),
                'entity_type': entity.get('entity_type', 'unknown'),
                **entity.get('attributes', {}),  # Flatten attributes
                'metadata': entity.get('metadata', {})  # Use 'metadata' instead of '_metadata'
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Error preparing entity for output: {str(e)}")
            return {
                'id': str(uuid.uuid4()),
                'name': 'Error',
                'entity_type': 'unknown',
                'metadata': {'error': str(e)}
            }
    
    def get_relationships(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        direction: str = 'both'
    ) -> List[Dict[str, Any]]:
        """Get relationships for an entity with direction control."""
        if entity_id not in self.entities:
            return []
        
        relationships = []
        
        # Get outgoing relationships
        if direction in ['both', 'outgoing']:
            if relationship_type:
                relationships.extend(self._relationship_index[relationship_type][entity_id])
            else:
                for rel_type in self._relationship_index:
                    relationships.extend(self._relationship_index[rel_type][entity_id])
        
        # Get incoming relationships
        if direction in ['both', 'incoming']:
            if relationship_type:
                relationships.extend(self._reverse_relationship_index[relationship_type][entity_id])
            else:
                for rel_type in self._reverse_relationship_index:
                    relationships.extend(self._reverse_relationship_index[rel_type][entity_id])
        
        return relationships
    
    def get_entities_by_generation(self, generation: str) -> List[Dict[str, Any]]:
        """Get all entities of a specific generation."""
        results = []
        for entity_id, entity in self.entities.items():
            try:
                entity_gen = entity.get('attributes', {}).get('generation')
                if entity_gen and str(entity_gen) == str(generation):
                    results.append(self._prepare_entity_for_output(entity))
            except Exception as e:
                logger.error(f"Error processing entity {entity_id}: {str(e)}")
                continue
        return results
    
    def get_entity_types(self) -> List[str]:
        """Get all entity types in the graph."""
        return list({entity['entity_type'] for entity in self.entities.values()})

    def get_relationship_types(self) -> List[str]:
        """Get all relationship types in the graph."""
        return list(self.relationship_types)
    
    def _get_elements_distribution(self) -> Dict[str, int]:
        """Get distribution of elements across entities."""
        element_counts = defaultdict(int)
        for entity in self.entities.values():
            element = entity.get('attributes', {}).get('element')
            if element:
                if isinstance(element, str):
                    # Handle special cases
                    if element.lower() == 'both':
                        element_counts['Light'] += 1
                        element_counts['Dark'] += 1
                    elif element.lower() == 'lunar':
                        element_counts['Light'] += 1
                    else:
                        element_counts[element.capitalize()] += 1
        return dict(element_counts)

    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics of the knowledge graph."""
        # Update statistics
        self._stats['total_entities'] = len(self.entities)
        self._stats['total_relationships'] = len(self.relationships)
        
        # Update relationship counts
        relationship_counts = defaultdict(int)
        for rel in self.relationships.values():
            relationship_counts[rel['type']] += 1
        self._stats['relationship_counts'].update(relationship_counts)
        
        # Update entity type counts
        entity_type_counts = defaultdict(int)
        for entity in self.entities.values():
            entity_type_counts[entity['entity_type']] += 1
        self._stats['entity_type_counts'].update(entity_type_counts)
        
        try:
            element_distribution = self._get_elements_distribution()
        except Exception as e:
            logger.warning(f"Could not get element distribution: {str(e)}")
            element_distribution = {}
        
        return {
            'total_entities': self._stats['total_entities'],
            'entity_types': dict(self._stats['entity_type_counts']),
            'total_relationships': self._stats['total_relationships'],
            'relationship_types': dict(relationship_counts),
            'element_counts': element_distribution
        }
    
    def get_related_entities(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        max_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """Get related entities with improved relationship handling."""
        if entity_id not in self.entities:
            return []
        
        related = []
        visited = set()
        
        def traverse(current_id: str, depth: int):
            if depth > max_depth or current_id in visited:
                return
            
            visited.add(current_id)
            relationships = self.get_relationships(current_id, relationship_type)
            
            for rel in relationships:
                target_id = rel['target'] if rel['source'] == current_id else rel['source']
                if target_id not in visited:
                    target_entity = self.entities[target_id]
                    related.append({
                        'entity': target_entity,
                        'relationship': rel['type'],
                        'direction': 'outgoing' if rel['source'] == current_id else 'incoming',
                        'properties': rel['properties']
                    })
                    
                    if depth < max_depth:
                        traverse(target_id, depth + 1)
        
        traverse(entity_id, 1)
        return related
    
    def get_entity_context(
        self,
        entity_id: str,
        include_relationships: bool = True,
        max_relationship_depth: int = 1
    ) -> Dict[str, Any]:
        """Get full context for an entity."""
        entity = self.get_entity(entity_id)
        if not entity:
            return {}
            
        context = {
            'entity': entity,
            'related_entities': [] if include_relationships else None
        }
        
        if include_relationships:
            context['related_entities'] = self.get_related_entities(
                entity_id,
                max_depth=max_relationship_depth
            )
            
        return context
    
    def get_entities_by_relationship(self, relationship_type: str) -> List[Dict[str, Any]]:
        """Get all entities with a specific relationship type."""
        entities = []
        for entity_id, rels in self.relationships.items():
            if any(r['type'] == relationship_type for r in rels):
                if entity_id in self.entities:
                    entities.append(self.entities[entity_id])
        return entities
    
    def get_entities_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get all entities from a specific source."""
        return [self.entities[eid] for eid in self.source_registry.get(source, [])]
    
    def merge_entities(
        self,
        source_id: str,
        target_id: str,
        strategy: str = 'keep_both'
    ) -> str:
        """Merge two entities, handling conflicts according to strategy."""
        if source_id not in self.entities or target_id not in self.entities:
            raise ValueError("Both entities must exist")
            
        source = self.entities[source_id]
        target = self.entities[target_id]
        
        # Create merged attributes based on strategy
        if strategy == 'keep_both':
            merged = {
                **target,
                **{k: v for k, v in source.items() if k not in target}
            }
        elif strategy == 'prefer_source':
            merged = {**target, **source}
        elif strategy == 'prefer_target':
            merged = {**source, **target}
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")
            
        # Create new entity
        new_id = self.add_entity(
            merged['name'],
            merged['entity_type'],
            {k: v for k, v in merged.items() if k not in ['id', 'name', 'entity_type']},
            merged.get('_metadata', {})
        )
        
        # Migrate relationships
        for _, old_target, data in self.graph.edges([source_id, target_id], data=True):
            if old_target != source_id and old_target != target_id:
                self.add_relationship(new_id, old_target, data['type'], data)
                
        for old_source, _, data in self.graph.in_edges([source_id, target_id], data=True):
            if old_source != source_id and old_source != target_id:
                self.add_relationship(old_source, new_id, data['type'], data)
        
        # Remove old entities
        self.graph.remove_node(source_id)
        self.graph.remove_node(target_id)
        del self.entities[source_id]
        del self.entities[target_id]
        
        return new_id
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export the knowledge graph to a dictionary format."""
        return {
            'entities': self.entities,
            'relationships': [
                {
                    'source': source,
                    'target': target,
                    'type': data['type'],
                    'attributes': {k: v for k, v in data.items() if k != 'type'}
                }
                for source, target, data in self.graph.edges(data=True)
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HatchyKnowledgeGraph':
        """Create a knowledge graph from exported dictionary data."""
        graph = cls()
        
        # Add entities
        for entity_id, entity_data in data['entities'].items():
            graph.entities[entity_id] = entity_data
            graph.graph.add_node(entity_id, **entity_data)
        
        # Add relationships
        for rel in data['relationships']:
            graph.add_relationship(
                rel['source'],
                rel['target'],
                rel['type'],
                rel.get('attributes', {})
            )
        
        return graph

    def _process_relationships(self, entity_id: str, entity: Dict[str, Any]):
        """Process and store entity relationships with enhanced relationship types."""
        try:
            # Evolution relationships
            if 'evolves_from' in entity:
                self.add_relationship(entity_id, entity['evolves_from'], 'evolves_from')
            
            # Element relationships
            if 'Element' in entity:
                element_id = self.resolve_entity(f"element_{entity['Element'].lower()}", 'element')
                self.add_relationship(entity_id, element_id, 'has_element')
            
            # Size/mountable relationships
            description = entity.get('Description', '').lower()
            size_keywords = ['large', 'huge', 'massive', 'giant', 'enormous', 'colossal']
            if any(kw in description for kw in size_keywords):
                # Get generation from multiple sources
                gen = None
                if 'generation' in entity.get('attributes', {}):
                    gen = str(entity['attributes']['generation'])
                elif '_metadata' in entity and 'generation' in entity['_metadata']:
                    gen = str(entity['_metadata']['generation'])
                elif 'metadata' in entity and 'generation' in entity['metadata']:
                    gen = str(entity['metadata']['generation'])
                
                # Check generation and description for mountable status
                if gen == '3' or 'final' in description or any(
                    term in description for term in 
                    ['can be ridden', 'mountable', 'rideable', 'can ride']
                ):
                    self.add_relationship(
                        entity_id,
                        'mountable',
                        'can_be_mounted',
                        metadata={'confidence': 0.9}
                    )
            
            # Faction relationships
            if 'Faction' in entity:
                faction_id = self.resolve_entity(entity['Faction'], 'faction')
                self.add_relationship(entity_id, faction_id, 'member_of')
            
            # Political relationships
            if 'Political Conflict' in entity:
                conflicts = entity['Political Conflict'].split(';')
                for conflict in conflicts:
                    conflict = conflict.strip()
                    if conflict:
                        conflict_id = self.resolve_entity(conflict, 'faction')
                        self.add_relationship(entity_id, conflict_id, 'in_conflict_with')
            
            # Location relationships
            if 'Location' in entity:
                location_id = self.resolve_entity(entity['Location'], 'location')
                self.add_relationship(entity_id, location_id, 'located_in')
            
            # Nation relationships
            if 'Nation' in entity:
                nation_id = self.resolve_entity(entity['Nation'], 'nation')
                self.add_relationship(entity_id, nation_id, 'belongs_to')
            
            # Group relationships
            if 'Groups' in entity:
                groups = entity['Groups'].split(',')
                for group in groups:
                    group = group.strip()
                    if group:
                        group_id = self.resolve_entity(group, 'faction')
                        self.add_relationship(entity_id, group_id, 'member_of')
            
            # Process explicit relationships
            for rel in entity.get('relationships', []):
                if isinstance(rel, dict) and 'target_id' in rel and 'type' in rel:
                    self.add_relationship(entity_id, rel['target_id'], rel['type'])
            
            # Process narrative relationships from description
            if 'Description' in entity:
                self._process_narrative_relationships(entity_id, entity['Description'])
            
        except Exception as e:
            logger.error(f"Error processing relationships for entity {entity_id}: {str(e)}")

    def _process_narrative_relationships(self, entity_id: str, text: str):
        """Extract relationships from narrative text."""
        patterns = {
            'allied_with': [
                r'allied with (the )?([\w\s]+)',
                r'alliance with (the )?([\w\s]+)'
            ],
            'at_war_with': [
                r'at war with (the )?([\w\s]+)',
                r'fighting against (the )?([\w\s]+)'
            ],
            'leads': [
                r'leader of (the )?([\w\s]+)',
                r'commands (the )?([\w\s]+)'
            ],
            'located_in': [
                r'in (the )?([\w\s]+ region)',
                r'within (the )?([\w\s]+ kingdom)',
                r'located in (the )?([\w\s]+)'
            ]
        }
        
        for rel_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    target_name = match.group(2).strip()
                    if target_name:
                        # Determine entity type based on relationship
                        entity_type = 'faction'
                        if 'region' in target_name.lower():
                            entity_type = 'location'
                        elif 'kingdom' in target_name.lower():
                            entity_type = 'nation'
                        
                        target_id = self.resolve_entity(target_name, entity_type)
                        self.add_relationship(entity_id, target_id, rel_type)

    def resolve_entity(self, name: str, entity_type: str) -> str:
        """Resolve or create entity by name/type with improved matching."""
        if not name:
            return None
        
        name = name.strip()
        
        # Special handling for elements
        if entity_type == 'element':
            name = name.split('_')[-1].capitalize()  # Handle 'element_fire' -> 'Fire'
            
        # First try exact match
        for ent in self.entities.values():
            if (ent['name'].lower() == name.lower() and 
                ent['entity_type'] == entity_type):
                return ent['id']
        
        # Then try partial match
        for ent in self.entities.values():
            if (name.lower() in ent['name'].lower() and 
                ent['entity_type'] == entity_type):
                return ent['id']
        
        # Special handling for elements - create if it's a core element
        if entity_type == 'element' and name.lower() in {e.lower() for e in self.core_elements}:
            new_id = str(uuid.uuid4())
            self.add_entity(
                name=name.capitalize(),
                entity_type=entity_type,
                attributes={
                    'name': name.capitalize(),
                    'symbol': self._get_element_symbol(name.lower()),
                    'is_core_element': True
                },
                metadata={'auto_generated': True, 'core_element': True}
            )
            return new_id
        
        # Create new placeholder entity if not found
        new_id = str(uuid.uuid4())
        self.add_entity(
            name=name,
            entity_type=entity_type,
            attributes={'is_placeholder': True, 'name': name},
            metadata={'auto_generated': True, 'needs_resolution': True}
        )
        
        logger.info(f"Created placeholder entity: {name} ({entity_type})")
        return new_id

    def get_entities(self, entity_type: Optional[str] = None, element: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all entities, optionally filtered by type and element."""
        entities = []
        
        for entity in self.entities.values():
            if entity_type and entity.get('entity_type') != entity_type:
                continue
            if element and entity.get('element', '').lower() != element.lower():
                continue
            entities.append(entity)
        
        return entities

    def is_valid_type(self, relationship_type: str) -> bool:
        """Check if a relationship type is valid."""
        return relationship_type in self.relationship_registry 

    def resolve_or_create_entity(
        self,
        name: str,
        entity_type: str,
        attributes: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Find or create an entity with the given name and type."""
        if not name:
            raise ValueError("Entity name cannot be empty")
            
        name = str(name).strip()
        
        # First try to find existing entity by name and type
        for entity_id, entity in self.entities.items():
            # Handle both dictionary and sequence types
            entity_name = None
            entity_type_val = None
            
            if isinstance(entity, dict):
                entity_name = entity.get('name', '').lower()
                entity_type_val = entity.get('entity_type')
            elif isinstance(entity, (list, tuple)):
                entity_name = str(entity[0]).lower() if len(entity) > 0 else None
                entity_type_val = str(entity[1]) if len(entity) > 1 else None
            
            if entity_name == name.lower() and entity_type_val == entity_type:
                return entity_id
            
        # Create new entity if not found
        merged_attributes = {
            'auto_created': True,
            'name': name,  # Ensure name is included in attributes
            **(attributes or {})
        }
        
        merged_metadata = {
            'source': 'relationship_resolution',
            'created_at': datetime.now().isoformat(),
            **(metadata or {})
        }
        
        # Special handling for element entities
        if entity_type == 'element':
            merged_attributes.update({
                'name': name.capitalize(),
                'is_core_element': name.lower() in self.core_elements,
                'symbol': self._get_element_symbol(name)
            })
        
        # Special handling for egg type entities
        elif entity_type == 'egg_type':
            merged_attributes.update({
                'name': name.capitalize(),
                'is_special_type': name.lower() in ['both', 'lunar', 'solar']
            })
        
        entity_id = str(uuid.uuid4())
        # Always store as dictionary for consistency
        self.entities[entity_id] = {
            'id': entity_id,
            'name': name,
            'entity_type': entity_type,
            'attributes': merged_attributes,
            'metadata': merged_metadata
        }
        
        # Update indexes
        self._update_indexes(entity_id, self.entities[entity_id])
        
        return entity_id

    def get_entity_name(self, entity_id: str) -> str:
        """Safely get entity name with fallback."""
        entity = self.get_entity(entity_id)  # Use get_entity which handles type conversion
        if not entity:
            return f"Unknown Entity ({entity_id})"
        return entity.get('name', f"Unnamed Entity ({entity_id})")

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID."""
        return self.entities.get(entity_id) 