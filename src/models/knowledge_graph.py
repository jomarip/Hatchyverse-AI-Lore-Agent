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
        self.relationships = []  # Store all relationships
        
        # Add indexes
        self._name_index = {}  # For fast name lookups
        self._type_index = defaultdict(set)  # For fast type-based lookups
        self._attribute_index = defaultdict(lambda: defaultdict(set))  # For attribute-based lookups
        self._relationship_index = defaultdict(list)  # For relationship lookups
        
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
            'electric', 'earth', 'air', 'metal', 'chaos', 'order'
        }
        self._init_core_entities()
        
    def _init_core_entities(self):
        """Create base entities for core game concepts"""
        for element in self.core_elements:
            self.add_entity(
                name=element.capitalize(),
                entity_type="element",
                attributes={
                    "name": element.capitalize(),
                    "symbol": self._get_element_symbol(element)
                }
            )
    
    def _get_element_symbol(self, element: str) -> str:
        """Maps elements to display symbols"""
        symbols = {
            'fire': '🔥', 'water': '💧', 'plant': '🌿',
            'void': '🌌', 'light': '✨', 'dark': '🌑',
            'electric': '⚡', 'earth': '🌍', 'air': '🌪️',
            'metal': '⚙️', 'chaos': '🌀', 'order': '⚖️'
        }
        return symbols.get(element, '❓')

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
        # Update name index
        self._name_index[entity_data['name']] = entity_id
        
        # Update type index
        entity_type = entity_data['entity_type']
        if entity_type not in self._type_index:
            self._type_index[entity_type] = set()
        self._type_index[entity_type].add(entity_id)
        
        # Update attribute index
        for attr, value in entity_data.get('attributes', {}).items():
            if isinstance(value, (str, int, float, bool)):
                self._attribute_index[attr][str(value)].add(entity_id)

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
        """Extract generation info from source or entity data."""
        # Try entity data first
        generation = entity.get('attributes', {}).get('generation')
        if generation:
            return str(generation)
            
        # Try filename pattern if source is provided
        if source:
            gen_match = re.search(r'gen(?:eration)?[\s\-_]*(\d+)', source, re.IGNORECASE)
            if gen_match:
                return gen_match.group(1)
                
        return None

    def add_entity(
        self,
        name: str,
        entity_type: str,
        attributes: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None
    ) -> str:
        """Add an entity to the knowledge graph with validation."""
        if not name or not entity_type:
            raise ValueError("Name and entity_type are required and cannot be empty")
            
        # Prepare entity data
        entity_data = {
            'name': name,
            'entity_type': entity_type,
            'attributes': attributes or {},
            'metadata': metadata or {},
            'source': source
        }
        
        # Validate data
        is_valid, validation_errors = self._validate_entity_data(entity_data)
        if not is_valid:
            raise ValueError(f"Invalid entity data: {', '.join(validation_errors)}")
        
        # Check consistency
        is_consistent, consistency_issues = self._check_data_consistency(entity_data)
        if not is_consistent:
            raise ValueError(f"Data consistency issues: {', '.join(consistency_issues)}")
        
        # Generate ID and create entity
        entity_id = str(uuid.uuid4())
        entity = {
            'id': entity_id,
            'name': name,
            'entity_type': entity_type,
            'attributes': attributes or {},
            '_metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        
        if source:
            entity['_metadata']['source'] = source
            self.source_registry[source].append(entity_id)
        
        # Store entity and update indexes
        self.entities[entity_id] = entity
        self._update_indexes(entity_id, entity)
        
        # Add to graph
        self.graph.add_node(entity_id, **entity)
        
        # Extract and cache generation if present
        generation = self._extract_generation(source, entity)
        if generation:
            if generation not in self.generation_cache:
                self.generation_cache[generation] = set()
            self.generation_cache[generation].add(entity_id)
            # Also add generation to attributes if not already present
            if 'generation' not in attributes:
                attributes['generation'] = generation
        
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
        """Get entity by name using index."""
        entity_id = self._name_index.get(name)
        return self.entities.get(entity_id) if entity_id else None

    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
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
        relationship_type: str
    ):
        """Add a relationship between two entities."""
        try:
            logger.debug(f"=== Starting relationship creation ===")
            logger.debug(f"Source ID: {source_id}")
            logger.debug(f"Target ID: {target_id}")
            logger.debug(f"Relationship Type: {relationship_type}")
            
            # Validate source and target exist
            source_entity = self.get_entity(source_id)
            target_entity = self.get_entity(target_id)
            
            if not source_entity or not target_entity:
                error_msg = f"Source or target entity not found: {source_id} -> {target_id}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.debug(f"Source entity found: {source_entity['name']}")
            logger.debug(f"Target entity found: {target_entity['name']}")
            
            # Create relationship object
            relationship = {
                'source_id': source_id,
                'target_id': target_id,
                'type': relationship_type,
                'metadata': {}
            }
            
            # Add to relationships list
            self.relationships.append(relationship)
            logger.debug(f"Added relationship to relationships list. Total relationships: {len(self.relationships)}")
            
            # Update relationship index
            self._relationship_index[source_id].append(relationship)
            self._relationship_index[target_id].append(relationship)
            logger.debug(f"Updated relationship index for both source and target")
            
            # Add to relationship types set and update statistics
            self.relationship_types.add(relationship_type)
            self._stats['total_relationships'] += 1
            self._stats['relationship_counts'][relationship_type] += 1
            
            logger.debug(f"Added relationship type to set. Current types: {self.relationship_types}")
            logger.debug(f"Updated statistics. Total relationships: {self._stats['total_relationships']}")
            logger.debug(f"Relationship counts by type: {dict(self._stats['relationship_counts'])}")
            
            # Add to graph structure
            self.graph.add_edge(
                source_id,
                target_id,
                type=relationship_type,
                metadata=relationship['metadata']
            )
            logger.debug("Added relationship to graph structure")
            
            logger.info(f"Successfully created relationship: {source_entity['name']} -{relationship_type}-> {target_entity['name']}")
            logger.debug("=== Relationship creation completed ===")
            
        except Exception as e:
            logger.error(f"Error creating relationship: {str(e)}")
            raise

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

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
        limit: int = 10
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
        output = {
            'id': entity['id'],
            'name': entity['name'],
            'entity_type': entity['entity_type'],
            **entity['attributes'],  # Flatten attributes to top level
            '_metadata': entity['_metadata']
        }
        return output
    
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
        
        # Get relationships where entity is source
        if direction in ['both', 'outgoing']:
            for rel in self._relationship_index[entity_id]:
                if rel['source'] == entity_id:
                    if not relationship_type or rel['type'] == relationship_type:
                        relationships.append(rel)
        
        # Get relationships where entity is target
        if direction in ['both', 'incoming']:
            for rel in self._relationship_index[entity_id]:
                if rel['target'] == entity_id:
                    if not relationship_type or rel['type'] == relationship_type:
                        relationships.append(rel)
        
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
        for rel in self.relationships:
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
        """Process and store entity relationships."""
        # Evolution relationships
        if 'evolves_from' in entity:
            self.add_relationship(entity_id, entity['evolves_from'], 'evolves_from')
            
        # Element relationships
        if 'Element' in entity:
            self.add_relationship(entity_id, f"element_{entity['Element'].lower()}", 'has_element')
            
        # Size/mountable relationships
        description = entity.get('Description', '').lower()
        if any(kw in description for kw in ['large', 'huge', 'massive']):
            gen = entity.get('_metadata', {}).get('generation')
            if gen == '3' or 'final' in description:
                self.add_relationship(entity_id, 'mountable', 'can_be_mounted')
                
        # Process explicit relationships
        for rel in entity.get('relationships', []):
            if isinstance(rel, dict) and 'target_id' in rel and 'type' in rel:
                self.add_relationship(entity_id, rel['target_id'], rel['type'])
        
    def get_entities_by_generation(self, generation: str) -> List[Dict[str, Any]]:
        """Get all entities from a specific generation."""
        if generation in self.generation_cache:
            return [self.entities[eid] for eid in self.generation_cache[generation]]
        return []
    
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
    
    def get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for an entity."""
        relationships = []
        
        # Get outgoing relationships
        for _, target, data in self.graph.edges(entity_id, data=True):
            target_entity = self.entities[target]
            relationships.append({
                'source': entity_id,
                'target': target,
                'target_name': target_entity['name'],
                'type': data['type'],
                'direction': 'outgoing',
                'attributes': {k: v for k, v in data.items() if k != 'type'}
            })
        
        # Get incoming relationships
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            source_entity = self.entities[source]
            relationships.append({
                'source': source,
                'source_name': source_entity['name'],
                'target': entity_id,
                'type': data['type'],
                'direction': 'incoming',
                'attributes': {k: v for k, v in data.items() if k != 'type'}
            })
        
        return relationships

    def get_entity_count(self, filters: Dict[str, Any] = None) -> int:
        """Count entities matching filter criteria."""
        if not filters:
            return len(self.entities)
            
        def matches_filter(entity: Dict[str, Any], key: str, value: Any) -> bool:
            """Check if entity matches a single filter criterion."""
            if key == 'entity_type':
                return entity.get('entity_type', '').lower() == str(value).lower()
            elif key == 'generation':
                return str(entity.get('attributes', {}).get('generation', '')) == str(value)
            elif key == 'element':
                entity_element = entity.get('attributes', {}).get('element', '')
                return entity_element and entity_element.lower() == str(value).lower()
            else:
                entity_value = entity.get('attributes', {}).get(key)
                if entity_value is None:
                    return False
                return str(entity_value).lower() == str(value).lower()
        
        return sum(
            1 for e in self.entities.values()
            if all(matches_filter(e, k, v) for k, v in filters.items())
        )

    def get_entities_by_element(self, element: str) -> List[Dict[str, Any]]:
        """Get all entities of a specific element."""
        element = element.lower()
        return [
            self._prepare_entity_for_output(entity)
            for entity in self.entities.values()
            if entity.get('attributes', {}).get('element', '').lower() == element
        ]

    def get_entities_by_generation(self, generation: str) -> List[Dict[str, Any]]:
        """Get entities by generation."""
        return [
            self._prepare_entity_for_output(entity)
            for entity in self.entities.values()
            if str(entity.get('attributes', {}).get('generation', '')) == str(generation)
        ]

    def find_entity_by_name(self, name: str, fuzzy_match: bool = True) -> Optional[Dict[str, Any]]:
        """Find entity by name with optional fuzzy matching."""
        if not name:
            return None
            
        name_lower = name.lower().strip()
        
        # Try exact match first
        for entity in self.entities.values():
            entity_name = entity.get('name', '').lower().strip()
            if entity_name == name_lower:
                return entity
        
        if fuzzy_match:
            # Try fuzzy matching if exact match fails
            best_match = None
            best_ratio = 0
            
            for entity in self.entities.values():
                entity_name = entity.get('name', '').lower().strip()
                # Check if name is contained or similar
                if (name_lower in entity_name or 
                    entity_name in name_lower or 
                    self._string_similarity(name_lower, entity_name) > 0.8):
                    ratio = self._string_similarity(name_lower, entity_name)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_match = entity
            
            return best_match
        
        return None

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity ratio."""
        # Simple Levenshtein-based similarity
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 0
        distance = sum(1 for a, b in zip(s1, s2) if a != b) + abs(len(s1) - len(s2))
        return 1 - (distance / max_len)

    def resolve_entity(self, name: str, entity_type: str) -> Optional[str]:
        """Fuzzy match entity by name/type with fallback creation"""
        # First try exact match
        for ent in self.entities.values():
            if ent['attributes'].get('name', '').lower() == name.lower() and ent['type'] == entity_type:
                return ent['id']
        
        # Then try partial match
        for ent in self.entities.values():
            if name.lower() in ent['attributes'].get('name', '').lower() and ent['type'] == entity_type:
                return ent['id']
        
        # Fallback: Create placeholder entity
        new_id = f"{entity_type}_{name.lower().replace(' ', '_')}"
        self.add_entity(
            entity_id=new_id,
            entity_type=entity_type,
            attributes={'name': name, 'auto_generated': True}
        )
        return new_id 