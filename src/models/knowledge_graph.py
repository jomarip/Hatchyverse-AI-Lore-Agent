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
        self.graph = nx.MultiDiGraph()
        self.entities = {}  # Entity storage
        self.relationship_types = set()
        self.source_registry = defaultdict(list)  # Track entities by source
        self.generation_cache = {}  # Cache for generation lookups
        
        # Add indexes
        self._name_index = {}  # For fast name lookups
        self._type_index = defaultdict(set)  # For fast type-based lookups
        self._attribute_index = defaultdict(lambda: defaultdict(set))  # For attribute-based lookups
        
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
        
        # Check for duplicate names
        if entity_data['name'] in self._name_index:
            issues.append(f"Entity with name '{entity_data['name']}' already exists")
        
        # Check attribute types consistency
        for attr, value in entity_data.get('attributes', {}).items():
            # Get existing entities of same type
            similar_entities = self._type_index.get(entity_data['entity_type'], set())
            for entity_id in similar_entities:
                entity = self.entities[entity_id]
                if attr in entity['attributes']:
                    if type(value) != type(entity['attributes'][attr]):
                        issues.append(
                            f"Attribute '{attr}' type mismatch: expected "
                            f"{type(entity['attributes'][attr])}, got {type(value)}"
                        )
        
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
        relationship_type: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a relationship between entities."""
        if source_id not in self.entities or target_id not in self.entities:
            logger.warning(f"Relationship not added: source {source_id} or target {target_id} not found")
            return None
        
        # Add relationship type to set
        self.relationship_types.add(relationship_type)
        
        # Generate relationship ID
        rel_id = str(uuid.uuid4())
        
        # Add edge to graph with relationship ID
        edge_attrs = {
            'id': rel_id,
            'type': relationship_type,
            **(attributes or {})
        }
        
        self.graph.add_edge(source_id, target_id, **edge_attrs)
        logger.debug(
            f"Added relationship: {self.entities[source_id]['name']} "
            f"--[{relationship_type}]--> {self.entities[target_id]['name']}"
        )
        
        return rel_id

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
        entity_id_or_name: str,
        relationship_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get relationships for an entity."""
        # Try to get entity by ID first, then by name
        if entity_id_or_name in self.entities:
            entity_id = entity_id_or_name
        else:
            entity = self.get_entity_by_name(entity_id_or_name)
            if not entity:
                return []
            entity_id = entity['id']
            
        relationships = []
        for source, target, data in self.graph.edges(entity_id, data=True):
            if relationship_type and data['type'] != relationship_type:
                continue
                
            target_entity = self.entities[target]
            relationships.append({
                'source': self.entities[source]['name'],
                'target': target_entity['name'],
                'type': data['type'],
                'attributes': {k: v for k, v in data.items() if k != 'type'}
            })
        
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            'total_entities': len(self.entities),
            'entity_types': self.get_entity_types(),
            'relationship_types': list(self.relationship_types),
            'total_relationships': self.graph.number_of_edges()
        }
    
    def get_related_entities(
        self,
        entity_id: str,
        relationship_type: Optional[str] = None,
        max_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """Get entities related to the given entity."""
        if entity_id not in self.entities:
            return []
        
        related = []
        
        # Get immediate neighbors
        for _, target_id, edge_data in self.graph.edges(entity_id, data=True):
            if relationship_type and edge_data['type'] != relationship_type:
                continue
                
            target_entity = self.entities[target_id]
            related.append({
                'entity': target_entity,
                'relationship': edge_data['type'],
                'attributes': {k: v for k, v in edge_data.items() if k != 'type'}
            })
            
        # If max_depth > 1, recursively get more distant relationships
        if max_depth > 1:
            for rel in related.copy():
                nested = self.get_related_entities(
                    rel['entity']['id'],
                    relationship_type,
                    max_depth - 1
                )
                related.extend(nested)
        
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