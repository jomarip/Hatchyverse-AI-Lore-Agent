from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import uuid
import re
from collections import defaultdict
import logging
import networkx as nx

logger = logging.getLogger(__name__)

class HatchyKnowledgeGraph:
    """Core knowledge representation for the Hatchyverse."""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities = {}  # Entity storage
        self.relationship_types = set()
        self.source_registry = defaultdict(list)  # Track entities by source
        self.generation_cache = {}  # Cache for generation lookups
        
    def add_entity(self, entity: Dict[str, Any], source: str) -> str:
        """Add entity with rich metadata and source tracking."""
        entity_id = entity.get('id') or str(uuid.uuid4())
        
        # Create a copy to avoid modifying the input
        entity_data = {
            'id': entity_id,
            'name': entity.get('name', f"Entity_{entity_id[:8]}"),
            'type': entity.get('type', 'unknown'),
            'attributes': entity.get('attributes', {}),
            '_metadata': {
                'source': source,
                'last_updated': datetime.now().isoformat(),
                **entity.get('_metadata', {})
            }
        }
        
        # Store entity
        self.entities[entity_id] = entity_data
        self.source_registry[source].append(entity_id)
        
        # Update generation cache if generation is present
        generation = entity_data['attributes'].get('generation')
        if generation:
            gen = str(generation)
            if gen not in self.generation_cache:
                self.generation_cache[gen] = set()
            self.generation_cache[gen].add(entity_id)
        
        # Add node to graph
        self.graph.add_node(entity_id, **entity_data)
        
        logger.debug(f"Added entity: {entity_data['name']} ({entity_id})")
        return entity_id
    
    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a relationship between entities."""
        if source_id not in self.entities or target_id not in self.entities:
            logger.warning(f"Relationship not added: source {source_id} or target {target_id} not found")
            return
        
        # Add relationship type to set
        self.relationship_types.add(relationship_type)
        
        # Add edge to graph
        edge_attrs = {
            'type': relationship_type,
            **(attributes or {})
        }
        
        self.graph.add_edge(source_id, target_id, **edge_attrs)
        logger.debug(
            f"Added relationship: {self.entities[source_id]['name']} "
            f"--[{relationship_type}]--> {self.entities[target_id]['name']}"
        )
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID."""
        return self.entities.get(entity_id)
    
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
                if entity_type and entity.get('type') != entity_type:
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
            'type': entity['type'],
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
        """Get all entities from a specific generation."""
        entity_ids = self.generation_cache.get(str(generation), [])
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]
    
    def get_entity_types(self) -> List[str]:
        """Get all entity types in the graph."""
        return list({entity['type'] for entity in self.entities.values()})
    
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
        entity = self.get_entity_by_id(entity_id)
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
            merged['type'],
            {k: v for k, v in merged.items() if k not in ['id', 'name', 'type']},
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

    def _extract_generation(self, source: str, entity: Dict[str, Any]) -> Optional[str]:
        """Extract generation info from source or entity data."""
        # Try filename pattern
        gen_match = re.search(r'gen(?:eration)?[\s\-_]*(\d+)', source, re.IGNORECASE)
        if gen_match:
            return gen_match.group(1)
            
        # Try entity data
        for key in ['Generation', 'generation', 'gen']:
            if key in entity:
                return str(entity[key])
                
        return None
        
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