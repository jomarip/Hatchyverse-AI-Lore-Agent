import uuid
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class EnhancedDataLoader:
    """Enhanced data loader for the knowledge graph."""
    
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
    
    def load_entity(
        self,
        entity_data: Dict[str, Any],
        entity_type: str,
        relationship_mapping: Optional[Dict[str, str]] = None
    ) -> str:
        """Load an entity into the knowledge graph with relationship mapping."""
        try:
            # Extract core entity attributes
            name = entity_data.get('name', f"{entity_type}_{uuid.uuid4().hex[:8]}")
            
            # Separate relationship fields from attributes
            attributes = {}
            relationship_data = {}
            
            for key, value in entity_data.items():
                if relationship_mapping and key in relationship_mapping:
                    relationship_data[key] = value
                elif key != 'name':  # Store everything except name in attributes
                    attributes[key] = value
            
            # Add entity to knowledge graph
            entity = {
                'name': name,
                'type': entity_type,
                'attributes': attributes,
                '_metadata': {'source': 'manual_entry'}
            }
            
            entity_id = self.knowledge_graph.add_entity(entity, 'manual_entry')
            logger.debug(f"Created entity {name} with ID {entity_id}")
            
            # Process relationships if mapping provided
            if relationship_mapping:
                for source_field, rel_type in relationship_mapping.items():
                    if source_field in relationship_data:
                        target_value = relationship_data[source_field]
                        # Skip empty or invalid targets
                        if not target_value or str(target_value).lower() == "none":
                            continue
                        
                        # Create target entity if needed
                        target_name = str(target_value)
                        target_entity = self.knowledge_graph.get_entity_by_name(target_name)
                        
                        if not target_entity:
                            # Create target entity
                            target_type = self._infer_target_type(rel_type)
                            target_entity = {
                                'name': target_name,
                                'type': target_type,
                                'attributes': {'name': target_name},
                                '_metadata': {'source': 'generated'}
                            }
                            target_id = self.knowledge_graph.add_entity(
                                target_entity,
                                'generated'
                            )
                            logger.debug(f"Created target entity {target_name} with ID {target_id}")
                        else:
                            target_id = target_entity['id']
                        
                        # Add the relationship
                        self.knowledge_graph.add_relationship(
                            entity_id,
                            target_id,
                            rel_type,
                            {'source_field': source_field}
                        )
                        logger.debug(
                            f"Added relationship {rel_type} from {name} to {target_name}"
                        )
            
            logger.info(f"Successfully loaded entity: {name}")
            return entity_id
            
        except Exception as e:
            logger.error(f"Error loading entity: {str(e)}")
            raise
    
    def _infer_target_type(self, relationship_type: str) -> str:
        """Infer the target entity type based on relationship type."""
        type_mapping = {
            'evolution_source': 'monster',
            'evolves_to': 'monster',
            'located_in': 'location',
            'lives_in': 'location',
            'has_ability': 'ability',
            'uses_item': 'item',
            'belongs_to': 'trainer'
        }
        return type_mapping.get(relationship_type, 'entity') 