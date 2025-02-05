from typing import Dict, List, Any, Optional
import re
import logging
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from .knowledge_graph import HatchyKnowledgeGraph

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """Analyze queries to extract intent and entities."""
    
    def __init__(self):
        self.generation_pattern = r'gen(?:eration)?\s*(\d+)'
        self.element_pattern = r'(fire|water|plant|dark|light|void)'
        self.relationship_patterns = {
            'mountable': r'(ride|mount|riding|mounting)',
            'evolution': r'(evolve|evolution|transform)',
            'habitat': r'(live|found|habitat)',
            'ability': r'(do|can|ability|power)'
        }
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze query for intent and entities."""
        query_lower = query.lower()
        
        return {
            'generation': self._extract_generation(query_lower),
            'element': self._extract_element(query_lower),
            'relationships': self._extract_relationships(query_lower),
            'attributes': self._extract_attributes(query_lower)
        }
    
    def _extract_generation(self, query: str) -> Optional[str]:
        """Extract generation number from query."""
        match = re.search(self.generation_pattern, query)
        return match.group(1) if match else None
    
    def _extract_element(self, query: str) -> Optional[str]:
        """Extract element type from query."""
        match = re.search(self.element_pattern, query)
        return match.group(1) if match else None
    
    def _extract_relationships(self, query: str) -> List[str]:
        """Extract relationship types from query."""
        relationships = []
        for rel_type, pattern in self.relationship_patterns.items():
            if re.search(pattern, query):
                relationships.append(rel_type)
        return relationships
    
    def _extract_attributes(self, query: str) -> List[str]:
        """Extract attribute requirements from query."""
        attributes = []
        
        # Size attributes
        if any(word in query for word in ['large', 'huge', 'massive', 'giant']):
            attributes.append('large_size')
            
        # Stage attributes
        if any(word in query for word in ['final', 'ultimate', '3rd', 'third']):
            attributes.append('final_stage')
            
        return attributes

class ContextualRetriever:
    """Retrieves relevant context using vector search and knowledge graph."""
    
    def __init__(
        self,
        knowledge_graph: Any,
        vector_store: Any,
        max_results: int = 5,
        max_relationship_depth: int = 2
    ):
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
        self.max_results = max_results
        self.max_relationship_depth = max_relationship_depth
        self.query_analyzer = QueryAnalyzer()
    
    def get_context(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get relevant context for a query."""
        try:
            # First try direct entity search
            entities = self.knowledge_graph.search_entities(
                query,
                filters=filters,
                limit=self.max_results
            )
            
            entity_contexts = []
            seen_entities = set()
            
            # Get context for each entity found
            for entity in entities:
                if entity.get('id') in seen_entities:
                    continue
                    
                context = self.knowledge_graph.get_entity_context(
                    entity['id'],
                    include_relationships=True,
                    max_relationship_depth=self.max_relationship_depth
                )
                
                if context:
                    entity_contexts.append(context)
                    seen_entities.add(entity['id'])
            
            # If no direct matches, try vector search
            if not entity_contexts:
                # Extract potential entity names from query
                words = query.split()
                potential_entities = []
                for i in range(len(words)):
                    for j in range(i + 1, len(words) + 1):
                        name = " ".join(words[i:j])
                        entity = self.knowledge_graph.get_entity_by_name(name)
                        if entity:
                            potential_entities.append(entity)
                
                # Get context for each potential entity
                for entity in potential_entities:
                    if entity.get('id') in seen_entities:
                        continue
                        
                    context = self.knowledge_graph.get_entity_context(
                        entity['id'],
                        include_relationships=True,
                        max_relationship_depth=self.max_relationship_depth
                    )
                    
                    if context:
                        entity_contexts.append(context)
                        seen_entities.add(entity['id'])
                
                # If still no matches, try vector search
                if not entity_contexts:
                    docs = self.vector_store.similarity_search(
                        query,
                        k=self.max_results,
                        filter=filters
                    )
                    
                    # Extract entity mentions from documents
                    for doc in docs:
                        # Extract entity IDs from document metadata
                        entity_ids = doc.metadata.get('entity_ids', [])
                        if not entity_ids:
                            continue
                        
                        # Get context for each entity
                        for entity_id in entity_ids:
                            if entity_id in seen_entities:
                                continue
                                
                            context = self.knowledge_graph.get_entity_context(
                                entity_id,
                                include_relationships=True,
                                max_relationship_depth=self.max_relationship_depth
                            )
                            
                            if context:
                                entity_contexts.append(context)
                                seen_entities.add(entity_id)
            
            return entity_contexts
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    def get_filtered_context(
        self,
        query: str,
        entity_type: Optional[str] = None,
        relationship_types: Optional[List[str]] = None,
        source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get context filtered by entity type, relationships, or source."""
        filters = {}
        
        if entity_type:
            filters['entity_type'] = entity_type
        if source:
            filters['source'] = source
            
        contexts = self.get_context(query, filters)
        
        # Filter by relationship types if specified
        if relationship_types:
            filtered_contexts = []
            for ctx in contexts:
                if not ctx.get('related_entities'):
                    continue
                    
                matching_rels = [
                    rel for rel in ctx['related_entities']
                    if rel['relationship'] in relationship_types
                ]
                
                if matching_rels:
                    ctx_copy = ctx.copy()
                    ctx_copy['related_entities'] = matching_rels
                    filtered_contexts.append(ctx_copy)
                    
            contexts = filtered_contexts
        
        return contexts
    
    def get_entity_timeline(
        self,
        entity_id: str,
        max_events: int = 10
    ) -> List[Dict[str, Any]]:
        """Get a timeline of events related to an entity."""
        timeline = []
        
        # Get entity context with relationships
        context = self.knowledge_graph.get_entity_context(
            entity_id,
            include_relationships=True,
            max_relationship_depth=1
        )
        
        if not context:
            return timeline
            
        # Extract events from relationships
        entity = context['entity']
        related = context.get('related_entities', [])
        
        for rel in related:
            # Look for temporal relationships or events
            if rel['relationship'] in ['happened_before', 'happened_after', 'during']:
                timeline.append({
                    'entity': entity['name'],
                    'event': rel['entity']['name'],
                    'relationship': rel['relationship'],
                    'timestamp': rel['attributes'].get('timestamp'),
                    'description': rel['entity'].get('description', '')
                })
        
        # Sort by timestamp if available
        timeline.sort(
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        
        return timeline[:max_events]
    
    def get_entity_network(
        self,
        entity_id: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Get the network of entities connected to the given entity."""
        network = {
            'nodes': [],
            'edges': [],
            'central_entity': None
        }
        
        # Get central entity
        entity = self.knowledge_graph.get_entity_by_id(entity_id)
        if not entity:
            return network
            
        network['central_entity'] = entity
        seen_entities = {entity_id}
        
        def add_related_entities(current_id: str, depth: int):
            if depth > max_depth:
                return
                
            related = self.knowledge_graph.get_related_entities(
                current_id,
                max_depth=1
            )
            
            for rel in related:
                target_id = rel['entity']['id']
                if target_id not in seen_entities:
                    network['nodes'].append(rel['entity'])
                    seen_entities.add(target_id)
                    
                network['edges'].append({
                    'source': current_id,
                    'target': target_id,
                    'relationship': rel['relationship'],
                    'attributes': rel['attributes']
                })
                
                # Recursively add connected entities
                add_related_entities(target_id, depth + 1)
        
        # Start from central entity
        network['nodes'].append(entity)
        add_related_entities(entity_id, 1)
        
        return network
    
    def analyze_relationships(
        self,
        entity_ids: List[str]
    ) -> Dict[str, Any]:
        """Analyze relationships between multiple entities."""
        analysis = {
            'entities': [],
            'direct_relationships': [],
            'common_relationships': [],
            'relationship_paths': []
        }
        
        # Get entity information
        for entity_id in entity_ids:
            entity = self.knowledge_graph.get_entity_by_id(entity_id)
            if entity:
                analysis['entities'].append(entity)
        
        # Find direct relationships
        for i, entity_id in enumerate(entity_ids):
            for other_id in entity_ids[i+1:]:
                related = self.knowledge_graph.get_related_entities(
                    entity_id,
                    max_depth=1
                )
                
                for rel in related:
                    if rel['entity']['id'] == other_id:
                        analysis['direct_relationships'].append({
                            'source': entity_id,
                            'target': other_id,
                            'relationship': rel['relationship'],
                            'attributes': rel['attributes']
                        })
        
        # Find common relationships
        all_related = {}
        for entity_id in entity_ids:
            related = self.knowledge_graph.get_related_entities(
                entity_id,
                max_depth=1
            )
            all_related[entity_id] = {
                rel['entity']['id']: rel
                for rel in related
            }
        
        # Find entities that are related to multiple input entities
        common_entities = set.intersection(*[
            set(related.keys())
            for related in all_related.values()
        ])
        
        for common_id in common_entities:
            common_rel = {
                'entity': self.knowledge_graph.get_entity_by_id(common_id),
                'relationships': [
                    all_related[entity_id][common_id]
                    for entity_id in entity_ids
                ]
            }
            analysis['common_relationships'].append(common_rel)
        
        return analysis 

    def _get_entity_context(self, entity_id: str) -> str:
        """Generate natural language context for an entity."""
        entity = self.knowledge_graph.get_entity(entity_id)
        if not entity:
            return ""
            
        context = [
            f"{entity['attributes'].get('name', 'Unnamed entity')} "
            f"({entity['type']}): {entity['attributes'].get('description', '')}"
        ]
        
        # Add relationships
        for rel in self.knowledge_graph.get_relationships(entity_id):
            target = self.knowledge_graph.get_entity(rel['target'])
            if target:
                context.append(
                    f"Related to {target['attributes'].get('name', 'unknown')} "
                    f"via {rel['type']} relationship"
                )
                
        return ". ".join(context) 