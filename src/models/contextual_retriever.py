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
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query type and extract parameters."""
        # Check for count queries
        query_lower = query.lower()
        if 'how many' in query_lower:
            filters = {}
            
            # Check for generation
            if 'gen1' in query_lower or 'generation 1' in query_lower:
                filters['generation'] = '1'
            elif 'gen2' in query_lower or 'generation 2' in query_lower:
                filters['generation'] = '2'
            
            # Check for type
            if 'monster' in query_lower or 'hatchy' in query_lower:
                filters['entity_type'] = 'monster'
                
            return {
                'query_type': 'count',
                'filters': filters
            }
            
        # Default to standard query
        return {
            'query_type': 'standard',
            'filters': {}
        }

    def get_context(self, query: str) -> List[Dict[str, Any]]:
        """Get relevant context for query."""
        analysis = self.analyze_query(query)
        
        if analysis['query_type'] == 'count':
            count = self.knowledge_graph.get_entity_count(analysis['filters'])
            return [{
                'text_content': f"There are {count} entities matching the specified criteria.",
                'metadata': {'type': 'count_result'},
                'count': count,
                'filters': analysis['filters']
            }]
        
        # Original context retrieval logic
        contexts = []
        
        # Get relevant documents from vector store
        docs = self.vector_store.similarity_search(query, k=5)
        for doc in docs:
            context = {
                'text_content': doc.page_content,
                'metadata': doc.metadata
            }
            
            # If document has an entity_id, get entity details
            if 'entity_id' in doc.metadata:
                entity = self.knowledge_graph.get_entity_by_id(doc.metadata['entity_id'])
                if entity:
                    context['entity'] = entity
                    # Get relationships for entity
                    context['relationships'] = self.knowledge_graph.get_entity_relationships(doc.metadata['entity_id'])
            
            contexts.append(context)
        
        return contexts
    
    def get_entity_context(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get context for a specific entity."""
        try:
            entity = self.knowledge_graph.get_entity(entity_id)
            if not entity:
                return None
            
            context = {
                'entity': entity,
                'relationships': self.knowledge_graph.get_entity_relationships(entity_id),
                'similar_entities': self._get_similar_entities(entity)
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting entity context: {str(e)}")
            return None
    
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

    def _get_similar_entities(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get similar entities based on vector similarity."""
        try:
            # Create a query from entity attributes
            query_parts = [
                entity['name'],
                entity.get('description', ''),
                ' '.join(f"{k}: {v}" for k, v in entity.get('attributes', {}).items())
            ]
            query = ' '.join(query_parts)
            
            # Get similar documents from vector store
            similar_docs = self.vector_store.similarity_search(
                query,
                k=self.max_results
            )
            
            # Convert documents to entities
            similar_entities = []
            for doc in similar_docs:
                entity_id = doc.metadata.get('entity_id')
                if entity_id and entity_id != entity['id']:
                    similar_entity = self.knowledge_graph.get_entity(entity_id)
                    if similar_entity:
                        similar_entities.append(similar_entity)
            
            return similar_entities
            
        except Exception as e:
            logger.error(f"Error getting similar entities: {str(e)}")
            return [] 