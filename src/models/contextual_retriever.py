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
        # Define attribute patterns that can be searched
        self.attribute_patterns = {
            'generation': {
                'patterns': [
                    r'gen(?:eration)?\s*(\d+)',
                    r'(?:gen|generation)\s*(one|two|three|1|2|3)',
                    r'(?:first|second|third)\s+generation'
                ],
                'value_map': {
                    'one': '1', 'two': '2', 'three': '3',
                    'first': '1', 'second': '2', 'third': '3'
                }
            },
            'element': {
                'patterns': [
                    r'(fire|water|plant|dark|light|void)\s+(?:type|element)?',
                    r'(?:type|element)\s+(fire|water|plant|dark|light|void)'
                ],
                'value_map': {},  # Direct mapping
                'emoji_map': {
                    'fire': '🔥', 'water': '💧', 'plant': '🌿',
                    'dark': '🌑', 'light': '✨', 'void': '🌀'
                }
            },
            'size': {
                'patterns': [
                    r'(large|huge|massive|giant|big|enormous|colossal)',
                    r'(?:large|big)\s+enough',
                    r'(?:size|sized)\s+(large|huge|massive)'
                ],
                'attribute': 'size',
                'value': 'large',
                'related_terms': ['large', 'huge', 'massive', 'giant', 'enormous', 'colossal']
            },
            'mountable': {
                'patterns': [
                    r'(?:can\s+be\s+)?(rid(?:e|ing|eable)|mount(?:ed|able)?)',
                    r'(?:can|possible)\s+(?:to\s+)?ride',
                    r'(?:for|suitable\s+for)\s+riding'
                ],
                'attribute': 'mountable',
                'value': True,
                'related_terms': ['can be ridden', 'mountable', 'rideable', 'can ride', 'for riding']
            },
            'habitat': {
                'patterns': [
                    r'(?:live|found|habitat|home)\s+(?:in|at)\s+(\w+)',
                    r'native\s+to\s+(\w+)',
                    r'from\s+(?:the\s+)?(\w+)'
                ],
                'value_map': {}
            }
        }
        
        # Define query type patterns with priorities
        self.query_type_patterns = {
            'count': {
                'patterns': [
                    r'how\s+many',
                    r'count\s+(?:of|the)',
                    r'number\s+of',
                    r'total\s+(?:number|count)'
                ],
                'priority': 1
            },
            'description': {
                'patterns': [
                    r'(?:what|tell\s+me)\s+about',
                    r'describe',
                    r'who\s+is',
                    r'what\s+is'
                ],
                'priority': 2
            },
            'comparison': {
                'patterns': [
                    r'(?:what|which)\s+(?:is|are)\s+(?:the\s+)?(bigger|larger|strongest)',
                    r'compare\s+(?:between\s+)?(.+)\s+and\s+(.+)',
                    r'difference\s+between'
                ],
                'priority': 3
            },
            'evolution': {
                'patterns': [
                    r'evolution(?:\s+chain|\s+line|\s+path)',
                    r'how\s+does\s+\w+\s+evolve',
                    r'what\s+does\s+\w+\s+evolve\s+(?:into|from)'
                ],
                'priority': 2
            }
        }
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze query for intent and attributes with improved accuracy."""
        query_lower = query.lower()
        
        # Determine query type with priority handling
        query_type = 'standard'
        highest_priority = 0
        
        for qtype, type_info in self.query_type_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in type_info['patterns']):
                if type_info['priority'] > highest_priority:
                    query_type = qtype
                    highest_priority = type_info['priority']
        
        # Extract attributes with confidence scores
        filters = {}
        confidences = {}
        
        for attr_name, attr_config in self.attribute_patterns.items():
            for pattern in attr_config['patterns']:
                matches = re.search(pattern, query_lower)
                if matches:
                    if matches.groups():
                        value = matches.group(1)
                        # Apply value mapping if exists
                        value = attr_config.get('value_map', {}).get(value, value)
                    else:
                        value = attr_config.get('value', True)
                    
                    # Use configured attribute name if specified
                    attr_key = attr_config.get('attribute', attr_name)
                    filters[attr_key] = value
                    
                    # Calculate confidence based on pattern match position
                    match_pos = matches.start()
                    confidence = 1.0 - (match_pos / len(query_lower) * 0.3)  # Position penalty
                    confidences[attr_key] = confidence
                    break
        
        # Add entity type for Hatchy-related queries with context
        if re.search(r'hatchy|monster', query_lower):
            filters['entity_type'] = 'monster'
            confidences['entity_type'] = 1.0
        
        # Add metadata for response formatting
        metadata = {
            'emojis': {
                k: v for k, v in self.attribute_patterns.get('element', {}).get('emoji_map', {}).items()
                if k in str(filters.get('element', '')).lower()
            }
        }
        
        return {
            'query_type': query_type,
            'filters': filters,
            'confidences': confidences,
            'metadata': metadata
        }

class ContextualRetriever:
    """Retrieves relevant context using vector search and knowledge graph."""
    
    def __init__(
        self,
        knowledge_graph: HatchyKnowledgeGraph,
        vector_store: VectorStore,
        max_results: int = 5,
        max_relationship_depth: int = 2
    ):
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
        self.max_results = max_results
        self.max_relationship_depth = max_relationship_depth
        self.query_analyzer = QueryAnalyzer()

    def get_context(self, query: str) -> List[Dict[str, Any]]:
        """Get relevant context for query with improved filtering."""
        analysis = self.query_analyzer.analyze(query)
        
        if analysis['query_type'] == 'count':
            return self._handle_count_query(analysis)
        elif analysis['query_type'] == 'comparison':
            return self._handle_comparison_query(analysis)
        elif analysis['query_type'] == 'evolution':
            return self._handle_evolution_query(analysis)
        
        return self._handle_standard_query(query, analysis)
    
    def _handle_count_query(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle count queries with improved accuracy."""
        filters = analysis['filters'].copy()
        confidences = analysis.get('confidences', {})
        
        # Special handling for attribute-based filtering
        attribute_filters = ['size', 'mountable', 'habitat']
        needs_content_filtering = any(f in filters for f in attribute_filters)
        
        if needs_content_filtering:
            # Get base entities
            base_filters = {k: v for k, v in filters.items() 
                          if k not in attribute_filters}
            entities = self.knowledge_graph.search_entities("", filters=base_filters)
            
            # Filter based on content and attributes with confidence scores
            filtered_entities = []
            for entity in entities:
                desc = entity.get('description', '').lower()
                attrs = entity.get('attributes', {})
                
                # Check each attribute filter
                matches_all = True
                match_confidence = 1.0
                
                for attr in attribute_filters:
                    if attr not in filters:
                        continue
                        
                    if attr == 'size' and filters['size'] == 'large':
                        size_terms = self.query_analyzer.attribute_patterns['size']['related_terms']
                        if not any(term in desc for term in size_terms):
                            matches_all = False
                            break
                        match_confidence *= confidences.get('size', 0.8)
                            
                    elif attr == 'mountable':
                        ride_terms = self.query_analyzer.attribute_patterns['mountable']['related_terms']
                        if not any(term in desc for term in ride_terms):
                            matches_all = False
                            break
                        match_confidence *= confidences.get('mountable', 0.8)
                            
                    elif attr == 'habitat':
                        habitat = filters['habitat'].lower()
                        habitat_patterns = [
                            f"found in {habitat}",
                            f"lives in {habitat}",
                            f"habitat is {habitat}",
                            f"native to {habitat}",
                            f"from {habitat}"
                        ]
                        if not any(pattern in desc for pattern in habitat_patterns):
                            matches_all = False
                            break
                
                if matches_all:
                    entity['match_confidence'] = match_confidence
                    filtered_entities.append(entity)
            
            # Sort by confidence
            filtered_entities.sort(key=lambda x: x.get('match_confidence', 0), reverse=True)
            count = len(filtered_entities)
        else:
            count = self.knowledge_graph.get_entity_count(filters)
            filtered_entities = None
        
        # Build response with metadata
        return [{
            'text_content': "Entity count result",
            'metadata': {
                'type': 'count_result',
                **analysis.get('metadata', {})
            },
            'count': count,
            'filters': filters,
            'confidences': confidences,
            'entities': filtered_entities if needs_content_filtering else None
        }]
    
    def _handle_standard_query(
        self,
        query: str,
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Handle standard queries with enhanced context."""
        # Add generation-aware retrieval
        generation = analysis.get('filters', {}).get('generation')
        if generation:
            query = f"Gen{generation} {query}"
        
        # Hybrid search with boosted generation terms
        docs = self.vector_store.max_marginal_relevance_search(
            query, 
            k=self.max_results,
            lambda_mult=0.5,
            filter={"generation": generation} if generation else None
        )
        
        # Fallback to full-text search if no generation results
        if not docs and generation:
            docs = self.vector_store.similarity_search(query, k=self.max_results)
        
        return self._process_docs(docs)
    
    def _handle_comparison_query(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle comparison queries."""
        # Implementation for comparison queries
        pass
    
    def _handle_evolution_query(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle evolution chain queries."""
        # Implementation for evolution queries
        pass

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

    def get_world_context(self, query: str) -> str:
        """Specialized retrieval for world concepts"""
        return self.vector_store.similarity_search(
            query=query,
            k=5,
            filter={"source": "world_design"},
            score_threshold=0.7
        )

    def _process_docs(self, docs: List[Document]) -> List[Dict[str, Any]]:
        """Process retrieved documents and format for LLM consumption."""
        processed_results = []
        
        for doc in docs:
            # Extract entity information if available
            entity_id = doc.metadata.get('entity_id')
            entity_context = None
            if entity_id:
                entity_context = self.get_entity_context(entity_id)
            
            # Create formatted result
            result = {
                'text_content': doc.page_content,
                'metadata': {
                    'source': doc.metadata.get('source', 'unknown'),
                    'type': doc.metadata.get('type', 'text'),
                    'generation': doc.metadata.get('generation'),
                    'element': doc.metadata.get('element')
                }
            }
            
            # Add entity context if available
            if entity_context:
                result['entity_context'] = entity_context
            
            processed_results.append(result)
        
        return processed_results 