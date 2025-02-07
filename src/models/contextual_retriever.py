from typing import Dict, List, Any, Optional, Set
import re
import logging
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from .knowledge_graph import HatchyKnowledgeGraph

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """Analyzes queries to extract relevant attributes and patterns."""
    
    def __init__(self):
        self.attribute_patterns = {
            'generation': {
                'patterns': [
                    r'gen(?:eration)?\s*(\d+)',
                    r'(\d+)(?:st|nd|rd|th)\s+gen(?:eration)?',
                    r'generation\s+(\d+)',
                    r'first|second|third|fourth',
                    r'gen-?(\d+)'  # Handle gen-1 format
                ],
                'value_map': {
                    'first': '1', 'second': '2', 'third': '3', 'fourth': '4'
                }
            },
            'evolution_stage': {
                'patterns': [
                    r'\b(third|3rd|final)\s+evolution',
                    r'evolution\s+stage\s+(three|3)',
                    r'fully\s+evolved',
                    r'stage\s*(\d+)',
                    r'(\d+)(?:st|nd|rd|th)\s+stage'
                ],
                'value_map': {
                    'third': '3', '3rd': '3', 'final': '3',
                    'three': '3', 'fully': '3'
                }
            },
            'equipment': {
                'patterns': [
                    r'(armor|weapon|gear)\s+(?:called|named)\s+([\w\-]+)',
                    r'([\w\-]+)\s+(?:armor|shield|helmet)',
                    r'related\s+to\s+([\w\-]+)\b',
                    r'(?:equipment|gear)\s+(?:for|from)\s+([\w\-]+)'
                ],
                'value_map': {}
            },
            'proper_nouns': {
                'patterns': [
                    r'\b([A-Z][a-z]+(?:[\s\-][A-Za-z]+)*)\b',  # Hyphenated/capitalized
                    r'(?i)\b(Ixor|Omniterra|Buzzkill|Firadactus|Firret)\b',  # Case-insensitive known names
                    r'\b([A-Z][A-Za-z\-]+)\b',  # Hyphenated proper nouns
                    r'(Gen\d+|Generation\s+\d+)',  # Generation references
                    r'(?i)\b(Crystal\s+Lake|Dark\s+Empire|Light\s+Kingdom)\b'  # Known locations
                ],
                'value_map': {}
            },
            'mountable': {
                'patterns': [
                    r'(?:can\s+be\s+)?(ridden|mounted|rideable)',
                    r'(?:suitable|appropriate)\s+for\s+riding',
                    r'large\s+enough\s+to\s+ride',
                    r'(?:can|possible)\s+(?:to\s+)?ride'
                ],
                'value': True
            },
            'count_query': {
                'patterns': [
                    r'how\s+many',
                    r'number\s+of',
                    r'count\s+of',
                    r'total\s+(?:number|count)',
                    r'list\s+(?:all|the)'  # Added for list queries
                ],
                'value': True
            }
        }
        
    def analyze(self, query: str) -> Dict[str, Any]:
        """Analyze query to extract attributes and patterns."""
        analysis = {
            'query_type': self._determine_query_type(query),
            'attributes': {},
            'proper_nouns': self._extract_proper_nouns(query),
            'is_count_query': self._is_count_query(query)
        }
        
        # Extract attributes based on patterns
        for attr_name, attr_config in self.attribute_patterns.items():
            if attr_name in ['proper_nouns', 'count_query']:
                continue
                
            for pattern in attr_config['patterns']:
                matches = re.finditer(pattern, query, re.IGNORECASE)
                for match in matches:
                    value = match.group(1)
                    if 'value_map' in attr_config and value.lower() in attr_config['value_map']:
                        value = attr_config['value_map'][value.lower()]
                    elif 'value' in attr_config:
                        value = attr_config['value']
                    analysis['attributes'][attr_name] = value
                    break
        
        return analysis
        
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of query."""
        query_lower = query.lower()
        
        if self._is_count_query(query_lower):
            return 'count'
        elif any(term in query_lower for term in ['armor', 'weapon', 'gear', 'equipment']):
            return 'equipment'
        elif any(term in query_lower for term in ['evolution', 'evolved']):
            return 'evolution'
        elif any(term in query_lower for term in ['gen', 'generation']):
            return 'generation'
        elif any(term in query_lower for term in ['where', 'location', 'found']):
            return 'location'
        
        return 'general'
        
    def _extract_proper_nouns(self, query: str) -> List[str]:
        """Extract proper nouns from query."""
        proper_nouns = []
        patterns = self.attribute_patterns['proper_nouns']['patterns']
        
        for pattern in patterns:
            matches = re.finditer(pattern, query)
            for match in matches:
                noun = match.group(1)
                if noun and noun.lower() not in ['how', 'what', 'where', 'when', 'who', 'why']:
                    proper_nouns.append(noun)
        
        return list(set(proper_nouns))  # Remove duplicates
        
    def _is_count_query(self, query: str) -> bool:
        """Check if query is asking for a count."""
        patterns = self.attribute_patterns['count_query']['patterns']
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in patterns)

class ContextualRetriever:
    """Enhanced retriever for contextual information."""
    
    def __init__(
        self,
        knowledge_graph,
        vector_store=None,
        relationship_registry=None
    ):
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
        self.relationship_registry = relationship_registry
        self.query_analyzer = QueryAnalyzer()
        
    def get_context(self, query: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Get enhanced context for a query."""
        try:
            # Analyze query
            analysis = self.query_analyzer.analyze(query)
            logger.debug(f"Query analysis: {analysis}")
            
            context = {
                'query_intent': analysis,
                'context': [],
                'metadata': {
                    'confidence': 0.0,
                    'sources': set(),
                    'coverage': {
                        'factual': 0.0,
                        'relationship': 0.0,
                        'narrative': 0.0
                    }
                }
            }
            
            # Get context based on query type
            if analysis['query_type'] == 'count':
                self._add_count_context(context, analysis)
            elif analysis['query_type'] == 'equipment':
                self._add_equipment_context(context, analysis)
            elif analysis['query_type'] == 'evolution':
                self._add_evolution_context(context, analysis)
            
            # Add context from proper nouns
            for noun in analysis['proper_nouns']:
                self._add_entity_context(context, noun)
            
            # Add vector store context if available
            if self.vector_store:
                self._add_vector_context(context, query, analysis)
            
            # Update metadata
            self._update_metadata(context)
            
            logger.debug(f"Retrieved {len(context['context'])} context items")
            return context
            
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return self._get_error_context(str(e))
            
    def _add_count_context(self, context: Dict[str, Any], analysis: Dict[str, Any]):
        """Add context for count queries."""
        if 'generation' in analysis['attributes']:
            gen = analysis['attributes']['generation']
            entities = self.knowledge_graph.get_entities_by_generation(gen)
            if entities:
                context['context'].append({
                    'content': f"Generation {gen} count: {len(entities)}",
                    'metadata': {
                        'source': 'knowledge_graph',
                        'type': 'count',
                        'score': 1.0
                    }
                })
                # Add sample entities
                for entity in entities[:5]:
                    context['context'].append({
                        'content': self._format_entity(entity),
                        'metadata': {
                            'source': 'knowledge_graph',
                            'type': 'entity',
                            'score': 0.9
                        }
                    })
                    
    def _add_equipment_context(self, context: Dict[str, Any], analysis: Dict[str, Any]):
        """Add context for equipment-related queries."""
        if 'equipment' in analysis['attributes']:
            equipment_name = analysis['attributes']['equipment']
            # Search for equipment entity
            equipment = self.knowledge_graph.find_entity_by_name(
                equipment_name,
                entity_type='equipment',
                fuzzy_match=True
            )
            if equipment:
                context['context'].append({
                    'content': self._format_entity(equipment),
                    'metadata': {
                        'source': 'knowledge_graph',
                        'type': 'equipment',
                        'score': 1.0
                    }
                })
                # Get related entities
                related = self.knowledge_graph.get_related_entities(equipment['id'])
                for rel in related:
                    context['context'].append({
                        'content': self._format_relationship(rel),
                        'metadata': {
                            'source': 'knowledge_graph',
                            'type': 'relationship',
                            'score': 0.8
                        }
                    })
                    
    def _add_evolution_context(self, context: Dict[str, Any], analysis: Dict[str, Any]):
        """Add context for evolution-related queries."""
        if 'evolution_stage' in analysis['attributes']:
            stage = analysis['attributes']['evolution_stage']
            entities = self.knowledge_graph.get_entities_by_attribute(
                'evolution_stage',
                stage
            )
            if entities:
                for entity in entities:
                    context['context'].append({
                        'content': self._format_entity(entity),
                        'metadata': {
                            'source': 'knowledge_graph',
                            'type': 'evolution',
                            'score': 0.9
                        }
                    })
                    
    def _add_entity_context(self, context: Dict[str, Any], entity_name: str):
        """Add context for a specific entity."""
        entity = self.knowledge_graph.find_entity_by_name(
            entity_name,
            fuzzy_match=True
        )
        if entity:
            context['context'].append({
                'content': self._format_entity(entity),
                'metadata': {
                    'source': 'knowledge_graph',
                    'type': 'entity',
                    'score': 1.0
                }
            })
            # Get relationships
            related = self.knowledge_graph.get_related_entities(entity['id'])
            for rel in related:
                context['context'].append({
                    'content': self._format_relationship(rel),
                    'metadata': {
                        'source': 'knowledge_graph',
                        'type': 'relationship',
                        'score': 0.8
                    }
                })
                
    def _add_vector_context(self, context: Dict[str, Any], query: str, analysis: Dict[str, Any]):
        """Add context from vector store with improved filtering and scoring."""
        try:
            # Expand query for better retrieval
            expanded_query = self._expand_query(query, analysis)
            
            # Get documents from vector store with MMR reranking
            docs = self.vector_store.similarity_search_with_score(
                expanded_query,
                k=8,  # Get more documents
                score_threshold=0.5,  # Lower threshold for better recall
                fetch_k=20,  # Fetch more candidates for filtering
                lambda_mult=0.5  # Balance diversity and relevance
            )
            
            # Process and filter documents
            processed_docs = []
            seen_content = set()  # Track unique content
            
            for doc, score in docs:
                # Skip if content is too similar to already seen content
                content_hash = self._get_content_hash(doc.page_content)
                if content_hash in seen_content:
                    continue
                seen_content.add(content_hash)
                
                # Check relevance
                if self._is_content_relevant(doc.page_content, analysis):
                    # Adjust score based on content quality
                    adjusted_score = self._adjust_score(score, doc, analysis)
                    
                    # Add context with enhanced metadata
                    context['context'].append({
                        'content': doc.page_content,
                        'metadata': {
                            'source': doc.metadata.get('source', 'vector_store'),
                            'type': 'text',
                            'score': adjusted_score,
                            'file_name': doc.metadata.get('source', '').split('/')[-1],
                            'relevance': self._calculate_relevance(doc.page_content, query)
                        }
                    })
            
            # Sort by adjusted score
            context['context'].sort(
                key=lambda x: (
                    x['metadata'].get('score', 0),
                    x['metadata'].get('relevance', 0)
                ),
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Error adding vector context: {str(e)}")
            
    def _get_content_hash(self, content: str) -> str:
        """Get a simple hash of content for deduplication."""
        # Normalize content
        normalized = ' '.join(content.lower().split())
        # Use first 100 chars for quick comparison
        return normalized[:100]
        
    def _adjust_score(self, base_score: float, doc: Document, analysis: Dict[str, Any]) -> float:
        """Adjust similarity score based on content quality."""
        score = base_score
        
        # Boost for proper noun matches
        if any(noun.lower() in doc.page_content.lower() for noun in analysis['proper_nouns']):
            score += 0.1
            
        # Boost for attribute matches
        for attr in analysis['attributes']:
            if attr.lower() in doc.page_content.lower():
                score += 0.05
                
        # Boost for query type relevance
        if analysis['query_type'] in doc.page_content.lower():
            score += 0.1
            
        # Penalize very short content
        if len(doc.page_content.split()) < 20:
            score -= 0.1
            
        return min(1.0, score)  # Cap at 1.0
        
    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate semantic relevance score."""
        # Normalize text
        content_words = set(content.lower().split())
        query_words = set(query.lower().split())
        
        # Calculate word overlap
        overlap = len(content_words & query_words)
        
        # Calculate relevance score
        if not query_words:
            return 0.0
            
        return overlap / len(query_words)
        
    def _is_content_relevant(self, content: str, analysis: Dict[str, Any]) -> bool:
        """Enhanced content relevance check."""
        # Always include content with proper nouns
        if any(noun.lower() in content.lower() for noun in analysis['proper_nouns']):
            return True
            
        # Check query-type specific relevance
        if analysis['query_type'] == 'count':
            # Look for number patterns
            number_pattern = r'\b\d+\s+(?:hatchy|hatchies|monsters|creatures)\b'
            return bool(re.search(number_pattern, content, re.IGNORECASE))
            
        elif analysis['query_type'] == 'equipment':
            equipment_terms = [
                'armor', 'weapon', 'gear', 'equipment',
                'shield', 'helmet', 'item', 'accessory'
            ]
            return any(term in content.lower() for term in equipment_terms)
            
        elif analysis['query_type'] == 'evolution':
            evolution_terms = [
                'evolution', 'evolves', 'stage', 'form',
                'final', 'evolved', 'transformation'
            ]
            return any(term in content.lower() for term in evolution_terms)
            
        elif analysis['query_type'] == 'location':
            location_terms = [
                'region', 'city', 'town', 'kingdom', 'area',
                'located', 'found', 'place', 'territory'
            ]
            return any(term in content.lower() for term in location_terms)
            
        # Include by default for general queries
        return True
        
    def _expand_query(self, query: str, analysis: Dict[str, Any]) -> str:
        """Expand query with related terms."""
        expanded_terms = []
        
        # Add generation-specific terms
        if 'generation' in analysis['attributes']:
            gen = analysis['attributes']['generation']
            expanded_terms.extend([
                f"generation {gen}",
                f"gen {gen}",
                f"{gen}st generation" if gen == '1' else f"{gen}th generation"
            ])
        
        # Add evolution-specific terms
        if 'evolution_stage' in analysis['attributes']:
            stage = analysis['attributes']['evolution_stage']
            expanded_terms.extend([
                f"stage {stage} evolution",
                "final evolution" if stage == '3' else f"evolution {stage}"
            ])
        
        # Add equipment-specific terms
        if 'equipment' in analysis['attributes']:
            equipment = analysis['attributes']['equipment']
            expanded_terms.extend([
                f"{equipment} gear",
                f"{equipment} equipment",
                f"related to {equipment}"
            ])
        
        # Combine with original query
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        return query
        
    def _format_entity(self, entity: Dict[str, Any]) -> str:
        """Format entity for context."""
        parts = [f"Name: {entity['name']}"]
        if 'entity_type' in entity:
            parts.append(f"Type: {entity['entity_type']}")
        
        # Add attributes
        attrs = []
        for key, value in entity.get('attributes', {}).items():
            if key not in ['name', 'entity_type']:
                attrs.append(f"{key}: {value}")
        if attrs:
            parts.append("Attributes: " + ", ".join(attrs))
            
        return "\n".join(parts)
        
    def _format_relationship(self, relationship: Dict[str, Any]) -> str:
        """Format relationship for context."""
        return (
            f"Relationship: {relationship['source_name']} "
            f"{relationship['relationship_type']} {relationship['target_name']}"
        )
        
    def _update_metadata(self, context: Dict[str, Any]):
        """Update context metadata."""
        if not context['context']:
            return
            
        # Calculate average confidence
        scores = [
            ctx['metadata'].get('score', 0.0)
            for ctx in context['context']
        ]
        context['metadata']['confidence'] = sum(scores) / len(scores)
        
        # Update sources
        context['metadata']['sources'].update(
            ctx['metadata'].get('source', 'unknown')
            for ctx in context['context']
        )
        
        # Calculate coverage
        types = [ctx['metadata'].get('type', 'unknown') for ctx in context['context']]
        context['metadata']['coverage'] = {
            'factual': len([t for t in types if t in ['entity', 'count']]) / len(types),
            'relationship': len([t for t in types if t == 'relationship']) / len(types),
            'narrative': len([t for t in types if t == 'text']) / len(types)
        }
        
    def _get_error_context(self, error_msg: str) -> Dict[str, Any]:
        """Create error context."""
        return {
            'query_intent': {'query_type': 'error'},
            'context': [],
            'metadata': {
                'error': error_msg,
                'confidence': 0.0,
                'sources': set(),
                'coverage': {'factual': 0.0, 'relationship': 0.0, 'narrative': 0.0}
            }
        } 