"""Enhanced context management with intelligent retrieval and ranking."""

import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from .knowledge_graph import HatchyKnowledgeGraph
from .relationship_extractor import RelationshipRegistry, AdaptiveRelationshipExtractor

logger = logging.getLogger(__name__)

@dataclass
class QueryIntent:
    """Represents the analyzed intent of a user query."""
    query_type: str  # 'factual', 'relationship', 'comparison', etc.
    entities: List[str]
    relationships: List[Dict[str, str]]
    constraints: Dict[str, Any]
    confidence: float

@dataclass
class ContextSource:
    """Represents a source of context information."""
    source_type: str  # 'knowledge_graph', 'text', 'relationship'
    content: Any
    relevance: float
    metadata: Dict[str, Any]

class EnhancedContextManager:
    """Manages context retrieval and ranking for chat responses."""
    
    def __init__(
        self,
        knowledge_graph: HatchyKnowledgeGraph,
        relationship_registry: RelationshipRegistry,
        llm
    ):
        self.knowledge_graph = knowledge_graph
        self.relationship_registry = relationship_registry
        self.relationship_extractor = AdaptiveRelationshipExtractor(llm, relationship_registry)
        self.context_cache = {}
        self.query_patterns = self._init_query_patterns()
        
    def _init_query_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for query type classification."""
        return {
            'factual': [
                r'what is',
                r'who is',
                r'where is',
                r'when',
                r'how many'
            ],
            'relationship': [
                r'how (?:is|are) .+ related to',
                r'what is the relationship between',
                r'who (?:is|are) .+ allied with',
                r'who supports',
                r'who opposes'
            ],
            'comparison': [
                r'compare',
                r'what are the differences between',
                r'how does .+ differ from',
                r'which is (?:better|stronger|faster)'
            ],
            'narrative': [
                r'tell me about',
                r'describe',
                r'explain',
                r'what happened'
            ]
        }
        
    def get_context(
        self,
        query: str,
        history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Get relevant context for a query."""
        try:
            # 1. Analyze query intent
            query_intent = self._analyze_query(query)
            
            # 2. Get context from multiple sources
            context_sources = self._gather_context(query_intent)
            
            # 3. Rank and filter context
            ranked_context = self._rank_context(context_sources, query_intent)
            
            # 4. Enhance context
            enhanced_context = self._enhance_context(ranked_context, history)
            
            # 5. Add metadata
            result = {
                'context': enhanced_context,
                'query_intent': query_intent,
                'metadata': self._get_context_metadata(enhanced_context, query_intent)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {str(e)}")
            return self._get_fallback_context(query)
            
    def _analyze_query(self, query: str) -> QueryIntent:
        """Analyze the query to determine intent and extract key components."""
        # Determine query type
        query_type = 'general'
        max_confidence = 0.0
        
        for qtype, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query.lower()):
                    confidence = 0.8  # Base confidence for pattern match
                    if confidence > max_confidence:
                        query_type = qtype
                        max_confidence = confidence
                        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Extract relationships
        relationships = self.relationship_extractor.extract(query)
        
        # Extract constraints
        constraints = self._extract_constraints(query)
        
        return QueryIntent(
            query_type=query_type,
            entities=entities,
            relationships=[{
                'type': r.type,
                'target': r.target
            } for r in relationships],
            constraints=constraints,
            confidence=max_confidence
        )
        
    def _gather_context(self, query_intent: QueryIntent) -> List[ContextSource]:
        """Gather context from multiple sources."""
        sources = []
        
        # 1. Knowledge Graph Context
        if query_intent.entities:
            for entity in query_intent.entities:
                entity_context = self.knowledge_graph.get_entity_context(
                    entity,
                    include_relationships=True
                )
                if entity_context:
                    sources.append(ContextSource(
                        source_type='knowledge_graph',
                        content=entity_context,
                        relevance=0.9,  # High base relevance for direct entity matches
                        metadata={'entity': entity}
                    ))
                    
        # 2. Relationship Context
        if query_intent.relationships:
            for rel in query_intent.relationships:
                rel_context = self.knowledge_graph.get_relationships(
                    rel['target'],
                    relationship_type=rel['type']
                )
                if rel_context:
                    sources.append(ContextSource(
                        source_type='relationship',
                        content=rel_context,
                        relevance=0.85,
                        metadata={'relationship': rel}
                    ))
                    
        # 3. Text Context
        text_context = self.knowledge_graph.search_text_chunks(
            query_intent.query_type,
            limit=5
        )
        if text_context:
            sources.append(ContextSource(
                source_type='text',
                content=text_context,
                relevance=0.7,  # Lower base relevance for text matches
                metadata={'query_type': query_intent.query_type}
            ))
            
        return sources
        
    def _rank_context(
        self,
        sources: List[ContextSource],
        query_intent: QueryIntent
    ) -> List[Dict[str, Any]]:
        """Rank and filter context sources."""
        # Score each context piece
        scored_context = []
        for source in sources:
            base_score = source.relevance
            
            # Adjust score based on query intent
            if query_intent.query_type == 'relationship' and source.source_type == 'relationship':
                base_score *= 1.2
            elif query_intent.query_type == 'factual' and source.source_type == 'knowledge_graph':
                base_score *= 1.1
                
            # Adjust for constraint matching
            constraint_score = self._check_constraints(source.content, query_intent.constraints)
            final_score = base_score * (1 + constraint_score)
            
            scored_context.append({
                'content': source.content,
                'score': final_score,
                'metadata': {
                    **source.metadata,
                    'source_type': source.source_type
                }
            })
            
        # Sort by score and take top N
        ranked = sorted(scored_context, key=lambda x: x['score'], reverse=True)
        return ranked[:10]  # Limit to top 10 most relevant pieces
        
    def _enhance_context(
        self,
        context: List[Dict[str, Any]],
        history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Enhance context with additional information."""
        enhanced = {
            'context': context,
            'metadata': {
                'confidence': self._calculate_confidence(context),
                'coverage': self._calculate_coverage(context),
                'sources': self._get_unique_sources(context)
            }
        }
        
        # Add relevant history if available
        if history:
            history_context = self._get_relevant_history(history, context)
            if history_context:
                enhanced['history_context'] = history_context
                
        return enhanced
        
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entity mentions from query."""
        entities = []
        
        # First try exact matches from knowledge graph
        known_entities = self.knowledge_graph.get_all_entity_names()
        for entity in known_entities:
            if entity.lower() in query.lower():
                entities.append(entity)
                
        # Then try partial matches
        words = query.split()
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                phrase = ' '.join(words[i:j])
                entity = self.knowledge_graph.find_entity_by_name(phrase)
                if entity and entity not in entities:
                    entities.append(entity)
                    
        return entities
        
    def _extract_constraints(self, query: str) -> Dict[str, Any]:
        """Extract constraints from query."""
        constraints = {}
        
        # Time constraints
        time_patterns = [
            (r'before (\d{4})', 'before_year'),
            (r'after (\d{4})', 'after_year'),
            (r'during (?:the )?([\w\s]+)', 'during_period')
        ]
        
        for pattern, key in time_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                constraints[key] = match.group(1)
                
        # Location constraints
        location_patterns = [
            (r'in (?:the )?([\w\s]+)', 'location'),
            (r'near (?:the )?([\w\s]+)', 'near_location')
        ]
        
        for pattern, key in location_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                constraints[key] = match.group(1)
                
        return constraints
        
    def _check_constraints(self, content: Any, constraints: Dict[str, Any]) -> float:
        """Check how well content matches constraints."""
        if not constraints:
            return 0
            
        matches = 0
        total = len(constraints)
        
        for key, value in constraints.items():
            if isinstance(content, dict):
                if key in content and content[key] == value:
                    matches += 1
            elif isinstance(content, str):
                if value.lower() in content.lower():
                    matches += 1
                    
        return matches / total if total > 0 else 0
        
    def _calculate_confidence(self, context: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence in the context."""
        if not context:
            return 0
            
        total_score = sum(c['score'] for c in context)
        return min(total_score / len(context), 1.0)
        
    def _calculate_coverage(self, context: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate how well the context covers different aspects."""
        coverage = {
            'factual': 0.0,
            'relationship': 0.0,
            'narrative': 0.0
        }
        
        for ctx in context:
            if ctx['metadata']['source_type'] == 'knowledge_graph':
                coverage['factual'] += 0.2
            elif ctx['metadata']['source_type'] == 'relationship':
                coverage['relationship'] += 0.2
            elif ctx['metadata']['source_type'] == 'text':
                coverage['narrative'] += 0.2
                
        return {k: min(v, 1.0) for k, v in coverage.items()}
        
    def _get_unique_sources(self, context: List[Dict[str, Any]]) -> Set[str]:
        """Get unique sources used in context."""
        sources = set()
        for ctx in context:
            if 'source' in ctx['metadata']:
                sources.add(ctx['metadata']['source'])
        return sources
        
    def _get_relevant_history(
        self,
        history: List[Dict],
        current_context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get relevant conversation history."""
        relevant_history = []
        
        # Extract key terms from current context
        current_terms = set()
        for ctx in current_context:
            if isinstance(ctx['content'], dict):
                current_terms.update(self._extract_terms(ctx['content']))
            elif isinstance(ctx['content'], str):
                current_terms.update(ctx['content'].split())
                
        # Check each history item for relevance
        for item in reversed(history[-5:]):  # Look at last 5 turns
            history_terms = set(self._extract_terms(item))
            overlap = len(current_terms & history_terms) / len(current_terms) if current_terms else 0
            
            if overlap > 0.3:  # If significant term overlap
                relevant_history.append(item)
                
        return relevant_history
        
    def _extract_terms(self, content: Any) -> Set[str]:
        """Extract significant terms from content."""
        terms = set()
        
        if isinstance(content, dict):
            for value in content.values():
                if isinstance(value, str):
                    terms.update(value.lower().split())
        elif isinstance(content, str):
            terms.update(content.lower().split())
            
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        return terms - stop_words
        
    def _get_fallback_context(self, query: str) -> Dict[str, Any]:
        """Get minimal context when normal retrieval fails."""
        return {
            'context': [],
            'query_intent': QueryIntent(
                query_type='general',
                entities=[],
                relationships=[],
                constraints={},
                confidence=0.0
            ),
            'metadata': {
                'error': 'Context retrieval failed',
                'fallback': True
            }
        } 