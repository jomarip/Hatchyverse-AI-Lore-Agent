from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from .knowledge_graph import HatchyKnowledgeGraph
from .contextual_retriever import ContextualRetriever
from .enhanced_loader import EnhancedDataLoader
from .response_validator import ResponseValidator
from ..prompts.chat_prompts import get_prompt_template
from .relationship_extractor import AdaptiveRelationshipExtractor
from .registry import RelationshipRegistry
from .context_manager import EnhancedContextManager
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generates responses using context and LLM."""
    
    def __init__(self, llm):
        self.llm = llm
        
    def generate(
        self,
        query: str,
        context: Dict[str, Any],
        history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Generate a response using context."""
        try:
            # Create prompt based on query type
            prompt = self._create_prompt(query, context, history)
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            # Process and enhance response
            enhanced = self._enhance_response(response.content, context)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return self._get_fallback_response(query)
            
    def _create_prompt(
        self,
        query: str,
        context: Dict[str, Any],
        history: Optional[List[Dict]]
    ) -> str:
        """Create an appropriate prompt based on query type and context."""
        prompt_parts = [
            "You are a Hatchyverse expert. Answer based on the following context:\n\n"
        ]
        
        # Add context sections
        if context['context']:
            prompt_parts.append("Context:")
            for ctx in context['context']:
                if isinstance(ctx['content'], dict):
                    prompt_parts.append(self._format_dict_content(ctx['content']))
                else:
                    prompt_parts.append(str(ctx['content']))
                    
        # Add relevant history if available
        if history and 'history_context' in context:
            prompt_parts.append("\nRelevant conversation history:")
            for hist in context['history_context']:
                prompt_parts.append(f"User: {hist['query']}")
                prompt_parts.append(f"Assistant: {hist['response']}")
                
        # Add query
        prompt_parts.append(f"\nQuestion: {query}")
        
        # Add response guidelines based on query type
        prompt_parts.append(self._get_response_guidelines(context['query_intent']))
        
        return "\n".join(prompt_parts)
        
    def _format_dict_content(self, content: Dict) -> str:
        """Format dictionary content for prompt."""
        lines = []
        for key, value in content.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                lines.extend(f"  {k}: {v}" for k, v in value.items())
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)
        
    def _get_response_guidelines(self, query_intent: Dict) -> str:
        """Get response guidelines based on query type."""
        guidelines = [
            "\nGuidelines:",
            "1. Only use information from the provided context",
            "2. If information is missing, say so",
            "3. Be clear about uncertainty",
            "4. Use exact quotes when possible"
        ]
        
        if query_intent['query_type'] == 'relationship':
            guidelines.extend([
                "5. Explain relationships clearly",
                "6. Note relationship confidence levels",
                "7. Mention any conflicting information"
            ])
        elif query_intent['query_type'] == 'comparison':
            guidelines.extend([
                "5. Compare entities systematically",
                "6. Note similarities and differences",
                "7. Consider multiple aspects"
            ])
            
        return "\n".join(guidelines)
        
    def _enhance_response(self, response: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the response with metadata and citations."""
        enhanced = {
            'response': response,
            'confidence': context['metadata']['confidence'],
            'sources': list(context['metadata']['sources']),
            'generated_at': datetime.now().isoformat()
        }
        
        # Add citations if we have high confidence
        if context['metadata']['confidence'] >= 0.8:
            enhanced['citations'] = self._add_citations(response, context)
            
        # Add follow-up suggestions based on coverage gaps
        if gaps := self._identify_coverage_gaps(context):
            enhanced['follow_up'] = self._generate_follow_up(gaps)
            
        return enhanced
        
    def _add_citations(self, response: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Add citations to response."""
        citations = []
        
        # Extract statements from response
        statements = response.split('. ')
        
        for statement in statements:
            # Find supporting context
            support = self._find_supporting_context(statement, context['context'])
            if support:
                citations.append({
                    'statement': statement,
                    'source': support['metadata'].get('source', 'unknown'),
                    'confidence': support['score']
                })
                
        return citations
        
    def _find_supporting_context(
        self,
        statement: str,
        context: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find context that supports a statement."""
        statement_terms = set(statement.lower().split())
        
        best_match = None
        best_score = 0
        
        for ctx in context:
            if isinstance(ctx['content'], str):
                ctx_terms = set(ctx['content'].lower().split())
                overlap = len(statement_terms & ctx_terms) / len(statement_terms)
                
                if overlap > 0.5 and overlap > best_score:
                    best_match = ctx
                    best_score = overlap
                    
        return best_match
        
    def _identify_coverage_gaps(self, context: Dict[str, Any]) -> List[str]:
        """Identify aspects that lack coverage."""
        gaps = []
        coverage = context['metadata']['coverage']
        
        if coverage['factual'] < 0.5:
            gaps.append('factual')
        if coverage['relationship'] < 0.5:
            gaps.append('relationship')
        if coverage['narrative'] < 0.5:
            gaps.append('narrative')
            
        return gaps
        
    def _generate_follow_up(self, gaps: List[str]) -> List[str]:
        """Generate follow-up questions based on coverage gaps."""
        follow_ups = []
        
        for gap in gaps:
            if gap == 'factual':
                follow_ups.append("Would you like to know more specific details about this?")
            elif gap == 'relationship':
                follow_ups.append("Should I explain how this relates to other entities?")
            elif gap == 'narrative':
                follow_ups.append("Would you like to hear more about the story behind this?")
                
        return follow_ups
        
    def _get_fallback_response(self, query: str) -> Dict[str, Any]:
        """Generate a fallback response when normal generation fails."""
        return {
            'response': "I apologize, but I'm having trouble generating a response. Could you rephrase your question?",
            'confidence': 0.0,
            'sources': [],
            'generated_at': datetime.now().isoformat(),
            'is_fallback': True
        }

class EnhancedChatbot:
    """Enhanced chatbot with improved relationship and context handling."""
    
    def __init__(
        self,
        llm: BaseLLM,
        knowledge_graph: Optional[HatchyKnowledgeGraph] = None,
        vector_store: Optional[VectorStore] = None,
        data_dir: Optional[Path] = None
    ):
        """Initialize chatbot with required components."""
        self.llm = llm
        self.knowledge_graph = knowledge_graph or HatchyKnowledgeGraph()
        self.relationship_registry = RelationshipRegistry()
        self.data_dir = data_dir or Path("data")
        self.vector_store = vector_store  # Store vector_store directly
        
        # Initialize retriever from vector store if provided
        self.retriever = None
        if vector_store:
            # Configure retriever with optimized settings
            self.retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": 8,
                    "score_threshold": 0.5,
                    "fetch_k": 20,
                    "lambda_mult": 0.5,
                    "filter_duplicates": True
                }
            )
        
        # Initialize context manager with components
        self.context_manager = EnhancedContextManager(
            self.knowledge_graph,
            self.relationship_registry,
            llm
        )
        self.response_generator = ResponseGenerator(llm)
        self.conversation_history = {}
        self.validator = ResponseValidator(self.knowledge_graph)
        
    def process_message(
        self,
        session_id: str,
        message: str
    ) -> Dict[str, Any]:
        """Process a user message and generate a response."""
        try:
            # Get conversation history
            history = self.conversation_history.get(session_id, [])
            
            # Get enhanced context
            context = self.context_manager.get_context(message, history)
            
            # Generate response
            response = self.response_generator.generate(
                message,
                context,
                history
            )
            
            # Update conversation history
            self._update_history(session_id, message, response)
            
            # Extract and learn from new relationships
            self._process_new_relationships(message, response, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Message processing failed: {str(e)}")
            return self._get_error_response(str(e))
            
    def _update_history(
        self,
        session_id: str,
        message: str,
        response: Dict[str, Any]
    ):
        """Update conversation history."""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
            
        self.conversation_history[session_id].append({
            'timestamp': datetime.now().isoformat(),
            'query': message,
            'response': response['response'],
            'context_used': response.get('sources', [])
        })
        
        # Keep only last 10 turns
        self.conversation_history[session_id] = self.conversation_history[session_id][-10:]
        
    def _process_new_relationships(
        self,
        message: str,
        response: Dict[str, Any],
        context: Dict[str, Any]
    ):
        """Extract and learn from new relationships in the conversation."""
        # Extract relationships from user message
        message_rels = self.context_manager.relationship_extractor.extract(message)
        
        # Extract relationships from response
        response_rels = self.context_manager.relationship_extractor.extract(
            response['response']
        )
        
        # Learn from high-confidence relationships
        for rel in message_rels + response_rels:
            if rel.confidence >= 0.9:
                self.knowledge_graph.add_relationship(
                    source_id=rel.source,
                    target_id=rel.target,
                    relationship_type=rel.type,
                    metadata={
                        'confidence': rel.confidence,
                        'context': rel.context,
                        'source': 'conversation'
                    }
                )
                
    def _get_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Generate an error response."""
        return {
            'response': "I encountered an error while processing your message. Please try again.",
            'error': error_msg,
            'confidence': 0.0,
            'sources': [],
            'generated_at': datetime.now().isoformat(),
            'is_error': True
        }

    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate response with enhanced context awareness and validation."""
        try:
            # Determine query type and get appropriate prompt
            query_type = self._determine_query_type(query, [])
            prompt_template = get_prompt_template(query_type)
            
            # Expand query with synonyms and related terms
            expanded_query = self._expand_query(query)
            
            # Get context based on query type
            contexts = []
            
            # Try knowledge graph lookup first for specific entities
            kg_contexts = self._get_knowledge_graph_context(expanded_query)
            if kg_contexts:
                logger.debug(f"Found {len(kg_contexts)} knowledge graph contexts")
                contexts.extend(kg_contexts)
            
            # Get vector store context - try multiple search strategies
            if hasattr(self, 'vector_store') and self.vector_store:  # Check attribute exists
                try:
                    # First try similarity search with scoring
                    if hasattr(self.vector_store, 'similarity_search_with_score'):
                        docs_and_scores = self.vector_store.similarity_search_with_score(
                            expanded_query,
                            k=8,
                            score_threshold=0.5
                        )
                        logger.debug(f"Found {len(docs_and_scores)} vector store results with scores")
                        for doc, score in docs_and_scores:
                            contexts.append({
                                'text_content': doc.page_content,
                                'metadata': {
                                    **doc.metadata,
                                    'score': score,
                                    'source': 'vector_store'
                                }
                            })
                    # Fallback to regular similarity search
                    else:
                        docs = self.vector_store.similarity_search(
                            expanded_query,
                            k=8
                        )
                        logger.debug(f"Found {len(docs)} vector store results")
                        contexts.extend([{
                            'text_content': doc.page_content,
                            'metadata': {
                                **doc.metadata,
                                'score': 0.8,  # Default score
                                'source': 'vector_store'
                            }
                        } for doc in docs])
                        
                except Exception as e:
                    logger.error(f"Vector store search failed: {str(e)}")
                    # Fallback to retriever
                    if self.retriever:
                        try:
                            docs = self.retriever.invoke(expanded_query)
                            logger.debug(f"Found {len(docs)} retriever results")
                            contexts.extend([{
                                'text_content': doc.page_content,
                                'metadata': doc.metadata
                            } for doc in docs])
                        except Exception as e2:
                            logger.error(f"Retriever fallback failed: {str(e2)}")
            
            # If still no context, try direct file search
            if not contexts:
                file_contexts = self._search_text_files(expanded_query, query_type)
                logger.debug(f"Found {len(file_contexts)} file search results")
                contexts.extend(file_contexts)
            
            logger.debug(f"Retrieved {len(contexts)} total context items")
            # Log sample of contexts for debugging
            if contexts:
                logger.debug("Sample context:")
                for ctx in contexts[:2]:  # Show first 2 contexts
                    logger.debug(f"- Source: {ctx.get('metadata', {}).get('source', 'unknown')}")
                    logger.debug(f"- Score: {ctx.get('metadata', {}).get('score', 'unknown')}")
                    logger.debug(f"- Content preview: {str(ctx.get('text_content', ''))[:100]}...")
            
            # Format context with size limits
            formatted_context = self._format_context(contexts)
            
            # Prepare prompt variables
            prompt_vars = {
                "query": query,
                "context": formatted_context
            }
            
            # Add query-type specific variables
            if query_type == 'element':
                element = self._extract_element(query)
                if element:
                    prompt_vars.update({
                        "element": element,
                        "element_emoji": self._get_element_emoji(element)
                    })
            
            # Generate response using LLM
            chain = prompt_template | self.llm | StrOutputParser()
            response_text = chain.invoke(prompt_vars)
            
            # Validate response
            validation = self.validator.validate(response_text, contexts)
            
            # Apply enhancements if needed
            if validation.get('enhancements'):
                response_text = self._enhance_response(response_text, validation['enhancements'])
            
            return {
                "response": response_text,
                "validation": validation,
                "context_used": len(contexts),
                "query_type": query_type
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._create_error_response(str(e))
            
    def _determine_query_type(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Determine the type of query for appropriate prompt selection."""
        query_lower = query.lower()
        
        # Check for location-specific queries
        locations = ['ixor', 'city', 'town', 'region', 'kingdom', 'temple']
        if any(loc in query_lower for loc in locations):
            return 'location'
        
        # Check for element-specific queries
        elements = ['fire', 'water', 'plant', 'dark', 'light', 'void']
        if any(element in query_lower for element in elements):
            return 'element'
        
        # Check for generation queries
        if any(term in query_lower for term in ['gen', 'generation']):
            return 'generation'
        
        # Check for world/location queries
        if any(term in query_lower for term in ['where', 'location', 'world', 'region', 'habitat']):
            return 'world'
        
        return 'base'
        
    def _format_context(self, contexts: List[Dict[str, Any]]) -> str:
        """Format context for LLM consumption with improved structure."""
        formatted_parts = []
        
        # Group contexts by type
        grouped_contexts = {
            'entity': [],
            'relationship': [],
            'text': [],
            'metadata': []
        }
        
        for ctx in contexts:
            if 'text_content' in ctx:
                # Add source context
                source = ctx.get('metadata', {}).get('source', '')
                if source:
                    formatted_parts.append(f"\nSource: {source}")
                formatted_parts.append(f"Content: {ctx['text_content']}")
            
            # Add entity context with relationships
            if 'entity_context' in ctx:
                entity_ctx = ctx['entity_context']
                entity = entity_ctx['entity']
                
                # Format entity details
                entity_parts = [
                    f"\nEntity: {entity['name']} ({entity['entity_type']})",
                    "Attributes:"
                ]
                
                # Sort attributes for consistent presentation
                sorted_attrs = sorted(entity['attributes'].items())
                for k, v in sorted_attrs:
                    entity_parts.append(f"- {k}: {v}")
                
                # Add relationships with confidence scores
                if 'relationships' in entity_ctx:
                    entity_parts.append("\nRelationships:")
                    for rel in entity_ctx['relationships']:
                        confidence = rel.get('metadata', {}).get('confidence', 0.0)
                        entity_parts.append(
                            f"- {rel['type']}: {rel['target_name']} "
                            f"(confidence: {confidence:.2f})"
                        )
                
                formatted_parts.extend(entity_parts)
            
            # Add metadata if relevant
            if 'metadata' in ctx:
                meta = ctx['metadata']
                meta_parts = []
                for k, v in meta.items():
                    if v and k not in ['source', 'type']:
                        meta_parts.append(f"{k}: {v}")
                if meta_parts:
                    formatted_parts.append("\nMetadata: " + ", ".join(meta_parts))
        
        return "\n".join(formatted_parts)
        
    def _extract_element(self, query: str) -> Optional[str]:
        """Extract the primary element from query."""
        elements = {
            'fire': '🔥', 'water': '💧', 'plant': '🌿',
            'dark': '🌑', 'light': '✨', 'void': '🌀'
        }
        
        query_lower = query.lower()
        for element in elements:
            if element in query_lower:
                return element
            
        return None
        
    def _get_element_emoji(self, element: str) -> str:
        """Get emoji for element type."""
        emoji_map = {
            'fire': '🔥',
            'water': '💧',
            'plant': '🌿',
            'dark': '🌑',
            'light': '✨',
            'void': '🌀'
        }
        return emoji_map.get(element.lower(), '❓')
        
    def _enhance_response(self, response: str, enhancements: List[Dict[str, Any]]) -> str:
        """Enhance response based on validation suggestions."""
        enhanced_parts = [response]
        
        # Add suggested information
        if enhancements:
            enhanced_parts.append("\n\n### Additional Information")
            for enhancement in enhancements:
                if 'suggestion' in enhancement:
                    enhanced_parts.append(f"- {enhancement['suggestion']}")
        
        return "\n".join(enhanced_parts)
        
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "response": "I apologize, but I encountered an error while processing your question.",
            "error": error_msg,
            "validation": {
                'is_valid': False,
                'issues': [{'type': 'error', 'message': error_msg}],
                'enhancements': [],
                'source_coverage': {'context_used': 0, 'coverage_score': 0.0}
            }
        }
    
    def load_data(self, file_path: str):
        """Load data from file."""
        if file_path.endswith('.csv'):
            self.data_loader.load_csv_data(file_path)
        else:
            self.data_loader.load_text_data(file_path)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the chatbot."""
        return """You are a Hatchyverse Lore Expert. Follow these rules:
        1. Only use information from the provided context
        2. If unsure, say "Based on available information..."
        3. Never invent details
        4. Cite sources when possible
        5. Be clear about relationships between entities
        6. Explain any special terms or concepts
        """
    
    def _format_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Format prompt with context."""
        prompt_parts = ["Question: " + query + "\n\nContext:"]
        
        for ctx in context:
            if 'entity' in ctx:
                entity = ctx['entity']
                prompt_parts.append(f"\n- {entity['name']} ({entity.get('entity_type', 'Unknown Type')})")
                if 'description' in entity:
                    prompt_parts.append(f"  Description: {entity['description']}")
                if 'attributes' in entity:
                    attrs = [f"{k}: {v}" for k, v in entity['attributes'].items() 
                            if k not in ['name', 'description']]
                    if attrs:
                        prompt_parts.append(f"  Attributes: {', '.join(attrs)}")
                
                # Add relationship information
                if 'relationships' in ctx:
                    rels = []
                    for rel in ctx['relationships']:
                        target = rel.get('entity', {})
                        rels.append(f"{target.get('name', 'Unknown')} ({rel.get('relationship', 'related')})")
                    if rels:
                        prompt_parts.append(f"  Related to: {', '.join(rels)}")
            
            elif 'text_content' in ctx:
                prompt_parts.append(f"\n- {ctx['text_content']}")
        
        return "\n".join(prompt_parts)
    
    def _find_unused_context(
        self,
        response: str,
        context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find context items not used in the response."""
        unused = []
        response_lower = response.lower()
        
        for ctx in context:
            entity = ctx['entity']
            name = entity.get('name', '').lower()
            
            # Check if entity name is used
            if name and name not in response_lower:
                # Check if any related entities are used
                related_used = False
                for related in ctx.get('related_entities', []):
                    related_name = related['entity'].get('name', '').lower()
                    if related_name and related_name in response_lower:
                        related_used = True
                        break
                
                if not related_used:
                    unused.append(ctx)
        
        return unused
    
    def _suggest_relationships(
        self,
        response: str,
        context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suggest relationship-based enhancements."""
        suggestions = []
        
        # Extract mentioned entities
        mentioned_entities = self._extract_entity_mentions(response)
        
        # Check for unused relationships
        for entity in mentioned_entities:
            related = self.knowledge_graph.get_related_entities(entity['id'])
            
            # Filter to relevant unused relationships
            unused_relations = []
            for rel in related:
                rel_name = rel['entity'].get('name', '').lower()
                if rel_name and rel_name not in response.lower():
                    unused_relations.append(rel)
            
            if unused_relations:
                suggestions.append({
                    'type': 'unused_relationships',
                    'entity': entity['name'],
                    'relationships': unused_relations,
                    'suggestion': f"Consider mentioning related entities for {entity['name']}: " +
                                ', '.join(r['entity']['name'] for r in unused_relations)
                })
        
        return suggestions

    def _get_location_context(self, query: str) -> List[Dict[str, Any]]:
        """Get context specific to a location query."""
        location_terms = query.lower().split()
        contexts = []
        
        # Search for location-specific content
        for term in location_terms:
            if term in ['ixor', 'city', 'town', 'region', 'kingdom', 'temple']:
                contexts.extend(self.retriever.get_context(f"location {term}"))
        
        return contexts 

    def _expand_query(self, query: str) -> str:
        """Expand query with enhanced Hatchyverse-specific terms."""
        query_lower = query.lower()
        expanded_terms = []
        
        # Generation expansions with variations
        gen_patterns = [
            (r'gen(?:eration)?\s*(\d+)', r'generation \1 first generation original'),
            (r'gen-?(\d+)', r'generation \1 gen\1')
        ]
        
        for pattern, expansion in gen_patterns:
            if match := re.search(pattern, query_lower):
                expanded_terms.extend([
                    expansion.replace('\\1', match.group(1)),
                    f"gen {match.group(1)} hatchy",
                    f"generation {match.group(1)} monsters"
                ])
        
        # Location expansions with context
        location_terms = {
            'ixor': ['crystalline city', 'quartz capital', 'crystal kingdom'],
            'omniterra': ['world', 'realm', 'universe', 'setting'],
            'crystal lake': ['water region', 'lake area', 'crystal waters']
        }
        
        for location, expansions in location_terms.items():
            if location in query_lower:
                expanded_terms.extend(expansions)
        
        # Element expansions with types
        elements = ['fire', 'water', 'plant', 'dark', 'light', 'void']
        for element in elements:
            if element in query_lower:
                expanded_terms.extend([
                    f'{element} type',
                    f'{element} element',
                    f'{element} hatchy',
                    f'{element} monsters'
                ])
        
        # Equipment and evolution terms
        if 'armor' in query_lower or 'weapon' in query_lower:
            expanded_terms.extend(['equipment', 'gear', 'items'])
        if 'evolution' in query_lower:
            expanded_terms.extend(['stage', 'form', 'evolved form'])
        
        # Combine with original query
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        return query
        
    def _get_knowledge_graph_context(self, query: str) -> List[Dict[str, Any]]:
        """Get context from knowledge graph for specific entities."""
        contexts = []
        query_lower = query.lower()
        
        # Check for comparison queries
        if any(term in query_lower for term in ['similar', 'compare', 'difference', 'like']):
            # Extract entity names using proper noun detection
            entities = []
            for name in re.findall(r'\b[A-Z][a-z]+\b', query):
                entity_id = self.knowledge_graph.find_entity_by_name(name, fuzzy_match=True)
                if entity_id:
                    entity = self.knowledge_graph.get_entity(entity_id)
                    if entity:
                        entities.append(entity)
                        # Get element and other key attributes
                        element = entity.get('attributes', {}).get('element') or entity.get('element')
                        if element:
                            contexts.append({
                                'text_content': f"{entity['name']} is a {element} type {entity['entity_type']}",
                                'metadata': {'source': 'knowledge_graph', 'type': 'element_info'}
                            })
                        # Get relationships
                        relationships = self.knowledge_graph.get_relationships(entity['id'])
                        for rel in relationships:
                            contexts.append({
                                'text_content': self._format_relationship(rel),
                                'metadata': {'source': 'knowledge_graph', 'type': 'relationship'}
                            })
            
            # Add comparison context if we found multiple entities
            if len(entities) > 1:
                similarities = self._find_similarities(entities)
                for similarity in similarities:
                    contexts.append({
                        'text_content': similarity,
                        'metadata': {'source': 'knowledge_graph', 'type': 'comparison'}
                    })
        
        # Check for generation-specific queries
        gen_match = re.search(r'gen(?:eration)?[\s\-]*(\d+)', query_lower)
        if gen_match:
            generation = gen_match.group(1)
            entities = self.knowledge_graph.get_entities_by_generation(generation)
            
            # Filter by element if specified
            element_match = re.search(r'(fire|water|plant|dark|light|void)', query_lower)
            if element_match and entities:
                element = element_match.group(1).capitalize()
                entities = [e for e in entities if 
                          e.get('attributes', {}).get('element', '').lower() == element.lower() or
                          e.get('element', '').lower() == element.lower()]
            
            if entities:
                contexts.append({
                    'text_content': f"Generation {generation} {element_match.group(1).capitalize() if element_match else ''} Hatchies count: {len(entities)}",
                    'metadata': {'source': 'knowledge_graph', 'type': 'generation_info'}
                })
                # Add all matching entities
                for entity in entities:
                    contexts.append({
                        'text_content': self._format_entity_content(entity),
                        'metadata': {'source': 'knowledge_graph', 'type': 'entity'}
                    })
        
        # Check for mountable/rideable queries
        if any(term in query_lower for term in ['rideable', 'mountable', 'can be ridden']):
            # Get all entities and filter by mountable attribute and generation if specified
            all_entities = self.knowledge_graph.get_entities()
            mountable = []
            for entity in all_entities:
                # Check mountable attribute and description for size indicators
                is_mountable = (
                    entity.get('attributes', {}).get('mountable', False) or
                    any(term in str(entity.get('attributes', {}).get('description', '')).lower() 
                        for term in ['large', 'huge', 'massive', 'giant', 'can be ridden', 'mountable'])
                )
                if is_mountable:
                    # Filter by generation if specified in query
                    entity_gen = entity.get('attributes', {}).get('generation')
                    if 'gen1' in query_lower and entity_gen == '1':
                        mountable.append(entity)
                    elif 'gen2' in query_lower and entity_gen == '2':
                        mountable.append(entity)
                    elif not ('gen1' in query_lower or 'gen2' in query_lower):
                        mountable.append(entity)
            
            if mountable:
                contexts.append({
                    'text_content': f"Found {len(mountable)} rideable Hatchies",
                    'metadata': {'source': 'knowledge_graph', 'type': 'mountable_info'}
                })
                for entity in mountable:
                    contexts.append({
                        'text_content': self._format_entity_content(entity),
                        'metadata': {'source': 'knowledge_graph', 'type': 'entity'}
                    })
        
        # Check for equipment queries
        if any(term in query_lower for term in ['armor', 'weapon', 'equipment']):
            equipment = self.knowledge_graph.search_entities(
                query,
                entity_type='equipment',
                fuzzy_match=True
            )
            if equipment:
                for item in equipment:
                    contexts.append({
                        'text_content': self._format_entity_content(item),
                        'metadata': {'source': 'knowledge_graph', 'type': 'equipment'}
                    })
                    # Get related entities
                    related = self.knowledge_graph.get_related_entities(item['id'])
                    for rel in related:
                        contexts.append({
                            'text_content': self._format_relationship(rel),
                            'metadata': {'source': 'knowledge_graph', 'type': 'relationship'}
                        })
        
        return contexts

    def _format_entity_content(self, entity: Dict[str, Any]) -> str:
        """Format entity information for response context."""
        elements = []
        
        # Add name and type header
        name = entity.get('name', 'Unknown')
        entity_type = entity.get('entity_type', 'Unknown')
        elements.append(f"## {name}")
        elements.append(f"**Type:** {entity_type}")
        
        # Add element with emoji if present
        element = None
        if 'element' in entity:
            element = entity['element']
        elif 'element' in entity.get('attributes', {}):
            element = entity['attributes']['element']
            
        if element:
            emoji = self._get_element_emoji(element)
            elements.append(f"**Element:** {element.capitalize()} {emoji}")
        
        # Add generation and evolution stage if present
        attrs = entity.get('attributes', {})
        if 'generation' in attrs:
            elements.append(f"**Generation:** Gen {attrs['generation']}")
        if 'evolution_stage' in attrs:
            elements.append(f"**Evolution Stage:** {attrs['evolution_stage']}")
            
        # Add mountable status if relevant
        if attrs.get('mountable') or any(
            term in str(attrs.get('description', '')).lower() 
            for term in ['large', 'huge', 'massive', 'giant', 'can be ridden', 'mountable']
        ):
            elements.append("**Rideable:** Yes")
            
        # Add other important attributes
        for key, value in attrs.items():
            if key not in ['name', 'element', 'generation', 'evolution_stage', 'mountable', 'description']:
                elements.append(f"**{key.replace('_', ' ').title()}:** {value}")
        
        # Add description if present
        description = attrs.get('description', '')
        if description:
            elements.append(f"\n**Description:** {description}")
        
        # Add relationships if available
        if hasattr(self, 'knowledge_graph') and self.knowledge_graph:
            try:
                relationships = self.knowledge_graph.get_relationships(entity['id'])
                if relationships:
                    elements.append("\n**Relationships:**")
                    for rel in relationships:
                        target_name = self.knowledge_graph.get_entity_name(rel['target_id'])
                        elements.append(f"- {rel['type']}: {target_name}")
            except Exception as e:
                logger.debug(f"Error getting relationships: {str(e)}")
        
        return "\n".join(elements)

    def _search_text_files(self, query: str, query_type: str) -> List[Dict[str, Any]]:
        """Search through text files directly when other methods fail."""
        contexts = []
        
        # Define relevant files based on query type
        file_patterns = {
            'world': [
                "Hatchy World _ world design.txt",
                "Hatchy World Comic_ Chaos saga.txt",
                "HWCS - Simplified main arc and arc suggestions.txt"
            ],
            'location': [
                "Hatchy World _ world design.txt",
                "Hatchyverse Eco Presentation v3.txt"
            ],
            'base': [
                "Hatchy World _ world design.txt",
                "Hatchy World Comic_ Chaos saga.txt",
                "HWCS - Simplified main arc and arc suggestions.txt",
                "Hatchyverse Eco Presentation v3.txt"
            ]
        }
        
        # Get relevant file patterns
        patterns = file_patterns.get(query_type, file_patterns['base'])
        
        # Search each file
        for pattern in patterns:
            try:
                file_path = self.data_dir / pattern
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Split into chunks for better context
                        chunks = self._split_content(content)
                        
                        # Search chunks for query terms
                        query_terms = set(query.lower().split())
                        for chunk in chunks:
                            chunk_terms = set(chunk.lower().split())
                            # Calculate term overlap
                            overlap = len(query_terms & chunk_terms)
                            if overlap > 0:
                                contexts.append({
                                    'text_content': chunk,
                                    'metadata': {
                                        'source': pattern,
                                        'score': overlap / len(query_terms),
                                        'type': 'text'
                                    }
                                })
            except Exception as e:
                logger.error(f"Error searching file {pattern}: {str(e)}")
                continue
        
        return contexts

    def _split_content(self, content: str, chunk_size: int = 1000) -> List[str]:
        """Split content into manageable chunks."""
        chunks = []
        sentences = content.split('.')
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            sentence_size = len(sentence)
            
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks 

    def _find_similarities(self, entities: List[Dict[str, Any]]) -> List[str]:
        """Find similarities between entities."""
        similarities = []
        
        # Compare elements
        elements = [e.get('attributes', {}).get('element') or e.get('element') for e in entities]
        if len(set(elements)) == 1 and elements[0]:
            similarities.append(f"All entities are {elements[0]} type.")
        
        # Compare types
        types = [e.get('entity_type') for e in entities]
        if len(set(types)) == 1:
            similarities.append(f"All entities are of type {types[0]}.")
        
        # Compare generations
        generations = [e.get('attributes', {}).get('generation') for e in entities]
        if len(set(generations)) == 1 and generations[0]:
            similarities.append(f"All entities are Generation {generations[0]}.")
        
        # Compare evolution stages
        stages = [e.get('attributes', {}).get('evolution_stage') for e in entities]
        if len(set(stages)) == 1 and stages[0]:
            similarities.append(f"All entities are at evolution stage {stages[0]}.")
        
        return similarities 