from typing import Dict, List, Any, Optional
import logging
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from .knowledge_graph import HatchyKnowledgeGraph
from .contextual_retriever import ContextualRetriever
from .enhanced_loader import EnhancedDataLoader
from .response_validator import ResponseValidator
from langchain.vectorstores import VectorStore
from ..prompts.chat_prompts import get_prompt_template

logger = logging.getLogger(__name__)

class EnhancedChatbot:
    """Enhanced chatbot with knowledge graph integration."""
    
    def __init__(
        self,
        llm: BaseLLM,
        knowledge_graph: HatchyKnowledgeGraph,
        vector_store: VectorStore
    ):
        self.llm = llm
        self.knowledge_graph = knowledge_graph
        self.retriever = ContextualRetriever(knowledge_graph, vector_store)
        self.validator = ResponseValidator(knowledge_graph)
        self.data_loader = EnhancedDataLoader(knowledge_graph)
        
    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate response with enhanced context awareness and validation."""
        try:
            # Get relevant context
            contexts = self.retriever.get_context(query)
            logger.debug(f"Retrieved {len(contexts)} context items")
            
            # Determine query type and get appropriate prompt
            query_type = self._determine_query_type(query, contexts)
            prompt_template = get_prompt_template(query_type)
            
            # Format context
            formatted_context = self._format_context(contexts)
            
            # Prepare prompt variables
            prompt_vars = {
                "query": query,
                "context": formatted_context
            }
            
            # Add query-type specific variables
            if query_type == 'element':
                element = self._extract_element(contexts)
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
            if validation['enhancements']:
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
        
        # Check for element-specific queries
        elements = ['fire', 'water', 'plant', 'dark', 'light', 'void']
        if any(f"{element} type" in query_lower or f"{element}-type" in query_lower for element in elements):
            return 'element'
            
        # Check for evolution queries
        if any(term in query_lower for term in ['evolve', 'evolution', 'evolves into', 'evolves from']):
            return 'evolution'
            
        # Check for world/location queries
        if any(term in query_lower for term in ['where', 'location', 'world', 'region', 'habitat']):
            return 'world'
            
        # Check for comparison queries
        if any(term in query_lower for term in ['compare', 'difference between', 'stronger', 'better']):
            return 'comparison'
            
        return 'base'
        
    def _format_context(self, contexts: List[Dict[str, Any]]) -> str:
        """Format context for LLM consumption."""
        formatted_parts = []
        
        for ctx in contexts:
            # Add main content
            if 'text_content' in ctx:
                formatted_parts.append(f"Content: {ctx['text_content']}")
            
            # Add entity context if available
            if 'entity_context' in ctx:
                entity_ctx = ctx['entity_context']
                entity = entity_ctx['entity']
                formatted_parts.append(
                    f"Entity: {entity['name']} ({entity['entity_type']})\n"
                    f"Attributes: {', '.join(f'{k}: {v}' for k, v in entity['attributes'].items())}"
                )
                
                # Add relationships
                if 'relationships' in entity_ctx:
                    rel_parts = []
                    for rel in entity_ctx['relationships']:
                        rel_parts.append(
                            f"- {rel['type']}: {rel['target_name']}"
                        )
                    if rel_parts:
                        formatted_parts.append("Relationships:\n" + "\n".join(rel_parts))
            
            # Add metadata
            if 'metadata' in ctx:
                meta = ctx['metadata']
                meta_parts = []
                for k, v in meta.items():
                    if v and k not in ['source', 'type']:
                        meta_parts.append(f"{k}: {v}")
                if meta_parts:
                    formatted_parts.append("Metadata: " + ", ".join(meta_parts))
        
        return "\n\n".join(formatted_parts)
        
    def _extract_element(self, contexts: List[Dict[str, Any]]) -> Optional[str]:
        """Extract the primary element from context."""
        for ctx in contexts:
            if 'metadata' in ctx and 'element' in ctx['metadata']:
                return ctx['metadata']['element']
        return None
        
    def _get_element_emoji(self, element: str) -> str:
        """Get emoji for element type."""
        emoji_map = {
            'fire': '🔥',
            'water': '💧',
            'plant': '🌿',
            'dark': '🌑',
            'light': '✨',
            'void': '🌀',
            'electric': '⚡',
            'earth': '🌍'
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