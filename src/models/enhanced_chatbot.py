from typing import Dict, List, Any, Optional
import logging
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .knowledge_graph import HatchyKnowledgeGraph
from .contextual_retriever import ContextualRetriever
from .enhanced_loader import EnhancedDataLoader
from .response_validator import ResponseValidator
from langchain.vectorstores import VectorStore

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
        
        # Enhanced system prompt
        self.qa_template = ChatPromptTemplate.from_template("""
            You are a Hatchyverse Lore Expert. Follow these rules:
            1. ONLY use information from the provided context
            2. Format responses in a Fandom Wiki style with clear sections
            3. NEVER invent or assume details not in context
            4. Cite specific sources when available
            5. Use proper terminology (e.g. "Hatchy" not "creature" or "monster")
            6. Include relevant relationships and connections
            7. If information is not in context, say "Based on available information, [answer]"
            8. Format numbers and statistics clearly
            9. Use bullet points for lists
            10. Include a "Trivia" section if interesting facts are available
            
            Question: {query}
            
            Context:
            {context}
            
            Answer in a clear, well-structured Fandom Wiki style.
        """)
    
    def load_data(self, file_path: str):
        """Load data from file."""
        if file_path.endswith('.csv'):
            self.data_loader.load_csv_data(file_path)
        else:
            self.data_loader.load_text_data(file_path)
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate response with enhanced context awareness."""
        try:
            # Get relevant context
            contexts = self.retriever.get_context(query)
            logger.debug(f"Retrieved {len(contexts)} context items")
            
            # Check if this is a count query
            if contexts and contexts[0].get('metadata', {}).get('type') == 'count_result':
                count_info = contexts[0]
                filters = count_info['filters']
                count = count_info['count']
                
                # Format count response
                if 'generation' in filters:
                    response_text = f"There are {count} Generation {filters['generation']} Hatchies in the database."
                    if count > 0:
                        response_text += "\nWould you like to know more about any specific Gen{} Hatchy?".format(filters['generation'])
                else:
                    response_text = f"There are {count} Hatchies matching your criteria."
                
                return {
                    "response": response_text,
                    "validation": {
                        'is_valid': True,
                        'issues': [],
                        'enhancements': [],
                        'source_coverage': {'context_used': 1, 'coverage_score': 1.0}
                    },
                    "context_used": 1
                }
            
            # Format context for prompt
            formatted_context = []
            for ctx in contexts:
                if 'entity' in ctx:
                    # Format entity context
                    entity = ctx['entity']
                    text = f"Entity: {entity['name']}\n"
                    text += f"Type: {entity.get('entity_type', 'Unknown')}\n"
                    
                    # Get attributes safely
                    attrs = entity.get('attributes', {})
                    if 'element' in attrs:
                        text += f"Element: {attrs['element']}\n"
                    if 'description' in attrs:
                        text += f"Description: {attrs['description']}\n"
                    
                    # Add relationships
                    if 'relationships' in ctx and ctx['relationships']:
                        text += "Relationships:\n"
                        for rel in ctx['relationships']:
                            text += f"- {rel['type']}: {rel['target_name']}\n"
                    
                    formatted_context.append(text)
                elif 'text_content' in ctx:
                    # Format text content
                    formatted_context.append(f"Source: {ctx['metadata'].get('type', 'Unknown')}\n{ctx['text_content']}")
            
            context_text = "\n\n".join(formatted_context) if formatted_context else "No specific context found."
            
            # Generate response
            chain = self.qa_template | self.llm | StrOutputParser()
            response_text = chain.invoke({
                "query": query,
                "context": context_text
            })
            
            # Validate response
            try:
                validation = self.validator.validate(response_text, contexts)
            except Exception as ve:
                logger.error(f"Validation error: {str(ve)}")
                validation = {
                    'is_valid': False,
                    'issues': [{'type': 'validation_error', 'message': str(ve)}],
                    'enhancements': [],
                    'source_coverage': {'context_used': 0, 'coverage_score': 0.0}
                }
            
            # Format final response
            return {
                "response": response_text,
                "validation": validation,
                "context_used": len(contexts)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error while processing your question.",
                "error": str(e),
                "validation": {
                    'is_valid': False,
                    'issues': [{'type': 'error', 'message': str(e)}],
                    'enhancements': [],
                    'source_coverage': {'context_used': 0, 'coverage_score': 0.0}
                }
            }
    
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