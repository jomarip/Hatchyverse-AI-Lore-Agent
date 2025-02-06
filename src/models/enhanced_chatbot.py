from typing import Dict, List, Any, Optional
import logging
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .knowledge_graph import HatchyKnowledgeGraph
from .contextual_retriever import ContextualRetriever
from .enhanced_loader import EnhancedDataLoader

logger = logging.getLogger(__name__)

class ResponseValidator:
    """Validate and enhance chatbot responses."""
    
    def __init__(self, knowledge_graph: HatchyKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
    
    def _extract_entity_mentions(self, text: str) -> List[Dict[str, Any]]:
        """Extract entity mentions from text."""
        mentions = []
        text_lower = text.lower()
        
        for entity_id, entity in self.knowledge_graph.entities.items():
            name = entity.get('name', '')
            if name and name.lower() in text_lower:
                mentions.append({
                    'id': entity_id,
                    'name': name,
                    'attributes': entity.get('attributes', {})
                })
        
        return mentions
    
    def validate(self, response: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate response against knowledge graph."""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'enhancements': [],
            'source_coverage': self._check_source_coverage(response, context)
        }
        
        # Check factual consistency
        consistency_check = self._check_factual_consistency(response, context)
        if not consistency_check['is_consistent']:
            validation_results['is_valid'] = False
            validation_results['issues'].extend(consistency_check['issues'])
        
        # Check for potential enhancements
        enhancements = self._suggest_enhancements(response, context)
        validation_results['enhancements'] = enhancements
        
        return validation_results
    
    def _check_factual_consistency(
        self,
        response: str,
        context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check response consistency with knowledge graph."""
        result = {
            'is_consistent': True,
            'issues': []
        }
        
        # Extract entity mentions
        entity_mentions = self._extract_entity_mentions(response)
        
        # Check each mentioned entity against knowledge graph
        for entity in entity_mentions:
            graph_entity = self.knowledge_graph.get_entity_by_id(entity['id'])
            if graph_entity:
                # Check attribute consistency
                for attr, value in entity['attributes'].items():
                    if attr in graph_entity and graph_entity[attr] != value:
                        result['is_consistent'] = False
                        result['issues'].append({
                            'type': 'attribute_mismatch',
                            'entity': entity['name'],
                            'attribute': attr,
                            'response_value': value,
                            'actual_value': graph_entity[attr]
                        })
        
        return result
    
    def _check_source_coverage(
        self,
        response: str,
        context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Check how well the response covers the provided context."""
        return {
            'context_used': len(context),
            'coverage_score': self._calculate_coverage_score(response, context)
        }
    
    def _suggest_enhancements(
        self,
        response: str,
        context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suggest potential response enhancements."""
        suggestions = []
        
        # Check for unused relevant context
        unused_context = self._find_unused_context(response, context)
        if unused_context:
            suggestions.append({
                'type': 'additional_context',
                'context': unused_context,
                'suggestion': 'Consider including information about: ' + 
                            ', '.join(c['entity']['name'] for c in unused_context)
            })
        
        # Check for relationship opportunities
        relationship_suggestions = self._suggest_relationships(response, context)
        suggestions.extend(relationship_suggestions)
        
        return suggestions
    
    def _calculate_coverage_score(
        self,
        response: str,
        context: List[Dict[str, Any]]
    ) -> float:
        """Calculate how well the response covers the context."""
        # Simple implementation - can be enhanced
        covered = 0
        for ctx in context:
            entity_name = ctx['entity'].get('name', '').lower()
            if entity_name and entity_name in response.lower():
                covered += 1
        
        return covered / len(context) if context else 0.0

class EnhancedChatbot:
    """Enhanced chatbot with knowledge graph integration."""
    
    def __init__(
        self,
        llm: BaseLLM,
        knowledge_graph: HatchyKnowledgeGraph,
        vector_store: Any
    ):
        self.llm = llm
        self.knowledge_graph = knowledge_graph
        self.retriever = ContextualRetriever(knowledge_graph, vector_store)
        self.validator = ResponseValidator(knowledge_graph)
        self.data_loader = EnhancedDataLoader(knowledge_graph)
        
        # Initialize prompt templates
        self.qa_template = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "{query}")
        ])
    
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
            context = self.retriever.get_context(query)
            logger.debug(f"Retrieved {len(context)} context items")
            
            # Format prompt with context
            prompt = self._format_prompt(query, context)
            
            # Generate response
            chain = self.qa_template | self.llm | StrOutputParser()
            response_text = chain.invoke({"query": prompt})
            
            # Validate response
            try:
                validation = self.validator.validate(response_text, context)
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
                "context_used": len(context)
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
        prompt_parts = [
            "Question: " + query,
            "\nContext:"
        ]
        
        # Add primary context
        for ctx in context:
            entity = ctx['entity']
            source_info = f"From {entity.get('_metadata', {}).get('source_file', 'unknown source')}"
            
            # Add entity information
            if 'name' in entity:
                prompt_parts.append(f"\n{source_info}:")
                prompt_parts.append(f"Name: {entity['name']}")
                if 'description' in entity:
                    prompt_parts.append(f"Description: {entity['description']}")
                
                # Add relationships if present
                if 'related_entities' in ctx:
                    relationships = ctx['related_entities']
                    if relationships:
                        prompt_parts.append("Related information:")
                        for rel in relationships:
                            prompt_parts.append(f"- {rel['relationship']}: {rel['entity']['name']}")
            
            # Add text content if present
            elif 'content' in entity:
                prompt_parts.append(f"\n{source_info}:")
                prompt_parts.append(entity['content'])
        
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