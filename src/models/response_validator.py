from typing import Dict, List, Any, Optional
import logging
from .knowledge_graph import HatchyKnowledgeGraph

logger = logging.getLogger(__name__)

class ResponseValidator:
    """Validate and enhance chatbot responses."""
    
    def __init__(self, knowledge_graph: HatchyKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
    
    def validate(self, response: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate response against knowledge graph."""
        try:
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
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            return {
                'is_valid': False,
                'issues': [{'type': 'error', 'message': str(e)}],
                'enhancements': [],
                'source_coverage': {'context_used': 0, 'coverage_score': 0.0}
            }
    
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
        for mention in entity_mentions:
            graph_entity = self.knowledge_graph.get_entity_by_id(mention['id'])
            if graph_entity:
                # Check attribute consistency
                for attr, value in mention['attributes'].items():
                    if attr in graph_entity['attributes'] and graph_entity['attributes'][attr] != value:
                        result['is_consistent'] = False
                        result['issues'].append({
                            'type': 'attribute_mismatch',
                            'entity': mention['name'],
                            'attribute': attr,
                            'response_value': value,
                            'actual_value': graph_entity['attributes'][attr]
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
    
    def _calculate_coverage_score(
        self,
        response: str,
        context: List[Dict[str, Any]]
    ) -> float:
        """Calculate how well the response covers the context."""
        covered = 0
        response_lower = response.lower()
        
        for ctx in context:
            entity = ctx.get('entity', {})
            name = entity.get('name', '').lower()
            
            if name and name in response_lower:
                covered += 1
                continue
                
            # Check for attribute coverage
            attrs = entity.get('attributes', {})
            for value in attrs.values():
                if isinstance(value, str) and value.lower() in response_lower:
                    covered += 0.5
                    break
        
        return covered / len(context) if context else 0.0
    
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
    
    def _find_unused_context(
        self,
        response: str,
        context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find context items not used in the response."""
        unused = []
        response_lower = response.lower()
        
        for ctx in context:
            entity = ctx.get('entity', {})
            name = entity.get('name', '').lower()
            
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
        for mention in mentioned_entities:
            related = self.knowledge_graph.get_relationships(mention['id'])
            
            # Filter to relevant unused relationships
            unused_relations = []
            response_lower = response.lower()
            
            for rel in related:
                target_entity = self.knowledge_graph.get_entity_by_id(rel['target'])
                if target_entity:
                    target_name = target_entity['name'].lower()
                    if target_name not in response_lower:
                        unused_relations.append({
                            'relationship': rel['type'],
                            'entity': target_entity['name']
                        })
            
            if unused_relations:
                suggestions.append({
                    'type': 'unused_relationships',
                    'entity': mention['name'],
                    'relationships': unused_relations,
                    'suggestion': f"Consider mentioning related entities for {mention['name']}: " +
                                ', '.join(f"{r['entity']} ({r['relationship']})" for r in unused_relations)
                }) 