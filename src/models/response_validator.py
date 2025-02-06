"""Enhanced response validation for the Hatchyverse chatbot."""

from typing import Dict, List, Any, Optional
import logging
from .knowledge_graph import HatchyKnowledgeGraph

logger = logging.getLogger(__name__)

class ResponseValidator:
    """Validate and enhance chatbot responses."""
    
    def __init__(self, knowledge_graph: HatchyKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
    
    def validate(self, response: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate response against knowledge graph with enhanced checks."""
        try:
            # Initialize validation results
            validation_results = {
                'is_valid': True,
                'issues': [],
                'enhancements': [],
                'source_coverage': self._check_source_coverage(response, context),
                'confidence': 1.0,
                'metadata': {
                    'context_items': len(context),
                    'generation': self._extract_generation(context),
                    'elements': self._extract_elements(context),
                    'relationships': self._extract_relationships(context)
                }
            }
            
            # Validate generation consistency
            self._validate_generation(response, validation_results)
            
            # Validate element consistency
            self._validate_elements(response, validation_results)
            
            # Validate entity relationships
            self._validate_relationships(response, validation_results)
            
            # Check factual consistency
            self._validate_facts(response, context, validation_results)
            
            # Suggest enhancements
            self._suggest_enhancements(response, context, validation_results)
            
            return validation_results
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            return {
                'is_valid': False,
                'issues': [{'type': 'error', 'message': str(e)}],
                'enhancements': [],
                'source_coverage': {'context_used': 0, 'coverage_score': 0.0},
                'confidence': 0.0
            }
    
    def _extract_generation(self, context: List[Dict[str, Any]]) -> Optional[str]:
        """Extract generation info from context."""
        for ctx in context:
            if 'metadata' in ctx and 'generation' in ctx['metadata']:
                return ctx['metadata']['generation']
        return None
    
    def _extract_elements(self, context: List[Dict[str, Any]]) -> List[str]:
        """Extract element types from context."""
        elements = set()
        for ctx in context:
            if 'metadata' in ctx and 'element' in ctx['metadata']:
                elements.add(ctx['metadata']['element'])
        return list(elements)
    
    def _extract_relationships(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships from context."""
        relationships = []
        for ctx in context:
            if 'entity_context' in ctx and 'relationships' in ctx['entity_context']:
                relationships.extend(ctx['entity_context']['relationships'])
        return relationships
    
    def _validate_generation(self, response: str, results: Dict[str, Any]):
        """Validate generation consistency."""
        generation = results['metadata']['generation']
        if generation:
            expected_gen = f"gen{generation}"
            if expected_gen not in response.lower():
                results['issues'].append({
                    'type': 'generation_mismatch',
                    'message': f"Response should specify {expected_gen}"
                })
                results['confidence'] *= 0.9
    
    def _validate_elements(self, response: str, results: Dict[str, Any]):
        """Validate element consistency."""
        elements = results['metadata']['elements']
        response_lower = response.lower()
        
        for element in elements:
            if element.lower() not in response_lower:
                results['issues'].append({
                    'type': 'element_missing',
                    'message': f"Response should mention {element} element"
                })
                results['confidence'] *= 0.9
    
    def _validate_relationships(self, response: str, results: Dict[str, Any]):
        """Validate relationship consistency."""
        relationships = results['metadata']['relationships']
        response_lower = response.lower()
        
        for rel in relationships:
            source = rel.get('source_name', '').lower()
            target = rel.get('target_name', '').lower()
            rel_type = rel.get('type', '').lower()
            
            if source in response_lower and target not in response_lower:
                results['issues'].append({
                    'type': 'relationship_incomplete',
                    'message': f"Response mentions {source} but not related {target} ({rel_type})"
                })
                results['confidence'] *= 0.95
    
    def _validate_facts(self, response: str, context: List[Dict[str, Any]], results: Dict[str, Any]):
        """Validate factual consistency."""
        # Extract entity mentions
        entity_mentions = self._extract_entity_mentions(response)
        
        # Check each mentioned entity against knowledge graph
        for mention in entity_mentions:
            graph_entity = self.knowledge_graph.get_entity_by_id(mention['id'])
            if graph_entity:
                # Check attribute consistency
                for attr, value in mention['attributes'].items():
                    if attr in graph_entity['attributes'] and graph_entity['attributes'][attr] != value:
                        results['is_valid'] = False
                        results['issues'].append({
                            'type': 'attribute_mismatch',
                            'message': f"Incorrect {attr} for {mention['name']}: stated {value}, actual {graph_entity['attributes'][attr]}"
                        })
                        results['confidence'] *= 0.8
    
    def _suggest_enhancements(self, response: str, context: List[Dict[str, Any]], results: Dict[str, Any]):
        """Suggest potential response enhancements."""
        # Check for unused relevant context
        unused_context = self._find_unused_context(response, context)
        if unused_context:
            results['enhancements'].append({
                'type': 'additional_context',
                'suggestion': 'Consider including information about: ' + 
                            ', '.join(c['entity']['name'] for c in unused_context)
            })
            
        # Check for relationship opportunities
        relationship_suggestions = self._suggest_relationships(response, context)
        results['enhancements'].extend(relationship_suggestions)
    
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
    
    def _find_unused_context(self, response: str, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    
    def _suggest_relationships(self, response: str, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
                    'suggestion': f"Consider mentioning related entities for {mention['name']}: " +
                                ', '.join(f"{r['entity']} ({r['relationship']})" for r in unused_relations)
                })
        
        return suggestions
    
    def _check_source_coverage(self, response: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check how well the response covers the provided context."""
        return {
            'context_used': len(context),
            'coverage_score': self._calculate_coverage_score(response, context)
        }
    
    def _calculate_coverage_score(self, response: str, context: List[Dict[str, Any]]) -> float:
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
    
    def validate_response(self, response: dict, context: list) -> dict:
        """Robust validation with error suppression"""
        try:
            safe_context = context or []
            issues = []
            
            # Check for empty responses
            if not response.get('response'):
                issues.append("Empty response generated")
            
            # Check context usage
            used_context = self._find_used_context(response['response'], safe_context)
            if not used_context:
                issues.append("Response not grounded in provided context")
            
            return {
                "is_valid": len(issues) == 0,
                "issues": issues,
                "confidence": 1.0 - (len(issues) * 0.3)
            }
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return {
                "is_valid": False,
                "issues": ["Validation system error"],
                "confidence": 0.0
            } 