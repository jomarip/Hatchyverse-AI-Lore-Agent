"""Enhanced response validation for the Hatchyverse chatbot."""

from typing import Dict, List, Any, Optional
import logging
from .knowledge_graph import HatchyKnowledgeGraph
import re

logger = logging.getLogger(__name__)

class ResponseValidator:
    """Validate and enhance chatbot responses."""
    
    def __init__(self, knowledge_graph: HatchyKnowledgeGraph):
        self.knowledge_graph = knowledge_graph
    
    def validate(self, response: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate response against provided context with improved error handling."""
        try:
            results = {
                'is_valid': True,
                'issues': [],
                'source_coverage': {},
                'fact_validation': {},
                'enhancements': []
            }
            
            # Skip validation if no context provided
            if not contexts:
                results['is_valid'] = True
                results['enhancements'].append({
                    'message': 'No context was provided to validate against',
                    'type': 'warning'
                })
                return results
            
            # Check context usage
            context_used = 0
            for ctx in contexts:
                content = ctx.get('content', '') or ctx.get('text_content', '')
                if content and self._content_used_in_response(content, response):
                    context_used += 1
            
            # Calculate coverage score
            coverage_score = context_used / len(contexts) if contexts else 0.0
            results['source_coverage'] = {
                'context_used': context_used,
                'coverage_score': coverage_score
            }
            
            # Check for factual consistency
            fact_results = self._validate_facts(response, contexts)
            results['fact_validation'] = fact_results
            if fact_results.get('issues'):
                results['issues'].extend(fact_results['issues'])
            
            # Check for element consistency
            element_results = self._validate_elements(response, contexts)
            if element_results.get('issues'):
                results['issues'].extend(element_results['issues'])
            
            # Check for relationship consistency
            relationship_results = self._validate_relationships(response, contexts)
            if relationship_results.get('issues'):
                results['issues'].extend(relationship_results['issues'])
            
            # Add enhancement suggestions
            if coverage_score < 0.5:
                results['enhancements'].append({
                    'message': 'Consider using more information from the provided context',
                    'type': 'suggestion'
                })
            
            # Mark response as invalid if there are issues
            if results['issues']:
                results['is_valid'] = False
            
            return results
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return {
                'is_valid': True,  # Default to valid on error
                'issues': [f"Validation error: {str(e)}"],
                'source_coverage': {},
                'fact_validation': {},
                'enhancements': []
            }
    
    def _content_used_in_response(self, content: str, response: str) -> bool:
        """Check if content is used in response."""
        # Extract key phrases from content
        content_lower = content.lower()
        response_lower = response.lower()
        
        # Split into sentences and check for key phrase usage
        content_sentences = content_lower.split('.')
        for sentence in content_sentences:
            words = sentence.strip().split()
            if len(words) >= 3:  # Only check substantial phrases
                phrase = ' '.join(words[:3])
                if phrase in response_lower:
                    return True
        
        return False
    
    def _validate_facts(self, response: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate factual consistency in response."""
        results = {
            'issues': [],
            'validated_facts': []
        }
        
        try:
            # Extract facts from response
            response_facts = self._extract_facts(response)
            
            # Extract facts from context
            context_facts = []
            for ctx in contexts:
                content = ctx.get('content', '') or ctx.get('text_content', '')
                if content:
                    context_facts.extend(self._extract_facts(content))
            
            # Compare facts
            for fact in response_facts:
                if not self._fact_supported_by_context(fact, context_facts):
                    results['issues'].append(f"Unsupported fact: {fact}")
                else:
                    results['validated_facts'].append(fact)
            
            return results
            
        except Exception as e:
            logger.error(f"Error validating facts: {str(e)}")
            return {'issues': [], 'validated_facts': []}
    
    def _validate_elements(self, response: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate element-related information in response."""
        results = {
            'issues': [],
            'validated_elements': []
        }
        
        try:
            # Extract element mentions from response
            response_elements = self._extract_elements(response)
            
            # Extract elements from context
            context_elements = set()
            for ctx in contexts:
                metadata = ctx.get('metadata', {})
                if 'element' in metadata:
                    context_elements.add(metadata['element'].lower())
            
            # Check each element mentioned in response
            for element in response_elements:
                if element.lower() not in context_elements:
                    results['issues'].append(f"Element '{element}' not found in context")
                else:
                    results['validated_elements'].append(element)
            
            return results
            
        except Exception as e:
            logger.error(f"Error validating elements: {str(e)}")
            return {'issues': [], 'validated_elements': []}
    
    def _validate_relationships(self, response: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate relationship information in response."""
        results = {
            'issues': [],
            'validated_relationships': []
        }
        
        try:
            # Extract relationships from response
            response_relationships = self._extract_relationships(response)
            
            # Extract relationships from context
            context_relationships = []
            for ctx in contexts:
                content = ctx.get('content', '') or ctx.get('text_content', '')
                if content:
                    context_relationships.extend(self._extract_relationships(content))
            
            # Compare relationships
            for rel in response_relationships:
                if not self._relationship_supported_by_context(rel, context_relationships):
                    results['issues'].append(f"Unsupported relationship: {rel}")
                else:
                    results['validated_relationships'].append(rel)
            
            return results
            
        except Exception as e:
            logger.error(f"Error validating relationships: {str(e)}")
            return {'issues': [], 'validated_relationships': []}
    
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

    def _extract_facts(self, text: str) -> List[Dict[str, Any]]:
        """Extract factual statements from text with improved accuracy."""
        facts = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Extract numerical facts
            numbers = re.findall(r'\d+', sentence)
            for num in numbers:
                facts.append({
                    'type': 'numerical',
                    'value': num,
                    'context': sentence
                })
            
            # Extract attribute statements
            attribute_patterns = [
                (r'is (?:a|an) ([\w\s]+)', 'attribute'),
                (r'has (?:a|an) ([\w\s]+)', 'property'),
                (r'can ([\w\s]+)', 'ability')
            ]
            
            for pattern, fact_type in attribute_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    facts.append({
                        'type': fact_type,
                        'value': match.group(1).strip(),
                        'context': sentence
                    })
            
            # Extract relationships
            relationship_patterns = [
                (r'([\w\s]+) is (?:in|at|near) ([\w\s]+)', 'location'),
                (r'([\w\s]+) belongs to ([\w\s]+)', 'membership'),
                (r'([\w\s]+) evolves? (?:into|from) ([\w\s]+)', 'evolution')
            ]
            
            for pattern, rel_type in relationship_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    facts.append({
                        'type': 'relationship',
                        'subtype': rel_type,
                        'source': match.group(1).strip(),
                        'target': match.group(2).strip(),
                        'context': sentence
                    })
        
        return facts

    def _fact_supported_by_context(self, fact: Dict[str, Any], context_facts: List[Dict[str, Any]]) -> bool:
        """Check if a fact is supported by context with improved matching."""
        # For numerical facts, check exact matches
        if fact['type'] == 'numerical':
            return any(
                cf['type'] == 'numerical' and cf['value'] == fact['value']
                for cf in context_facts
            )
        
        # For attribute and property facts, check semantic similarity
        if fact['type'] in ['attribute', 'property', 'ability']:
            return any(
                cf['type'] == fact['type'] and
                self._are_values_similar(cf['value'], fact['value'])
                for cf in context_facts
            )
        
        # For relationship facts, check both entities and relationship type
        if fact['type'] == 'relationship':
            return any(
                cf['type'] == 'relationship' and
                cf['subtype'] == fact['subtype'] and
                self._are_values_similar(cf['source'], fact['source']) and
                self._are_values_similar(cf['target'], fact['target'])
                for cf in context_facts
            )
        
        return False

    def _are_values_similar(self, val1: str, val2: str) -> bool:
        """Check if two values are semantically similar."""
        # Convert to lowercase and remove extra whitespace
        val1 = ' '.join(val1.lower().split())
        val2 = ' '.join(val2.lower().split())
        
        # Check exact match
        if val1 == val2:
            return True
        
        # Check if one is contained in the other
        if val1 in val2 or val2 in val1:
            return True
        
        # Check word overlap
        words1 = set(val1.split())
        words2 = set(val2.split())
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        return overlap / total > 0.5  # More than 50% word overlap

    def _extract_elements(self, text: str) -> List[str]:
        """Extract element mentions from text with improved accuracy."""
        elements = []
        
        # Define element patterns with variations
        element_patterns = {
            'fire': [r'fire(?:\s+type|\s+element)?', r'flames?'],
            'water': [r'water(?:\s+type|\s+element)?', r'aqua'],
            'plant': [r'plant(?:\s+type|\s+element)?', r'grass', r'nature'],
            'dark': [r'dark(?:\s+type|\s+element)?', r'shadow'],
            'light': [r'light(?:\s+type|\s+element)?', r'holy'],
            'void': [r'void(?:\s+type|\s+element)?', r'null']
        }
        
        text_lower = text.lower()
        
        for element, patterns in element_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    elements.append(element)
                    break  # Found one pattern for this element, move to next
        
        return list(set(elements))  # Remove duplicates

    def _extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships from text with improved accuracy."""
        relationships = []
        
        # Define relationship patterns
        patterns = [
            (r'([\w\s]+) evolves? (?:into|from) ([\w\s]+)', 'evolution'),
            (r'([\w\s]+) lives? (?:in|at) ([\w\s]+)', 'habitat'),
            (r'([\w\s]+) belongs? to (?:the\s+)?([\w\s]+)', 'faction'),
            (r'([\w\s]+) is (?:allied|friends) with (?:the\s+)?([\w\s]+)', 'alliance'),
            (r'([\w\s]+) (?:leads?|commands?) (?:the\s+)?([\w\s]+)', 'leadership')
        ]
        
        for pattern, rel_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                relationships.append({
                    'type': rel_type,
                    'source': match.group(1).strip(),
                    'target': match.group(2).strip(),
                    'context': text[max(0, match.start() - 50):min(len(text), match.end() + 50)]
                })
        
        return relationships 