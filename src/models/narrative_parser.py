"""Module for parsing narrative text to extract relationships and entities."""

import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ExtractedRelationship:
    """Represents a relationship extracted from text."""
    source: Optional[str]
    target: str
    type: str
    context: str
    confidence: float

class NarrativeRelationshipExtractor:
    """Parse complex relationships from story text."""
    
    def __init__(self):
        self.patterns = {
            'faction_relations': [
                (r'allied with (the )?([\w\s]+)', 'allied_with'),
                (r'at war with (the )?([\w\s]+)', 'at_war_with'),
                (r'supports? (the )?([\w\s]+)', 'supports'),
                (r'opposes? (the )?([\w\s]+)', 'opposes')
            ],
            'character_roles': [
                (r'leader of (the )?([\w\s]+)', 'leads'),
                (r'member of (the )?([\w\s]+)', 'member_of'),
                (r'represents? (the )?([\w\s]+)', 'represents'),
                (r'serves? (the )?([\w\s]+)', 'serves')
            ],
            'location_relations': [
                (r'in (the )?([\w\s]+ region)', 'located_in'),
                (r'from (the )?([\w\s]+ kingdom)', 'origin_in'),
                (r'near (the )?([\w\s]+)', 'near'),
                (r'borders? (the )?([\w\s]+)', 'borders')
            ],
            'political_relations': [
                (r'controls? (the )?([\w\s]+)', 'controls'),
                (r'rules? (over )?(the )?([\w\s]+)', 'rules'),
                (r'governs? (the )?([\w\s]+)', 'governs'),
                (r'influenced by (the )?([\w\s]+)', 'influenced_by')
            ],
            'event_relations': [
                (r'participated in (the )?([\w\s]+)', 'participated_in'),
                (r'involved in (the )?([\w\s]+)', 'involved_in'),
                (r'during (the )?([\w\s]+)', 'during'),
                (r'after (the )?([\w\s]+)', 'after')
            ]
        }
        
        # Entity type patterns
        self.entity_patterns = {
            'faction': [
                r'(the )?([\w\s]+) (faction|group|army|force|council)',
                r'(the )?([\w\s]+) (resistance|alliance|coalition)'
            ],
            'location': [
                r'(the )?([\w\s]+) (region|kingdom|city|town|village)',
                r'(the )?([\w\s]+) (temple|shrine|sanctuary)'
            ],
            'character': [
                r'([\w\s]+) (the )?(leader|commander|chief|ruler)',
                r'([\w\s]+), (a|the|an) (warrior|mage|priest|guardian)'
            ]
        }
    
    def extract_from_text(self, text: str, source_entity: Optional[str] = None) -> List[ExtractedRelationship]:
        """Extract relationships from text with improved context awareness."""
        relationships = []
        
        # Process each relationship category
        for category, patterns in self.patterns.items():
            for pattern, rel_type in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Get surrounding context
                    start_pos = max(0, match.start() - 100)
                    end_pos = min(len(text), match.end() + 100)
                    context = text[start_pos:end_pos]
                    
                    # Extract target entity
                    target = match.group(2).strip() if match.groups() else ''
                    
                    # Calculate confidence based on pattern match and context
                    confidence = self._calculate_confidence(match, context)
                    
                    if target:
                        relationships.append(ExtractedRelationship(
                            source=source_entity,
                            target=target,
                            type=rel_type,
                            context=context,
                            confidence=confidence
                        ))
        
        return relationships
    
    def _calculate_confidence(self, match: re.Match, context: str) -> float:
        """Calculate confidence score for an extracted relationship."""
        base_confidence = 0.7  # Base confidence for regex match
        
        # Adjust based on match position
        pos_factor = 1.0 - (match.start() / len(context) * 0.3)
        
        # Adjust based on context clarity
        clarity_bonus = 0.0
        if '.' in context:  # Clear sentence boundary
            clarity_bonus += 0.1
        if any(marker in context.lower() for marker in ['because', 'due to', 'as a result']):
            clarity_bonus += 0.1
            
        # Penalize for ambiguity
        ambiguity_penalty = 0.0
        if any(word in context.lower() for word in ['maybe', 'perhaps', 'possibly']):
            ambiguity_penalty += 0.1
        
        final_confidence = min(1.0, base_confidence * pos_factor + clarity_bonus - ambiguity_penalty)
        return round(final_confidence, 2)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract potential entities from text by type."""
        entities = {entity_type: [] for entity_type in self.entity_patterns.keys()}
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Extract entity name based on pattern groups
                    if len(match.groups()) >= 2:
                        entity_name = match.group(1) or match.group(2)
                        entity_name = entity_name.strip()
                        if entity_name and entity_name not in entities[entity_type]:
                            entities[entity_type].append(entity_name)
        
        return entities
    
    def analyze_text(self, text: str, source_entity: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive text analysis for relationships and entities."""
        # Extract relationships
        relationships = self.extract_from_text(text, source_entity)
        
        # Extract entities
        entities = self.extract_entities(text)
        
        # Group relationships by type
        grouped_relationships = {}
        for rel in relationships:
            if rel.type not in grouped_relationships:
                grouped_relationships[rel.type] = []
            grouped_relationships[rel.type].append({
                'source': rel.source,
                'target': rel.target,
                'context': rel.context,
                'confidence': rel.confidence
            })
        
        return {
            'relationships': grouped_relationships,
            'entities': entities,
            'stats': {
                'total_relationships': len(relationships),
                'total_entities': sum(len(ents) for ents in entities.values()),
                'relationship_types': list(grouped_relationships.keys()),
                'entity_types': list(entities.keys())
            }
        } 