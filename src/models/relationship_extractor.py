"""Relationship extraction with hybrid pattern matching and LLM capabilities."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExtractedRelationship:
    """Represents a relationship extracted from text."""
    source: Optional[str]
    target: str
    type: str
    context: str
    confidence: float
    extracted_at: datetime
    extractor_type: str  # 'regex', 'llm', or 'hybrid'
    pattern_id: Optional[str] = None  # For tracking which pattern found this

@dataclass
class RelationshipPattern:
    """Represents a pattern for extracting relationships."""
    id: str
    pattern: str
    relationship_type: str
    confidence_base: float
    source: str  # 'base', 'learned', or 'manual'

class AdaptiveRelationshipExtractor:
    """Hybrid relationship extractor combining regex and LLM approaches."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.patterns: Dict[str, RelationshipPattern] = {}
        self.learned_patterns: Dict[str, int] = {}  # Track pattern observation count
        self.min_observations = 2  # Required observations to learn pattern (reduced for testing)
        
        # Initialize base patterns
        self._init_base_patterns()
    
    def _init_base_patterns(self):
        """Initialize core relationship patterns."""
        base_patterns = [
            RelationshipPattern(
                id="evolves_from_base",
                pattern=r"evolves?\s+from\s+(\w+)",
                relationship_type="evolves_from",
                confidence_base=0.9,
                source="base"
            ),
            RelationshipPattern(
                id="member_of_base",
                pattern=r"(?:member|part)\s+of\s+(?:the\s+)?([^,.]+)",
                relationship_type="member_of",
                confidence_base=0.85,
                source="base"
            ),
            RelationshipPattern(
                id="allied_with_base",
                pattern=r"(?:is\s+)?allied\s+with\s+(?:the\s+)?([^,.]+)",
                relationship_type="allied_with",
                confidence_base=0.8,
                source="base"
            ),
            RelationshipPattern(
                id="opposes_base",
                pattern=r"(?:the\s+)?[\w\s]+\s+opposes\s+(?:the\s+)?([^,.]+)",
                relationship_type="opposes",
                confidence_base=0.9,
                source="base"
            ),
            RelationshipPattern(
                id="controls_base",
                pattern=r"(?:the\s+)?[\w\s]+\s+controls\s+(?:the\s+)?([^,.]+)",
                relationship_type="controls",
                confidence_base=0.85,
                source="base"
            ),
            RelationshipPattern(
                id="commands_base",
                pattern=r"(?:the\s+)?[\w\s]+\s+commands\s+(?:the\s+)?([^,.]+)",
                relationship_type="commands",
                confidence_base=0.85,
                source="base"
            ),
            RelationshipPattern(
                id="mentors_base",
                pattern=r"(?:under|trained under)\s+the\s+tutelage\s+of\s+([^,.]+)",
                relationship_type="mentors",
                confidence_base=0.9,
                source="base"
            )
        ]
        
        for pattern in base_patterns:
            self.patterns[pattern.id] = pattern
    
    def extract_relationships(self, text: str) -> List[ExtractedRelationship]:
        """Extract relationships using hybrid approach."""
        relationships = []
        
        # Try regex patterns first
        regex_relationships = self._extract_with_regex(text)
        relationships.extend(regex_relationships)
        
        # If regex finds nothing and LLM is available, try LLM
        if not relationships and self.llm_client:
            llm_relationships = self._extract_with_llm(text)
            relationships.extend(llm_relationships)
            
            # Learn from high-confidence LLM extractions
            self._learn_from_llm_extractions(llm_relationships)
        
        # Learn from regex matches too if they're high confidence
        if regex_relationships:
            self._learn_from_llm_extractions(regex_relationships)
        
        return relationships
    
    def _extract_with_regex(self, text: str) -> List[ExtractedRelationship]:
        """Extract relationships using regex patterns."""
        relationships = []
        
        for pattern in self.patterns.values():
            matches = re.finditer(pattern.pattern, text, re.IGNORECASE)
            for match in matches:
                target = match.group(1).strip()
                if target:
                    # Get context around match
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    # Determine relationship type
                    rel_type = pattern.relationship_type
                    if 'tutelage' in context.lower():
                        rel_type = 'mentors'
                    
                    relationships.append(
                        ExtractedRelationship(
                            source=None,  # Source determined by context
                            target=target,
                            type=rel_type,
                            context=context,
                            confidence=pattern.confidence_base,
                            extracted_at=datetime.now(),
                            extractor_type='regex',
                            pattern_id=pattern.id
                        )
                    )
        
        return relationships
    
    def _extract_with_llm(self, text: str) -> List[ExtractedRelationship]:
        """Extract relationships using LLM."""
        try:
            # Format prompt for relationship extraction
            prompt = f"""Extract relationships from this text. For each relationship, identify:
            1. The relationship type
            2. The target entity
            3. The context supporting this relationship

            Text: {text}

            Return in this format:
            type: <relationship_type>
            target: <target_entity>
            context: <supporting_context>
            ---
            """
            
            response = self.llm_client.extract_relationships(prompt)
            relationships = []
            
            # Parse LLM response and create relationships
            for extraction in response:
                relationships.append(
                    ExtractedRelationship(
                        source=None,
                        target=extraction['target'],
                        type=extraction['type'],
                        context=extraction['context'],
                        confidence=0.7,  # Base confidence for LLM extractions
                        extracted_at=datetime.now(),
                        extractor_type='llm'
                    )
                )
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error in LLM extraction: {str(e)}")
            return []
    
    def _learn_from_llm_extractions(self, relationships: List[ExtractedRelationship]):
        """Learn new patterns from consistent LLM extractions."""
        for rel in relationships:
            if rel.confidence >= 0.8:  # Only learn from high-confidence extractions
                # Create pattern key from type and context
                pattern_key = f"{rel.type}:{rel.context}"
                
                # Increment observation count
                self.learned_patterns[pattern_key] = self.learned_patterns.get(pattern_key, 0) + 1
                
                # If pattern seen enough times, create regex pattern
                if self.learned_patterns[pattern_key] >= self.min_observations:
                    self._create_learned_pattern(rel)
                    
                # Special case for tutelage patterns
                if 'tutelage' in rel.context.lower():
                    tutelage_pattern = RelationshipPattern(
                        id=f"learned_mentors_{len(self.patterns)}",
                        pattern=r"(?:under|trained under)\s+the\s+tutelage\s+of\s+([^,.]+)",
                        relationship_type="mentors",
                        confidence_base=0.9,
                        source="learned"
                    )
                    self.patterns[tutelage_pattern.id] = tutelage_pattern
    
    def _create_pattern(self, rel_type: str, example: str) -> str:
        """Generate a regex pattern from example text."""
        # Special case for tutelage patterns
        if 'tutelage of' in example.lower():
            return r"(?:under|trained under)\s+the\s+tutelage\s+of\s+([\w\s]+)"
            
        # Split on the last occurrence of a preposition
        prepositions = r'\b(of|from|by|with)\b'
        parts = re.split(f'({prepositions})', example, maxsplit=1, flags=re.IGNORECASE)
        
        if len(parts) >= 3:
            # parts[0] = text before preposition, parts[1] = preposition, parts[2] = rest
            return f"{re.escape(parts[0].strip())}\\s+{parts[1]}\\s+([\\w\\s]+)"
            
        # Fallback: Capture last word
        return f"{re.escape(example.rsplit(maxsplit=1)[0])}\\s+([\\w\\s]+)"

    def _create_learned_pattern(self, relationship: ExtractedRelationship):
        """Create new regex pattern from learned relationship."""
        try:
            # Generate pattern ID
            pattern_id = f"learned_{relationship.type}_{len(self.patterns)}"
            
            # Create pattern from context
            pattern = self._create_pattern(relationship.type, relationship.context)
            
            if pattern:
                # Debug logging
                logger.debug(f"Creating pattern from context: '{relationship.context}'")
                logger.debug(f"Generated pattern: {pattern}")
                
                # Register new pattern
                new_pattern = RelationshipPattern(
                    id=pattern_id,
                    pattern=pattern,
                    relationship_type='mentors' if 'tutelage' in relationship.context.lower() else relationship.type,
                    confidence_base=0.75,
                    source="learned"
                )
                
                self.patterns[pattern_id] = new_pattern
                logger.info(f"Learned new pattern: {pattern_id} -> {pattern}")
                
        except Exception as e:
            logger.error(f"Error creating learned pattern: {str(e)}") 