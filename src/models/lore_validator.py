from typing import List, Dict, Any, Optional
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .lore_entity import LoreEntity
import logging
from collections import defaultdict
from itertools import chain
import re
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoreValidator:
    """Handles validation of new lore submissions against existing canon."""
    
    def __init__(self, embeddings: Embeddings, threshold: float = 0.92):
        self.vector_store = None
        self.threshold = threshold
        self.embeddings = embeddings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Define valid element-ability combinations
        self.valid_combinations = {
            "Water": ["ice", "frost", "crystal", "aqua", "water", "snow", "freeze"],
            "Fire": ["flame", "heat", "magma", "blaze", "fire", "burn", "ember"],
            "Dark": ["shadow", "night", "void", "dark", "shade", "gloom"],
            "Electric": ["lightning", "thunder", "spark", "electric", "volt", "shock"],
            "Nature": ["plant", "leaf", "forest", "vine", "nature", "grass", "bloom"]
        }
        
        # Define invalid element combinations
        self.invalid_locations = {
            "Water": ["volcano", "lava", "magma"],
            "Fire": ["ocean", "lake", "river"],
            "Dark": ["sun", "light", "holy"],
            "Electric": ["ground", "earth"],
            "Nature": ["void", "abyss"]
        }
        
        # Add story validation parameters
        self.story_arc_elements = {
            "exposition": ["introduction", "setting", "background", "setup"],
            "rising_action": ["conflict", "challenge", "journey", "quest"],
            "climax": ["confrontation", "battle", "revelation", "turning point"],
            "falling_action": ["resolution", "aftermath", "consequence"],
            "conclusion": ["ending", "closure", "lesson", "moral"]
        }
        
        self.narrative_elements = {
            "character": ["protagonist", "antagonist", "ally", "mentor"],
            "setting": ["location", "time period", "world", "environment"],
            "plot": ["goal", "conflict", "obstacle", "resolution"],
            "theme": ["message", "moral", "lesson", "meaning"]
        }
        
    def _is_element_ability_valid(self, text: str) -> tuple[bool, str]:
        """Validate element-ability relationships in text"""
        text_lower = text.lower()
        
        # Check for NEW submissions separately
        if "NEW:" in text:
            submission_part = text.split("NEW:")[-1]
            return self._validate_single_submission(submission_part)
        
        return self._validate_combined_context(text_lower)

    def _validate_single_submission(self, text: str) -> tuple[bool, str]:
        """Validate a single submission without combined context"""
        text_lower = text.lower()
        
        # 1. Check invalid locations
        for element, invalid_locs in self.invalid_locations.items():
            if element.lower() in text_lower:
                for loc in invalid_locs:
                    if loc in text_lower:
                        return False, f"{element} type cannot be in {loc}"
        
        # 2. Check ability-element compatibility                
        found_elements = set()
        for element in self.valid_combinations:
            if element.lower() in text_lower:
                found_elements.add(element)
        
        found_abilities = set()
        for ability in chain(*self.valid_combinations.values()):
            if ability in text_lower:
                found_abilities.add(ability)
        
        # Validate abilities against detected elements
        for element in found_elements:
            valid_abilities = set(self.valid_combinations[element])
            invalid = found_abilities - valid_abilities
            
            if invalid:
                return False, f"{element} cannot have: {', '.join(invalid)}"
        
        return True, "Valid submission"

    def _validate_combined_context(self, text_lower: str) -> tuple[bool, str]:
        """Validate combined context"""
        # 1. Check for invalid location-element combinations
        for element, invalid_locs in self.invalid_locations.items():
            element_lower = element.lower()
            if element_lower in text_lower:
                for loc in invalid_locs:
                    if loc in text_lower:
                        return False, f"{element} type cannot be in {loc}"

        # 2. Validate ability-element relationships
        element_abilities = defaultdict(set)
        found_elements = set()
        
        # Map elements to their valid abilities
        for element, abilities in self.valid_combinations.items():
            if element.lower() in text_lower:
                found_elements.add(element)
                element_abilities[element] = set(abilities)

        # 3. Check for cross-element ability conflicts
        found_abilities = set()
        for ability in chain.from_iterable(self.valid_combinations.values()):
            if ability in text_lower:
                found_abilities.add(ability)

        # If multiple elements found, check ability compatibility
        if len(found_elements) > 1:
            # Check if abilities belong to multiple elements
            conflicting_abilities = set()
            for ability in found_abilities:
                valid_in = [e for e in found_elements 
                           if ability in self.valid_combinations[e]]
                if len(valid_in) == 0:
                    conflicting_abilities.add(ability)
            
            if conflicting_abilities:
                return False, f"Abilities conflict between elements: {', '.join(conflicting_abilities)}"

        # Validate abilities against their primary element
        for element in found_elements:
            valid_for_element = element_abilities[element]
            invalid_abilities = found_abilities - valid_for_element
            
            if invalid_abilities:
                return False, f"Invalid abilities for {element}: {', '.join(invalid_abilities)}"

        return True, "Valid combination"

    def build_knowledge_base(self, entities: List[LoreEntity]):
        """Build or update the knowledge base with enhanced context awareness."""
        try:
            texts = []
            metadatas = []
            
            for entity in entities:
                # Core entity information
                base_text = (
                    f"Name: {entity.name}\n"
                    f"Type: {entity.entity_type}\n"
                    f"Element: {entity.element}\n"
                    f"Description: {entity.description}\n"
                )
                
                # Add source fragments with context
                for fragment in entity.source_fragments:
                    context_text = (
                        f"Context ({fragment['context_type']}): {fragment['text']}\n"
                        f"Source: {fragment['source']}\n"
                    )
                    texts.append(context_text)
                    metadatas.append({
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.entity_type,
                        "context_type": fragment['context_type'],
                        "source": fragment['source'],
                        "context_tags": entity.context_tags
                    })
                
                # Add references with context
                for ref in entity.references:
                    ref_text = (
                        f"Related {ref.relationship_type}: {ref.entity_id}\n"
                        f"Context: {ref.context}\n"
                        f"Source: {ref.source_text}\n"
                    )
                    texts.append(ref_text)
                    metadatas.append({
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.entity_type,
                        "context_type": ref.relationship_type,
                        "related_entity": ref.entity_id,
                        "context_tags": entity.context_tags
                    })
                
                # Add base entity information
                texts.append(base_text)
                metadatas.append({
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.entity_type,
                    "element": entity.element,
                    "context_tags": entity.context_tags
                })
            
            # Create or update vector store with enhanced metadata
            if self.vector_store is None:
                self.vector_store = FAISS.from_texts(
                    texts,
                    self.embeddings,
                    metadatas=metadatas
                )
            else:
                self.vector_store.add_texts(texts, metadatas=metadatas)
                
        except Exception as e:
            logger.error(f"Error building knowledge base: {str(e)}", exc_info=True)
            raise
            
    def _is_list_query(self, query: str) -> bool:
        """Detect if a query is asking for a list of items."""
        list_indicators = [
            'list', 'all', 'what are', 'what is', 'show me', 'tell me',
            'what were', 'which', 'enumerate', 'name'
        ]
        return any(indicator in query.lower() for indicator in list_indicators)

    def get_entities_by_type(self, entity_type: str, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all entities of a specific type from a specific source."""
        if not self.vector_store:
            raise ValueError("Knowledge base not initialized")
            
        # Build filter conditions
        filter_conditions = [lambda meta: meta.get('type') == entity_type]
        if source:
            filter_conditions.append(lambda meta: source in meta.get('sources', []))
            
        search_filter = lambda meta: all(f(meta) for f in filter_conditions)
        
        # Get all matching documents
        results = self.vector_store.similarity_search_with_score(
            f"type:{entity_type}",  # Simple query to match type
            k=1000,  # Large k to get all matches
            filter=search_filter
        )
        
        # Process and deduplicate results
        unique_entities = {}
        for doc, score in results:
            entity_id = doc.metadata.get('id')
            if entity_id and entity_id not in unique_entities:
                unique_entities[entity_id] = {
                    'id': entity_id,
                    'name': doc.metadata.get('name'),
                    'type': doc.metadata.get('type'),
                    'element': doc.metadata.get('element'),
                    'description': doc.page_content,
                    'metadata': doc.metadata
                }
                
        return list(unique_entities.values())

    def search_knowledge_base(self, query: str, context_type: Optional[str] = None, 
                            entity_type: Optional[str] = None, k: int = 5) -> List[Dict[str, Any]]:
        """Enhanced search with better handling of list queries."""
        if not self.vector_store:
            raise ValueError("Knowledge base not initialized")
            
        # Check if this is a list query
        if self._is_list_query(query):
            # For list queries, we want to return all matching entities
            if entity_type:
                return self.get_entities_by_type(entity_type)
            else:
                # Try to infer entity type from query
                for type_indicator in ['monster', 'hatchy', 'item', 'location']:
                    if type_indicator in query.lower():
                        return self.get_entities_by_type(type_indicator.title())
                        
        # Regular search logic for non-list queries
        search_filter = None
        if context_type or entity_type:
            filter_conditions = []
            if context_type:
                filter_conditions.append(lambda meta: meta.get('context_type') == context_type)
            if entity_type:
                filter_conditions.append(lambda meta: meta.get('type') == entity_type)
                
            search_filter = lambda meta: all(f(meta) for f in filter_conditions)
            
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k,
            filter=search_filter
        )
        
        processed_results = []
        for doc, score in results:
            processed_results.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': score,
                'context_tags': doc.metadata.get('context_tags', [])
            })
            
        return processed_results

    def _create_conflict_entry(self, doc: Any, score: float, reason: str) -> Dict[str, Any]:
        """Creates an entry for a conflicting document.
        
        Args:
            doc: The document that conflicts
            score: The similarity score
            reason: The reason for the conflict
            
        Returns:
            Dict containing conflict information
        """
        return {
            "content": doc.page_content,
            "score": score,
            "reason": reason,
            "metadata": doc.metadata
        }

    def _create_similar_entry(self, doc: Any, score: float) -> Dict[str, Any]:
        """Creates an entry for a similar but non-conflicting document.
        
        Args:
            doc: The similar document
            score: The similarity score
            
        Returns:
            Dict containing similarity information
        """
        return {
            "content": doc.page_content,
            "score": score,
            "metadata": doc.metadata
        }

    def _create_validation_result(self, reason: str) -> Dict[str, Any]:
        """Creates a validation result when initial validation fails.
        
        Args:
            reason: The reason for validation failure
            
        Returns:
            Dict containing validation results with conflicts
        """
        return {
            "conflicts": [{
                "reason": reason,
                "score": 1.0,  # Maximum conflict score
                "content": None,
                "metadata": {
                    "name": "Validation",
                    "type": "System",
                    "element": None
                }
            }],
            "similar_concepts": [],
            "validation_score": 1.0
        }

    def check_conflict(self, user_input: str, k: int = 5) -> Dict[str, Any]:
        """
        Checks a new lore submission for conflicts with existing canon.
        
        Args:
            user_input: The new lore submission to check
            k: Number of similar entries to retrieve
            
        Returns:
            Dict containing:
            - conflicts: List of entries that conflict (high similarity)
            - similar_concepts: List of related but non-conflicting entries
            - validation_score: Highest similarity score found
        """
        try:
            if self.vector_store is None:
                raise ValueError("Knowledge base not initialized. Call build_knowledge_base first.")
            
            # First validate the submission independently
            is_valid, reason = self._is_element_ability_valid(user_input)
            if not is_valid:
                return self._create_validation_result(reason)
            
            # Then check against existing lore
            similar_items = self.vector_store.similarity_search_with_score(user_input, k=k)
            
            conflicts = []
            similar_concepts = []
            max_score = 0
            
            for doc, score in similar_items:
                logger.debug(f"Checking similarity score: {score:.2f}")
                
                # Only check combined context if score is above threshold
                if score > self.threshold:
                    combined_text = f"EXISTING: {doc.page_content}\nNEW: {user_input}"
                    combined_valid, combined_reason = self._is_element_ability_valid(combined_text)
                    
                    if not combined_valid:
                        conflicts.append(self._create_conflict_entry(doc, score, combined_reason))
                
                # Always track max score for reporting
                max_score = max(max_score, score)
                
                # Add to similar concepts if below threshold
                if score <= self.threshold:
                    similar_concepts.append(self._create_similar_entry(doc, score))

            return {
                "conflicts": conflicts,
                "similar_concepts": similar_concepts,
                "validation_score": max_score
            }
            
        except Exception as e:
            logger.error(f"Error checking conflicts: {str(e)}")
            raise

    def optimize_threshold(self, test_cases: List[Dict[str, Any]]) -> float:
        """
        Optimizes the conflict detection threshold using test cases.
        
        Args:
            test_cases: List of dicts with keys:
                - input: str (test input)
                - should_conflict: bool (whether it should be flagged)
                
        Returns:
            Optimal threshold value
        """
        try:
            best_threshold = 0.7
            best_score = 0
            
            for threshold in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
                correct = 0
                self.threshold = threshold
                
                for case in test_cases:
                    result = self.check_conflict(case["input"])
                    has_conflicts = len(result["conflicts"]) > 0
                    
                    if has_conflicts == case["should_conflict"]:
                        correct += 1
                
                accuracy = correct / len(test_cases)
                if accuracy > best_score:
                    best_score = accuracy
                    best_threshold = threshold
                    
                logger.info(f"Threshold {threshold}: Accuracy {accuracy}")
            
            self.threshold = best_threshold
            logger.info(f"Optimal threshold found: {best_threshold} (Accuracy: {best_score})")
            return best_threshold
            
        except Exception as e:
            logger.error(f"Error optimizing threshold: {str(e)}")
            raise

    def analyze_narrative_structure(self, story_text: str) -> Dict[str, Any]:
        """
        Analyzes the narrative structure of a story submission.
        
        Args:
            story_text: The story text to analyze
            
        Returns:
            Dict containing:
            - story_elements: Identified narrative elements
            - arc_analysis: Story arc components found
            - temporal_markers: Time-related references
            - character_relationships: Character interaction patterns
        """
        story_elements = {
            "characters": self._extract_characters(story_text),
            "settings": self._extract_settings(story_text),
            "plot_points": self._extract_plot_points(story_text),
            "themes": self._extract_themes(story_text)
        }
        
        arc_analysis = self._analyze_story_arc(story_text)
        temporal_markers = self._extract_temporal_markers(story_text)
        character_relationships = self._analyze_character_relationships(story_text)
        
        return {
            "story_elements": story_elements,
            "arc_analysis": arc_analysis,
            "temporal_markers": temporal_markers,
            "character_relationships": character_relationships,
            "validation_summary": self._validate_narrative_coherence(
                story_elements, arc_analysis, temporal_markers, character_relationships
            )
        }

    def _extract_characters(self, text: str) -> List[Dict[str, Any]]:
        """Extract and categorize characters from the story."""
        characters = []
        
        # Look for character introductions and descriptions
        character_pattern = r'([A-Z][a-zA-Z\s]+)(?:\s+(?:is|was|appeared|stood|walked|said))'
        matches = re.finditer(character_pattern, text)
        
        for match in matches:
            name = match.group(1).strip()
            context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
            
            # Analyze character role
            role = self._determine_character_role(name, context)
            
            characters.append({
                "name": name,
                "role": role,
                "introduction_context": context,
                "mentions": len(re.findall(re.escape(name), text))
            })
        
        return characters

    def _determine_character_role(self, name: str, context: str) -> str:
        """Determine the role of a character based on context."""
        context_lower = context.lower()
        
        role_indicators = {
            "protagonist": ["hero", "main character", "chosen", "journey", "quest"],
            "antagonist": ["villain", "enemy", "opposed", "dark", "evil"],
            "mentor": ["guide", "teacher", "wise", "elder", "master"],
            "ally": ["friend", "companion", "helper", "assist"]
        }
        
        for role, indicators in role_indicators.items():
            if any(indicator in context_lower for indicator in indicators):
                return role
                
        return "supporting"

    def _extract_settings(self, text: str) -> List[Dict[str, Any]]:
        """Extract and analyze story settings."""
        settings = []
        
        # Look for location and time markers
        location_pattern = r'(?:in|at|near|through)\s+(?:the\s+)?([A-Z][a-zA-Z\s]+)'
        time_pattern = r'(?:during|in|at)\s+(?:the\s+)?([A-Za-z\s]+(?:time|era|age|period))'
        
        # Process locations
        for match in re.finditer(location_pattern, text):
            location = match.group(1).strip()
            context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
            
            settings.append({
                "type": "location",
                "name": location,
                "context": context,
                "attributes": self._analyze_setting_attributes(location, context)
            })
            
        # Process time periods
        for match in re.finditer(time_pattern, text):
            time_period = match.group(1).strip()
            context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
            
            settings.append({
                "type": "time_period",
                "name": time_period,
                "context": context,
                "attributes": self._analyze_setting_attributes(time_period, context)
            })
            
        return settings

    def _analyze_setting_attributes(self, setting: str, context: str) -> Dict[str, List[str]]:
        """Analyze attributes of a story setting."""
        context_lower = context.lower()
        
        attributes = {
            "atmosphere": [],
            "significance": [],
            "elements": []
        }
        
        # Analyze atmosphere
        atmosphere_indicators = {
            "peaceful": ["calm", "quiet", "serene", "tranquil"],
            "dangerous": ["dark", "threatening", "ominous", "perilous"],
            "magical": ["mystical", "enchanted", "mysterious", "magical"],
            "ancient": ["old", "ancient", "historic", "primordial"]
        }
        
        for mood, indicators in atmosphere_indicators.items():
            if any(indicator in context_lower for indicator in indicators):
                attributes["atmosphere"].append(mood)
        
        # Check for elemental associations
        for element, keywords in self.valid_combinations.items():
            if any(keyword in context_lower for keyword in keywords):
                attributes["elements"].append(element)
        
        # Analyze significance
        if any(word in context_lower for word in ["important", "crucial", "key", "vital"]):
            attributes["significance"].append("plot_critical")
        if any(word in context_lower for word in ["sacred", "holy", "cursed", "blessed"]):
            attributes["significance"].append("mystical")
        if any(word in context_lower for word in ["ancient", "historic", "legendary"]):
            attributes["significance"].append("historical")
            
        return attributes

    def _extract_plot_points(self, text: str) -> List[Dict[str, Any]]:
        """Extract and analyze major plot points."""
        plot_points = []
        
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        
        for i, paragraph in enumerate(paragraphs):
            # Look for significant events
            event_indicators = [
                "suddenly", "finally", "eventually", "then",
                "but", "however", "although", "despite"
            ]
            
            if any(indicator in paragraph.lower() for indicator in event_indicators):
                # Get surrounding context
                context_start = max(0, i - 1)
                context_end = min(len(paragraphs), i + 2)
                context = '\n\n'.join(paragraphs[context_start:context_end])
                
                plot_points.append({
                    "content": paragraph,
                    "context": context,
                    "type": self._classify_plot_point(paragraph),
                    "significance": self._analyze_plot_significance(paragraph)
                })
        
        return plot_points

    def _classify_plot_point(self, text: str) -> str:
        """Classify the type of plot point."""
        text_lower = text.lower()
        
        for arc_type, indicators in self.story_arc_elements.items():
            if any(indicator in text_lower for indicator in indicators):
                return arc_type
                
        return "development"

    def _analyze_plot_significance(self, text: str) -> List[str]:
        """Analyze the significance of a plot point."""
        significance = []
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["crucial", "important", "key", "vital"]):
            significance.append("major")
        if any(word in text_lower for word in ["reveal", "discover", "learn", "realize"]):
            significance.append("revelation")
        if any(word in text_lower for word in ["fight", "battle", "confront", "face"]):
            significance.append("conflict")
        if any(word in text_lower for word in ["change", "transform", "evolve", "become"]):
            significance.append("transformation")
            
        return significance

    def _extract_themes(self, text: str) -> List[Dict[str, Any]]:
        """Extract and analyze story themes."""
        themes = []
        
        # Common theme patterns
        theme_patterns = {
            "growth": ["learn", "grow", "change", "develop", "become"],
            "conflict": ["struggle", "fight", "overcome", "defeat", "conquer"],
            "friendship": ["friend", "ally", "together", "help", "support"],
            "power": ["strength", "ability", "power", "control", "master"],
            "destiny": ["fate", "destiny", "chosen", "meant to", "prophecy"]
        }
        
        for theme, keywords in theme_patterns.items():
            mentions = []
            for keyword in keywords:
                for match in re.finditer(r'\b' + keyword + r'\b', text.lower()):
                    context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                    mentions.append({
                        "keyword": keyword,
                        "context": context
                    })
            
            if mentions:
                themes.append({
                    "name": theme,
                    "mentions": mentions,
                    "frequency": len(mentions)
                })
        
        return themes

    def _analyze_story_arc(self, text: str) -> Dict[str, Any]:
        """Analyze the story's narrative arc structure."""
        # Split text into sections
        sections = self._split_into_sections(text)
        
        arc_analysis = {
            "structure": [],
            "completeness": True,
            "missing_elements": []
        }
        
        # Analyze each section for arc elements
        for section in sections:
            section_type = self._identify_arc_section(section)
            if section_type:
                arc_analysis["structure"].append({
                    "type": section_type,
                    "content": section[:200] + "..." if len(section) > 200 else section
                })
        
        # Check for missing essential elements
        essential_elements = ["exposition", "rising_action", "climax", "conclusion"]
        found_elements = [item["type"] for item in arc_analysis["structure"]]
        
        for element in essential_elements:
            if element not in found_elements:
                arc_analysis["completeness"] = False
                arc_analysis["missing_elements"].append(element)
        
        return arc_analysis

    def _identify_arc_section(self, text: str) -> Optional[str]:
        """Identify the narrative arc section type."""
        text_lower = text.lower()
        
        for arc_type, indicators in self.story_arc_elements.items():
            if any(indicator in text_lower for indicator in indicators):
                return arc_type
                
        return None

    def _extract_temporal_markers(self, text: str) -> List[Dict[str, Any]]:
        """Extract and analyze temporal relationships in the story."""
        markers = []
        
        # Look for time-related phrases
        time_patterns = [
            r'(?:before|after|during|while|when)\s+([^,.]+)',
            r'(?:in the|at the)\s+([^,.]+(?:beginning|middle|end))',
            r'([A-Za-z]+\s+(?:days|weeks|months|years)(?:\s+(?:ago|later)))',
            r'(?:meanwhile|simultaneously|at the same time)'
        ]
        
        for pattern in time_patterns:
            for match in re.finditer(pattern, text):
                context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                markers.append({
                    "marker": match.group(0),
                    "context": context,
                    "type": self._classify_temporal_marker(match.group(0)),
                    "relative_position": self._get_relative_position(match.start(), len(text))
                })
        
        return markers

    def _classify_temporal_marker(self, marker: str) -> str:
        """Classify the type of temporal marker."""
        marker_lower = marker.lower()
        
        if any(word in marker_lower for word in ["before", "ago", "previously"]):
            return "past"
        if any(word in marker_lower for word in ["after", "later", "then"]):
            return "future"
        if any(word in marker_lower for word in ["during", "while", "meanwhile"]):
            return "simultaneous"
        if any(word in marker_lower for word in ["beginning", "start"]):
            return "start"
        if any(word in marker_lower for word in ["end", "finally"]):
            return "end"
            
        return "present"

    def _get_relative_position(self, position: int, total_length: int) -> str:
        """Get the relative position in the story (beginning, middle, end)."""
        relative_pos = position / total_length
        
        if relative_pos < 0.33:
            return "beginning"
        elif relative_pos < 0.66:
            return "middle"
        else:
            return "end"

    def _analyze_character_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Analyze relationships between characters."""
        relationships = []
        
        # Extract character names first
        characters = self._extract_characters(text)
        character_names = [char["name"] for char in characters]
        
        # Look for interactions between characters
        for i, char1 in enumerate(character_names):
            for char2 in character_names[i+1:]:
                # Find contexts where both characters appear
                interaction_pattern = f"({char1}.*?{char2}|{char2}.*?{char1})"
                for match in re.finditer(interaction_pattern, text, re.DOTALL):
                    context = match.group(0)
                    if len(context) > 200:  # Trim long contexts
                        context = context[:200] + "..."
                    
                    relationships.append({
                        "characters": [char1, char2],
                        "interaction_type": self._classify_interaction(context),
                        "context": context
                    })
        
        return relationships

    def _classify_interaction(self, context: str) -> str:
        """Classify the type of character interaction."""
        context_lower = context.lower()
        
        interaction_types = {
            "alliance": ["help", "support", "together", "friend", "ally"],
            "conflict": ["fight", "against", "oppose", "enemy", "rival"],
            "mentorship": ["teach", "guide", "learn", "mentor", "master"],
            "family": ["father", "mother", "brother", "sister", "parent"]
        }
        
        for interaction_type, indicators in interaction_types.items():
            if any(indicator in context_lower for indicator in indicators):
                return interaction_type
                
        return "neutral"

    def _validate_narrative_coherence(
        self,
        story_elements: Dict[str, Any],
        arc_analysis: Dict[str, Any],
        temporal_markers: List[Dict[str, Any]],
        character_relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate the overall coherence of the narrative."""
        validation = {
            "is_coherent": True,
            "issues": [],
            "strengths": []
        }
        
        # Check for basic story elements
        if not story_elements["characters"]:
            validation["is_coherent"] = False
            validation["issues"].append("No clear characters identified")
        else:
            validation["strengths"].append(f"Found {len(story_elements['characters'])} distinct characters")
            
        if not story_elements["settings"]:
            validation["is_coherent"] = False
            validation["issues"].append("No clear settings established")
        else:
            validation["strengths"].append(f"Found {len(story_elements['settings'])} distinct settings")
            
        # Check story arc
        if not arc_analysis["completeness"]:
            validation["is_coherent"] = False
            validation["issues"].append(
                f"Incomplete story arc. Missing: {', '.join(arc_analysis['missing_elements'])}"
            )
        else:
            validation["strengths"].append("Complete story arc present")
            
        # Check temporal consistency
        if temporal_markers:
            temporal_sequence = [marker["type"] for marker in temporal_markers]
            if "future" in temporal_sequence and "past" in temporal_sequence:
                if temporal_sequence.index("future") < temporal_sequence.index("past"):
                    validation["issues"].append("Potentially confusing temporal sequence")
        else:
            validation["issues"].append("Limited temporal markers for clear progression")
            
        # Check character relationship coherence
        if character_relationships:
            relationship_types = [rel["interaction_type"] for rel in character_relationships]
            if len(set(relationship_types)) > 1:
                validation["strengths"].append("Complex character relationships present")
        else:
            validation["issues"].append("Limited character interactions")
            
        return validation

    def validate_response(self, response: str, context: str) -> dict:
        """Ensure response is fully grounded in context."""
        validation_prompt = ChatPromptTemplate.from_template("""
            You are a Hatchyverse Fact Checker. Validate this response against the provided context.
            
            Response to validate:
            {response}
            
            Supporting Context:
            {context}
            
            Rules:
            1. Reject any names/terms not explicitly in context
            2. Flag unsupported numerical claims 
            3. Mark any speculative phrases like "would likely"
            4. Return JSON with keys: "is_valid", "errors", "corrected_response"
            
            Output JSON:
        """)
        
        chain = validation_prompt | self.llm.with_structured_output(
            schema=dict,
            method="json_mode",
            include_raw=False
        )
        
        return chain.invoke({
            "response": response,
            "context": context
        }) 