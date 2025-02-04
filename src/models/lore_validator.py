from typing import List, Dict, Any
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .lore_entity import LoreEntity
import logging
from collections import defaultdict
from itertools import chain

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

    def build_knowledge_base(self, documents: List[LoreEntity]) -> None:
        """Creates or updates the vector store with provided lore entities."""
        try:
            logger.debug(f"Building knowledge base with {len(documents) if documents else 'None'} documents")
            
            if documents is None:
                raise ValueError("documents parameter cannot be None")
                
            texts = []
            metadatas = []
            
            for entity in documents:
                logger.debug(f"Processing entity: {entity.name} ({entity.entity_type})")
                # Create rich text representation
                text = (
                    f"Name: {entity.name}\n"
                    f"Type: {entity.entity_type}\n"
                    f"Element: {entity.element}\n"
                    f"Description: {entity.description}\n"
                )
                
                if entity.relationships:
                    text += f"Relationships: {entity.relationships}\n"
                
                # Split into chunks if needed
                chunks = self.splitter.split_text(text)
                logger.debug(f"Split into {len(chunks)} chunks")
                
                for chunk in chunks:
                    texts.append(chunk)
                    metadatas.append({
                        "id": entity.id,
                        "name": entity.name,
                        "type": entity.entity_type,
                        "element": entity.element
                    })
            
            # Create or update vector store
            if self.vector_store is None:
                logger.debug("Creating new vector store")
                self.vector_store = FAISS.from_texts(
                    texts,
                    self.embeddings,
                    metadatas=metadatas
                )
                logger.info(f"Created new vector store with {len(texts)} chunks")
            else:
                logger.debug("Updating existing vector store")
                self.vector_store.add_texts(texts, metadatas=metadatas)
                logger.info(f"Added {len(texts)} new chunks to vector store")
                
        except Exception as e:
            logger.error(f"Error building knowledge base: {str(e)}", exc_info=True)
            raise

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