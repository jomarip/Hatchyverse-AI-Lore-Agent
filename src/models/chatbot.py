from typing import List, Dict, Any, Optional
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain_core.memory import BaseMemory
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field, ConfigDict
from .lore_validator import LoreValidator
import logging
from collections import defaultdict
import pandas as pd
import json
import yaml
from langchain_community.vectorstores import FAISS
from pathlib import Path
import uuid
import re

logger = logging.getLogger(__name__)

class FilteredRetriever(BaseRetriever):
    """Custom retriever that filters and prioritizes items based on query type and narrative context."""
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    base_retriever: BaseRetriever = Field(description="Base retriever to filter and enhance")
    vector_store: Any = Field(description="Vector store for additional filtering")
    item_store: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Dynamic item store containing categorized items and their metadata"
    )
    character_store: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Character registry with metadata and relationships"
    )
    timeline_store: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Timeline events and story arcs"
    )
    story_store: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Story segments and narrative context"
    )
    
    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_stores()
    
    def _initialize_stores(self):
        """Initialize all data stores with categorized information."""
        self._initialize_item_store()
        self._initialize_character_store()
        self._initialize_timeline_store()
        self._initialize_story_store()
    
    def _initialize_item_store(self):
        """Initialize the item store with categorized items and metadata."""
        # This would be populated from your item database files
        # Structure example:
        # {
        #     "element": {
        #         "fire": [{
        #             "name": "Flame Essence",
        #             "type": "consumable",
        #             "rarity": "common",
        #             "effects": ["attack_boost", "fire_damage"],
        #             "description": "...",
        #             "metadata": {...}
        #         }],
        #         "water": [...],
        #     },
        #     "type": {
        #         "weapon": [...],
        #         "armor": [...],
        #         "consumable": [...]
        #     },
        #     "rarity": {
        #         "common": [...],
        #         "rare": [...],
        #         "legendary": [...]
        #     }
        # }
        pass  # To be implemented when item data sources are connected
    
    def _initialize_character_store(self):
        """Initialize character registry with metadata and relationships."""
        # Structure:
        # {
        #     "character_id": {
        #         "name": str,
        #         "type": str,
        #         "affiliations": List[str],
        #         "relationships": Dict[str, str],
        #         "timeline_appearances": List[str],
        #         "story_arcs": List[str],
        #         "abilities": List[str],
        #         "metadata": Dict[str, Any]
        #     }
        # }
        pass  # To be populated from character data sources
    
    def _initialize_timeline_store(self):
        """Initialize timeline with events and story arcs."""
        # Structure:
        # {
        #     "event_id": {
        #         "name": str,
        #         "timestamp": str,
        #         "location": str,
        #         "characters": List[str],
        #         "story_arc": str,
        #         "description": str,
        #         "consequences": List[str],
        #         "metadata": Dict[str, Any]
        #     }
        # }
        pass  # To be populated from timeline data
    
    def _initialize_story_store(self):
        """Initialize story segments and narrative context."""
        # Structure:
        # {
        #     "story_id": {
        #         "title": str,
        #         "arc": str,
        #         "timeline_position": str,
        #         "locations": List[str],
        #         "characters": List[str],
        #         "events": List[str],
        #         "items": List[str],
        #         "content": str,
        #         "metadata": Dict[str, Any]
        #     }
        # }
        pass  # To be populated from story data
    
    def _get_relevant_items(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get relevant items based on query and filters."""
        relevant_items = []
        query_lower = query.lower()
        
        # Apply filters if provided
        filtered_items = []
        if filters:
            for category, value in filters.items():
                if category in self.item_store and value in self.item_store[category]:
                    filtered_items.extend(self.item_store[category][value])
        else:
            # If no filters, search across all items
            filtered_items = [
                item 
                for category in self.item_store.values()
                for subcategory in category.values()
                for item in subcategory
            ]
        
        # Score and rank items based on query relevance
        for item in filtered_items:
            score = 0
            # Check name match
            if query_lower in item["name"].lower():
                score += 2
            # Check description match
            if "description" in item and query_lower in item["description"].lower():
                score += 1
            # Check effects/properties match
            if "effects" in item:
                if any(query_lower in effect.lower() for effect in item["effects"]):
                    score += 1
            
            if score > 0:
                relevant_items.append((item, score))
        
        # Sort by relevance score
        relevant_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in relevant_items]
    
    def _get_relevant_characters(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get relevant characters based on query and filters."""
        relevant_chars = []
        query_lower = query.lower()
        
        for char_id, char_data in self.character_store.items():
            score = 0
            # Name match
            if query_lower in char_data["name"].lower():
                score += 3
            # Affiliation match
            if any(query_lower in aff.lower() for aff in char_data.get("affiliations", [])):
                score += 2
            # Ability match
            if any(query_lower in ability.lower() for ability in char_data.get("abilities", [])):
                score += 1
            
            if score > 0:
                relevant_chars.append((char_data, score))
        
        relevant_chars.sort(key=lambda x: x[1], reverse=True)
        return [char for char, _ in relevant_chars]
    
    def _get_relevant_timeline_events(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get relevant timeline events based on query and filters."""
        relevant_events = []
        query_lower = query.lower()
        
        for event_id, event_data in self.timeline_store.items():
            score = 0
            # Event name match
            if query_lower in event_data["name"].lower():
                score += 3
            # Location match
            if query_lower in event_data["location"].lower():
                score += 2
            # Character match
            if any(query_lower in char.lower() for char in event_data.get("characters", [])):
                score += 2
            # Description match
            if query_lower in event_data["description"].lower():
                score += 1
            
            if score > 0:
                relevant_events.append((event_data, score))
        
        relevant_events.sort(key=lambda x: x[1], reverse=True)
        return [event for event, _ in relevant_events]
    
    def _get_relevant_story_segments(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get relevant story segments based on query and filters."""
        relevant_segments = []
        query_lower = query.lower()
        
        for story_id, story_data in self.story_store.items():
            score = 0
            # Title match
            if query_lower in story_data["title"].lower():
                score += 3
            # Character match
            if any(query_lower in char.lower() for char in story_data.get("characters", [])):
                score += 2
            # Location match
            if any(query_lower in loc.lower() for loc in story_data.get("locations", [])):
                score += 2
            # Content match
            if query_lower in story_data["content"].lower():
                score += 1
            
            if score > 0:
                relevant_segments.append((story_data, score))
        
        relevant_segments.sort(key=lambda x: x[1], reverse=True)
        return [segment for segment, _ in relevant_segments]
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Synchronous implementation - required by BaseRetriever."""
        return self.invoke(query)
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Async implementation - required by BaseRetriever."""
        raise NotImplementedError("Async retrieval not implemented")
    
    async def ainvoke(self, input_str: str, **kwargs):
        """Asynchronously get documents relevant to the query."""
        return await self.base_retriever.ainvoke(input_str, **kwargs)
        
    def invoke(self, input_str: str, **kwargs):
        """Get documents relevant to the query with enhanced filtering."""
        try:
            # Get base results
            base_results = self.base_retriever.invoke(input_str, **kwargs)
            
            # Apply additional filtering and enhancement
            query_lower = input_str.lower()
            enhanced_results = []
            
            # Add relevant items if query is about items/equipment
            if any(keyword in query_lower for keyword in ["item", "equipment", "weapon"]):
                for items in self.item_store.values():
                    for item in items:
                        enhanced_results.append(Document(
                            page_content=f"{item.get('name', 'Unknown Item')}: {item.get('description', '')}",
                            metadata={"type": "item", "data": item}
                        ))
                        
            # Add character information if query is about characters
            if any(keyword in query_lower for keyword in ["character", "who", "hatchy"]):
                for char_id, char_data in self.character_store.items():
                    enhanced_results.append(Document(
                        page_content=f"{char_data.get('name', 'Unknown')}: {char_data.get('description', '')}",
                        metadata={"type": "character", "data": char_data}
                    ))
                    
            # Add story context if query is about events/story
            if any(keyword in query_lower for keyword in ["story", "event", "what happened"]):
                for story_id, story_data in self.story_store.items():
                    enhanced_results.append(Document(
                        page_content=f"{story_data.get('title', 'Unknown Event')}: {story_data.get('summary', '')}",
                        metadata={"type": "story", "data": story_data}
                    ))
            
            # Combine and deduplicate results
            all_results = base_results + enhanced_results
            seen = set()
            unique_results = []
            
            for doc in all_results:
                content_hash = hash(doc.page_content)
                if content_hash not in seen:
                    seen.add(content_hash)
                    unique_results.append(doc)
            
            return unique_results
            
        except Exception as e:
            logger.error(f"Error in FilteredRetriever.invoke: {str(e)}")
            # Fallback to base retriever results
            return self.base_retriever.invoke(input_str, **kwargs)

class LoreChatbot:
    """Manages conversations and lore interactions with users."""
    
    def __init__(self, data_store: Dict[str, List[Any]], validator: Optional[LoreValidator] = None):
        """Initialize the chatbot with data store and validator."""
        self.data_store = data_store
        self.validator = validator
        self.story_keywords = {
            'omniterra', 'felkyn', 'chaos god', 
            'alazaar', 'prophecy', 'arc', 'episode',
            'saga', 'story', 'tale', 'legend'
        }
        
        # Initialize data schema
        self.data_schema = {
            'hatchy': {},
            'items': {},
            'stories': {},
            'world_info': {}
        }
        
        # Initialize temperature for LLM
        self.temperature = 0.7
        
        # Initialize LLMs
        model_name = os.getenv('OPENAI_MODEL_NAME', 'gpt-3.5-turbo')
        fallback_model = os.getenv('OPENAI_FALLBACK_MODEL', 'gpt-4-0125-preview')
        
        # Create primary LLM
        self.primary_llm = ChatOpenAI(
            model_name=model_name,
            temperature=self.temperature
        )
        
        # Create fallback LLM
        self.fallback_llm = ChatOpenAI(
            model_name=fallback_model,
            temperature=self.temperature
        )
        
        self.base_prompt = ChatPromptTemplate.from_template("""
            You are a knowledgeable Hatchyverse Lorekeeper. Your role is to help users explore and 
            understand the Hatchyverse world while ensuring all information is accurate and based on the data.
            You must provide comprehensive, well-structured responses that cover all relevant aspects of the topic.

            Available data sources:
            {available_data}

            Context from knowledge base:
            {context}

            Current conversation history:
            {chat_history}

            User's message: {question}

            Instructions:
            1. ONLY use information that exists in the provided data
            2. When mentioning counts or statistics, use EXACT numbers from the data
            3. When listing hatchy or items, use ONLY names that appear in the data
            4. If information is not in the data, say so instead of making assumptions
            5. Pay attention to source files and generation information in the data
            6. Format responses with clear section headers and bullet points
            7. Include specific details but NEVER invent or assume details
            8. If you're not sure about something, say so explicitly
            9. For lore/story questions, prioritize these sources in order:
               a) Hatchy World Comic_ Chaos saga.txt
               b) World design documents
               c) Item/monster descriptions

            Response Structure:
            1. For general world/lore questions:
               - Overview: Brief introduction to the topic
               - Historical Context: Relevant historical events and significance
               - Key Elements: Important aspects, features, or components
               - Cultural Significance: Impact on Hatchyverse culture and society
               - Related Connections: Links to other aspects of the world
               - Notable Details: Specific interesting facts or characteristics

            2. For character-focused questions:
               - Introduction: Character's role and significance
               - Background: Origin and history
               - Relationships: Connections to other characters
               - Abilities & Traits: Notable characteristics and powers
               - Story Impact: Role in major events/narratives
               - Location Connections: Places of significance
               - Notable Moments: Key scenes or actions

            3. For location-based questions:
               - Description: Physical characteristics and atmosphere
               - History: Origins and development
               - Significance: Role in the world/story
               - Inhabitants: Notable residents or creatures
               - Features: Special attributes or landmarks
               - Events: Important occurrences
               - Connections: Links to other locations/elements

            4. For item/artifact questions:
               - Description: Physical appearance and properties
               - Origin: Creation or discovery history
               - Powers: Abilities and effects
               - Significance: Role in lore/story
               - Usage: How it's used or wielded
               - Related Items: Connections to other artifacts
               - Notable Events: Key moments involving the item

            5. For story/event questions:
               - Context: Setting and background
               - Key Players: Important characters involved
               - Sequence: Order of major events
               - Impact: Effects on the world/characters
               - Themes: Major themes and motifs
               - Connections: Links to other stories/events
               - Legacy: Lasting consequences

            6. For system/mechanics questions:
               - Overview: Basic concept explanation
               - Components: Key parts or elements
               - Functionality: How it works
               - Applications: Uses and implementations
               - Limitations: Constraints or restrictions
               - Examples: Specific instances or cases
               - Integration: Connection to other systems

            Remember to:
            - Use clear section headers (## for main sections, ### for subsections)
            - Include bullet points for lists and details
            - Provide specific examples when available
            - Maintain a logical flow between sections
            - Cross-reference related information
            - Cite sources when possible
            - Note any uncertainties or gaps in knowledge
        """)
        
        # Initialize empty QA chain - will be created after data is loaded
        self.qa_chain = None
        
        self.hatchy_data = []  # Will store loaded hatchy data
        
        # Add narrative analysis prompts
        self.narrative_analysis_prompt = ChatPromptTemplate.from_template("""
            You are a skilled Hatchyverse Narrative Analyst. Your role is to analyze and provide feedback on story submissions
            while ensuring they align with the established lore and maintain narrative quality.

            Story Analysis Results:
            {analysis_results}

            Narrative Elements Found:
            {narrative_elements}

            Existing Lore Context:
            {lore_context}

            Instructions:
            1. Evaluate the story's alignment with existing lore
            2. Assess narrative structure and coherence
            3. Identify strengths and potential improvements
            4. Suggest specific enhancements that maintain lore consistency
            5. Highlight particularly effective elements

            Please provide a detailed analysis focusing on:
            1. Lore Consistency
            2. Character Development
            3. Plot Structure
            4. World-building
            5. Theme Integration

            Assistant: Let me provide a comprehensive analysis based on the story elements and lore context.
        """)
        
    def _create_qa_chain(self) -> ConversationalRetrievalChain:
        """Create chain with anti-hallucination safeguards."""
        qa_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
                You are a Hatchyverse Lore Expert. Follow these rules:
                
                1. Only use information from the provided context
                2. If unsure, say "According to available records..."
                3. Never invent names or numbers
                4. Cite sources from metadata when possible
                
                Available Data:
                {formatted_data}
                
                Current Conversation:
                {chat_history}
                
                Question: 
                {question}
                
                Context:
                {context}
                """),
            HumanMessagePromptTemplate.from_template("{question}")
        ])
        
        return ConversationalRetrievalChain.from_llm(
            self.primary_llm,
            self.validator.vector_store.as_retriever(),
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            memory=None,  # No memory needed as we handle history explicitly
            verbose=True
        )
    
    @retry(
        stop=stop_after_attempt(int(os.getenv("MAX_RETRIES", 3))),
        wait=wait_exponential(
            multiplier=float(os.getenv("INITIAL_RETRY_DELAY", 1)),
            max=float(os.getenv("MAX_RETRY_DELAY", 60))
        )
    def _process_with_fallback(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Processes a request with fallback to GPT-4 if needed."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "rate_limit" in str(e).lower() and self.fallback_llm:
                logger.info("Rate limit hit, falling back to GPT-4")
                # Temporarily swap to fallback model
                original_chain = self.qa_chain
                self.qa_chain = self._create_qa_chain()
                try:
                    return func(*args, **kwargs)
                finally:
                    # Restore primary model
                    self.qa_chain = original_chain
            raise
            
    def _process_lore_submission(self, submission: str) -> Dict[str, Any]:
        """
        Processes a new lore submission for validation.
        
        Args:
            submission: The new lore content to validate
            
        Returns:
            Dict containing validation results and feedback
        """
        try:
            # Check for conflicts
            validation = self.validator.check_conflict(submission)
            
            # Generate appropriate response based on validation results
            if validation["conflicts"]:
                response = (
                    "⚠️ I've detected some potential conflicts with existing lore:\n\n"
                )
                
                for conflict in validation["conflicts"]:
                    # Handle conflicts with or without metadata
                    if 'metadata' in conflict and conflict['metadata'].get('name'):
                        response += f"- Conflicts with: {conflict['metadata']['name']}\n"
                    else:
                        response += f"- Conflict: {conflict['reason']}\n"
                        
                    if conflict.get('content'):
                        response += f"  Details: {conflict['content']}\n\n"
                    
                response += "\nSuggested improvements:\n"
                # Use LLM to generate improvement suggestions
                suggestions_prompt = f"""
                Given this new lore submission:
                {submission}

                And these conflicts with existing lore:
                {validation['conflicts']}

                Provide 2-3 specific suggestions for modifying the submission to better align with canon.
                Focus on maintaining the core idea while resolving conflicts.
                """
                suggestions = self.primary_llm.invoke(suggestions_prompt).content
                response += suggestions
                
            else:
                response = "✅ Your submission appears to align well with existing lore!\n\n"
                if validation["similar_concepts"]:
                    response += "Related existing elements you might want to reference:\n"
                    for concept in validation["similar_concepts"]:
                        if 'metadata' in concept and concept['metadata'].get('name'):
                            response += f"- {concept['metadata']['name']}: {concept['content']}\n"
                        else:
                            response += f"- Related: {concept['content']}\n"
            
            return {
                "response": response,
                "validation": validation
            }
            
        except Exception as e:
            logger.error(f"Error processing lore submission: {str(e)}")
            raise
            
    def _format_list_response(self, entities: List[Dict[str, Any]], query: str) -> str:
        """Format a response for list-type queries."""
        if not entities:
            return "I couldn't find any matching entities in the knowledge base."
            
        # Group entities by type and element
        grouped = defaultdict(lambda: defaultdict(list))
        for entity in entities:
            entity_type = entity.get('type', 'Unknown')
            element = entity.get('element') or 'No Element'
            grouped[entity_type][element].append(entity)
            
        response = []
        
        # Format the response
        for entity_type, element_groups in grouped.items():
            response.append(f"\n{entity_type}s by Element:")
            
            for element, entities in element_groups.items():
                response.append(f"\n{element}:")
                for entity in sorted(entities, key=lambda x: x.get('name', '')):
                    name = entity.get('name', 'Unknown')
                    desc = entity.get('description', '').split('\n')[0]  # First line only
                    response.append(f"- {name}: {desc}")
                    
        # Add summary
        total = sum(len(entities) for elements in grouped.values() for entities in elements.values())
        response.insert(0, f"Found {total} {entity_type.lower()}s total.")
        
        return "\n".join(response)

    def load_hatchy_data(self, data: List[Dict[str, Any]]):
        """Load hatchy monster data directly."""
        self.hatchy_data = data
        
    def get_hatchy_by_generation(self, generation: int, element: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all hatchy from a specific generation, optionally filtered by element."""
        # Get hatchy from the data store instead of hatchy_data
        hatchy_list = [h for h in self.data_store['hatchy'] 
                      if str(generation) in (h.get('Generation', ''), h.get('_source_file', ''))]
            
        # Apply element filter if specified
        if element and hatchy_list:
            return [h for h in hatchy_list if h.get('Element', '').lower() == element.lower()]
        return hatchy_list
        
    def _format_available_data(self) -> str:
        """Format available data sources for the prompt."""
        data_summary = []
        
        # Format hatchy data
        if self.data_store.get('hatchy'):
            data_summary.append(f"Hatchy Data: {len(self.data_store['hatchy'])} entries")
        
        # Format item data
        if self.data_store.get('items'):
            data_summary.append(f"Item Data: {len(self.data_store['items'])} entries")
        
        # Format story data
        if self.data_store.get('stories'):
            story_titles = [story.get('title', 'Untitled Story') for story in self.data_store['stories']]
            data_summary.append(f"Story Data: {', '.join(story_titles)}")
        
        # Format world info
        if self.data_store.get('world_info'):
            world_titles = [info.get('title', 'World Info') for info in self.data_store['world_info']]
            data_summary.append(f"World Info: {', '.join(world_titles)}")
        
        if not data_summary:
            return "No data sources available."
        
        return "\n".join(data_summary)

    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate a response to a user query."""
        try:
            # Validate query
            validation = self._validate_query(query)
            if not validation['is_valid']:
                return {
                    'response': validation['message'],
                    'validation': validation
                }
            
            # Get relevant context
            context = self._get_relevant_context(query)
            
            # Extract entity mentions
            entity_mentions = self._extract_entity_mentions(query)
            
            # Generate response using LLM
            response = self._generate_llm_response(query, context, entity_mentions)
            
            return {
                'response': response,
                'validation': {'is_valid': True, 'message': 'Valid query'}
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                'response': f"I apologize, but I encountered an error while processing your query: {str(e)}",
                'validation': {'is_valid': False, 'message': str(e)}
            }
            
    def _validate_query(self, query: str) -> Dict[str, Any]:
        """Validate the user query."""
        if not query.strip():
            return {
                'is_valid': False,
                'message': "Please provide a non-empty query."
            }
            
        if len(query) < 3:
            return {
                'is_valid': False,
                'message': "Query is too short. Please provide a more detailed question."
            }
            
        return {
            'is_valid': True,
            'message': "Valid query"
        }

    def _generate_llm_response(self, query: str, context: str, entity_mentions: List[str]) -> str:
        """Generate a response using the primary LLM based on the query, context, and entity mentions."""
        # Format the prompt
        prompt = self.base_prompt.format(
            question=query,
            context=context,
            chat_history="",
            available_data=self._format_available_data()
        )
        
        # Add entity mentions to the prompt
        for mention in entity_mentions:
            prompt += f"\n\nEntity Mention: {mention}"
        
        # Generate response using the primary LLM
        response = self.primary_llm.invoke(prompt).content
        
        return response

    def _get_relevant_context(self, query: str) -> str:
        """Get relevant context for the query with enhanced generation handling."""
        try:
            # Extract generation info from query using multiple patterns
            gen_patterns = [
                r'gen\s*(\d+)',
                r'generation\s*(\d+)',
                r'gen-?(\d+)'
            ]
            
            generation = None
            for pattern in gen_patterns:
                gen_match = re.search(pattern, query.lower())
                if gen_match:
                    generation = gen_match.group(1)
                    break
            
            # If asking about specific generation of Hatchy
            if generation and 'hatchy' in self.data_store:
                # Get Hatchy data from the specific generation
                gen_hatchy = [
                    h for h in self.data_store['hatchy'] 
                    if str(generation) == str(h.get('metadata', {}).get('generation', '')) or
                    (h.get('metadata', {}).get('generation_source') == 'filename' and 
                     str(generation) in h.get('metadata', {}).get('source', '').lower())
                ]
                
                if gen_hatchy:
                    # Group by element
                    by_element = defaultdict(list)
                    for hatchy in gen_hatchy:
                        element = hatchy.get('element', 'Unknown')
                        by_element[element].append(hatchy)
                    
                    # Format response with rich context
                    response_parts = [f"## Generation {generation} Hatchy Information\n"]
                    
                    # Add each element section
                    for element, hatchies in sorted(by_element.items()):
                        response_parts.append(f"\n### {element} Element")
                        for hatchy in sorted(hatchies, key=lambda x: x.get('name', '')):
                            name = hatchy.get('name', 'Unknown')
                            desc = hatchy.get('description', '').strip()
                            metadata = hatchy.get('metadata', {})
                            
                            # Add detailed entry
                            entry = [f"#### {name}"]
                            entry.append(f"Description: {desc}")
                            
                            # Add metadata if available
                            if metadata.get('height'):
                                entry.append(f"Height: {metadata['height']}")
                            if metadata.get('weight'):
                                entry.append(f"Weight: {metadata['weight']}")
                                
                            response_parts.append("\n".join(entry))
                    
                    # Add supplemental information if available
                    supplemental = [
                        h for h in self.data_store['hatchy']
                        if h.get('metadata', {}).get('file_group') == 'supplemental' and
                        str(generation) == str(h.get('metadata', {}).get('generation', ''))
                    ]
                    
                    if supplemental:
                        response_parts.append("\n### Additional Information")
                        for supp in supplemental:
                            response_parts.append(f"- {supp.get('description', '')}")
                    
                    return "\n".join(response_parts)
                else:
                    return f"No Hatchy data found for Generation {generation}."
            
            # Get base context for other queries
            relevant_docs = []
            
            # If generation specified, filter by generation in metadata
            if generation:
                relevant_docs = self.validator.vector_store.similarity_search(
                    query,
                    filter={"generation": str(generation)},
                    k=5
                )
                
                # Also get supplemental docs
                supp_docs = self.validator.vector_store.similarity_search(
                    query,
                    filter={"file_group": "supplemental", "generation": str(generation)},
                    k=3
                )
                relevant_docs.extend(supp_docs)
            else:
                relevant_docs = self.validator.search_knowledge_base(query)
            
            if relevant_docs:
                # Handle both Document objects and dictionaries
                context_parts = []
                for doc in relevant_docs:
                    if hasattr(doc, 'page_content'):
                        context_parts.append(doc.page_content)
                    elif isinstance(doc, dict):
                        # Extract content from dictionary format
                        if 'content' in doc:
                            context_parts.append(doc['content'])
                        elif 'text' in doc:
                            context_parts.append(doc['text'])
                return "\n".join(context_parts)
            return ""
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return ""

    def _extract_entity_mentions(self, text: str) -> List[str]:
        """Extract potential entity mentions from text."""
        # Get all entities from the knowledge graph
        all_entities = self.validator.vector_store.entities.values()
        
        # Sort by name length (longest first) to catch longer names before shorter ones
        entity_names = sorted(
            [entity['name'] for entity in all_entities],
            key=len,
            reverse=True
        )
        
        # Find matches in text
        matches = []
        text_lower = text.lower()
        
        for name in entity_names:
            if name.lower() in text_lower:
                matches.append(name)
        
        return matches

class EnhancedChatbot:
    """Enhanced chatbot with knowledge graph integration."""
    
    def __init__(
        self,
        llm: Any,
        knowledge_graph: Any,
        vector_store: Any,
        base_prompt: Optional[str] = None
    ):
        self.primary_llm = llm
        self.knowledge_graph = knowledge_graph
        self.vector_store = vector_store
        self.base_prompt = base_prompt or DEFAULT_PROMPT
        
    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate a response to a user query."""
        try:
            # Validate query
            validation = self._validate_query(query)
            if not validation['is_valid']:
                return {
                    'response': validation['message'],
                    'validation': validation
                }
            
            # Get relevant context
            context = self._get_relevant_context(query)
            
            # Extract entity mentions
            entity_mentions = self._extract_entity_mentions(query)
            
            # Generate response using LLM
            response = self._generate_llm_response(query, context, entity_mentions)
            
            return {
                'response': response,
                'validation': {'is_valid': True, 'message': 'Valid query'}
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                'response': f"I apologize, but I encountered an error while processing your query: {str(e)}",
                'validation': {'is_valid': False, 'message': str(e)}
            }
            
    def _validate_query(self, query: str) -> Dict[str, Any]:
        """Validate the user query."""
        if not query.strip():
            return {
                'is_valid': False,
                'message': "Please provide a non-empty query."
            }
            
        if len(query) < 3:
            return {
                'is_valid': False,
                'message': "Query is too short. Please provide a more detailed question."
            }
            
        return {
            'is_valid': True,
            'message': "Valid query"
        }
        
    def _extract_entity_mentions(self, text: str) -> List[str]:
        """Extract potential entity mentions from text."""
        # Get all entities from the knowledge graph
        all_entities = self.knowledge_graph.entities.values()
        
        # Sort by name length (longest first) to catch longer names before shorter ones
        entity_names = sorted(
            [entity['name'] for entity in all_entities],
            key=len,
            reverse=True
        )
        
        # Find matches in text
        matches = []
        text_lower = text.lower()
        
        for name in entity_names:
            if name.lower() in text_lower:
                matches.append(name)
        
        return matches
        
    def _get_relevant_context(self, query: str) -> str:
        """Get relevant context for the query."""
        try:
            # Get relevant documents from vector store
            docs = self.vector_store.similarity_search(
                query,
                k=5
            )
            
            # Extract entity mentions
            entity_mentions = self._extract_entity_mentions(query)
            
            # Get context from knowledge graph
            kg_context = []
            for entity_name in entity_mentions:
                entity = self.knowledge_graph.get_entity_by_name(entity_name)
                if entity:
                    context = self.knowledge_graph.get_entity_context(
                        entity['id'],
                        include_relationships=True
                    )
                    if context:
                        kg_context.append(context)
            
            # Format context
            context_parts = []
            
            # Add vector store results
            if docs:
                context_parts.append("From Vector Store:")
                for doc in docs:
                    context_parts.append(doc.page_content)
            
            # Add knowledge graph context
            if kg_context:
                context_parts.append("\nFrom Knowledge Graph:")
                for ctx in kg_context:
                    context_parts.append(
                        f"Entity: {ctx['entity']['name']}\n"
                        f"Type: {ctx['entity']['type']}\n"
                        f"Attributes: {json.dumps(ctx['entity']['attributes'], indent=2)}"
                    )
                    if ctx.get('related_entities'):
                        context_parts.append("Related Entities:")
                        for rel in ctx['related_entities']:
                            context_parts.append(
                                f"- {rel['entity']['name']} "
                                f"({rel['relationship']})"
                            )
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return ""
            
    def _generate_llm_response(self, query: str, context: str, entity_mentions: List[str]) -> str:
        """Generate a response using the LLM."""
        try:
            # Format prompt
            prompt = f"""Question: {query}

Available Context:
{context}

Entity Mentions: {', '.join(entity_mentions) if entity_mentions else 'None'}

Please provide a detailed answer based on the available context. If there isn't enough context to fully answer the question, please indicate what information is missing."""
            
            # Generate response
            response = self.primary_llm.invoke(prompt)
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question." 