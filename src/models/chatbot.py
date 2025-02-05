from typing import List, Dict, Any, Optional
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain_core.memory import BaseMemory
from langchain_core.prompts import ChatPromptTemplate
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
        """Creates a ConversationalRetrievalChain with the current LLM and data."""
        try:
            if not self.validator.vector_store:
                # Create initial vector store if none exists
                initial_texts = ["Hatchyverse initialization document"]
                self.validator.vector_store = FAISS.from_texts(
                    initial_texts,
                    self.validator.embeddings
                )
            
            # Create the filtered retriever
            base_retriever = self.validator.vector_store.as_retriever()
            filtered_retriever = FilteredRetriever(
                base_retriever=base_retriever,
                vector_store=self.validator.vector_store
            )
            
            return ConversationalRetrievalChain.from_llm(
                llm=self.primary_llm,
                retriever=filtered_retriever,
                combine_docs_chain_kwargs={"prompt": self.base_prompt}
            )
        except Exception as e:
            logger.error(f"Error creating QA chain: {e}")
            # Fallback to simple retriever if filtered retriever fails
            if self.validator.vector_store:
                return ConversationalRetrievalChain.from_llm(
                    llm=self.primary_llm,
                    retriever=self.validator.vector_store.as_retriever(),
                    combine_docs_chain_kwargs={"prompt": self.base_prompt}
                )
            else:
                raise ValueError("No vector store available for QA chain creation")
    
    @retry(
        stop=stop_after_attempt(int(os.getenv("MAX_RETRIES", 3))),
        wait=wait_exponential(
            multiplier=float(os.getenv("INITIAL_RETRY_DELAY", 1)),
            max=float(os.getenv("MAX_RETRY_DELAY", 60))
        )
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

    def generate_response(self, query: str, chat_history: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a response to a user query."""
        try:
            # Initialize chat history if None
            if chat_history is None:
                chat_history = []
            elif isinstance(chat_history, str):
                chat_history = [chat_history] if chat_history else []
            
            # Extract filters from query
            filters = self._extract_query_filters(query)
            
            # Get relevant context
            context = self._get_relevant_context(query, filters)
            
            # Format available data
            data_summary = self._format_available_data()
            
            # Format chat history
            formatted_history = self._format_chat_history(chat_history)
            
            # Combine all context
            full_context = f"{data_summary}\n\n{context}\n\n{formatted_history}"
            
            # Create QA chain if needed
            if not self.qa_chain:
                self.qa_chain = self._create_qa_chain()
            
            # Process query with fallback handling
            response = self.qa_chain.invoke({
                "question": query,
                "chat_history": chat_history,
                "context": full_context,
                "available_data": data_summary
            })
            
            # Get validation results if this is a submission
            validation_results = {}
            if query.startswith("[SUBMIT]"):
                validation_results = self._process_lore_submission(query[8:])
            
            return {
                "response": response.get("answer", response.get("response", response.get("content", "I could not generate a response."))),
                "validation": validation_results
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error while processing your request. Please try again or rephrase your question.",
                "validation": {}
            }

    def _extract_query_filters(self, query: str) -> Dict[str, Dict[str, Any]]:
        """Extract filters from the query text."""
        filters = {}
        
        # Extract generation info
        if "gen1" in query.replace(" ", "") or "generation 1" in query:
            filters["hatchy"] = {"Generation": "1"}
        elif "gen2" in query.replace(" ", "") or "generation 2" in query:
            filters["hatchy"] = {"Generation": "2"}
            
        # Extract element info
        elements = ["fire", "water", "earth", "air", "light", "dark"]
        for element in elements:
            if element in query:
                if "hatchy" not in filters:
                    filters["hatchy"] = {}
                filters["hatchy"]["Element"] = element.capitalize()
                
        # Extract item type info
        item_types = ["weapon", "armor", "accessory", "consumable"]
        for item_type in item_types:
            if item_type in query:
                filters["items"] = {"type": item_type}
                
        return filters

    def _format_entity_details(self, entities: List[Dict[str, Any]]) -> str:
        """Format entity details in a clear, readable way."""
        formatted = []
        
        for entity in entities:
            # Skip internal fields
            details = {k: v for k, v in entity.items() 
                      if not k.startswith('_') and v is not None}
            
            # Format based on entity type
            if entity.get('_data_type') == 'hatchy':
                formatted.append(f"Name: {details.get('Name', 'Unknown')}")
                if 'Element' in details:
                    formatted.append(f"Element: {details['Element']}")
                if 'Description' in details:
                    formatted.append(f"Description: {details['Description']}")
                    
            elif entity.get('_data_type') == 'items':
                formatted.append(f"Item: {details.get('name', 'Unknown')}")
                if 'description' in details:
                    formatted.append(f"Description: {details['description']}")
                if 'effects' in details:
                    formatted.append(f"Effects: {details['effects']}")
                    
            formatted.append("")  # Add spacing between entities
            
        return "\n".join(formatted)

    def get_specific_entities(self, data_type: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get specific entities from the data store with flexible filtering."""
        results = []
        
        for item in self.data_store.get(data_type, []):
            matches = True
            for key, value in filters.items():
                # Handle nested keys (e.g., 'metadata.element')
                item_value = item
                for key_part in key.split('.'):
                    if isinstance(item_value, dict) and key_part in item_value:
                        item_value = item_value[key_part]
                    else:
                        item_value = None
                        break
                
                # Flexible matching
                if item_value is not None:
                    if isinstance(value, (list, tuple)):
                        if not any(self._flexible_match(item_value, v) for v in value):
                            matches = False
                            break
                    else:
                        if not self._flexible_match(item_value, value):
                            matches = False
                            break
                else:
                    matches = False
                    break
            
            if matches:
                results.append(item)
        
        return results

    def _flexible_match(self, value1: Any, value2: Any) -> bool:
        """Flexible value matching with type coercion and case insensitivity."""
        # Convert to strings for case-insensitive comparison
        if isinstance(value1, str) and isinstance(value2, str):
            return value1.lower() == value2.lower()
        
        # Handle numeric comparisons
        try:
            if isinstance(value1, (int, float)) or isinstance(value2, (int, float)):
                num1 = float(value1) if value1 is not None else None
                num2 = float(value2) if value2 is not None else None
                return num1 == num2
        except (ValueError, TypeError):
            pass
            
        # Direct comparison fallback
        return value1 == value2

    def _chunk_text(self, text: str, max_tokens: int = 8000) -> List[str]:
        """Split text into chunks that won't exceed token limits."""
        # Rough estimate: 1 token ≈ 4 chars for English text
        chunk_size = max_tokens * 4
        chunks = []
        
        while text:
            if len(text) <= chunk_size:
                chunks.append(text)
                break
            
            # Try to split at a natural boundary
            split_point = text.rfind('\n', 0, chunk_size)
            if split_point == -1:
                split_point = text.rfind('. ', 0, chunk_size)
            if split_point == -1:
                split_point = chunk_size
                
            chunks.append(text[:split_point])
            text = text[split_point:].lstrip()
            
        return chunks

    def _batch_texts_for_embedding(self, texts: List[str], batch_size: int = 50) -> List[List[str]]:
        """Split texts into batches for embedding."""
        return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    def _update_vector_store_with_rate_limit(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        """Update vector store with rate limiting and chunking."""
        try:
            # First, chunk any large texts
            chunked_texts = []
            chunked_metadatas = []
            
            for text, metadata in zip(texts, metadatas):
                if len(text) > 32000:  # Conservative limit
                    chunks = self._chunk_text(text)
                    chunked_texts.extend(chunks)
                    # Duplicate metadata for each chunk
                    chunked_metadatas.extend([metadata.copy() for _ in chunks])
                else:
                    chunked_texts.append(text)
                    chunked_metadatas.append(metadata)
            
            # Then batch the chunks
            text_batches = self._batch_texts_for_embedding(chunked_texts)
            metadata_batches = self._batch_texts_for_embedding(chunked_metadatas)
            
            # Process each batch with delay between batches
            for text_batch, metadata_batch in zip(text_batches, metadata_batches):
                if self.validator.vector_store is None:
                    self.validator.vector_store = FAISS.from_texts(
                        text_batch,
                        self.validator.embeddings,
                        metadatas=metadata_batch
                    )
                else:
                    try:
                        self.validator.vector_store.add_texts(text_batch, metadatas=metadata_batch)
                    except Exception as e:
                        if "rate_limit" in str(e).lower():
                            logger.warning("Rate limit hit, waiting before retry...")
                            time.sleep(20)  # Wait 20 seconds before retry
                            self.validator.vector_store.add_texts(text_batch, metadatas=metadata_batch)
                        else:
                            raise
                
                # Add delay between batches to respect rate limits
                time.sleep(1)  # 1 second delay between batches
                
        except Exception as e:
            logger.error(f"Error updating vector store: {str(e)}")
            raise

    def load_data(self, file_path: str, data_type: str):
        """Load data with rich relationships and multiple perspectives.
        
        Creates multiple views of the data:
        - Hierarchical (type/category based)
        - Attribute-based (property relationships)
        - Temporal (timeline/evolution based)
        - Contextual (situation/usage based)
        - Cross-referenced (entity relationships)
        """
        try:
            file_name = Path(file_path).name
            
            # Read the data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                data = df.to_dict('records')
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif file_path.endswith(('.yml', '.yaml')):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            else:
                raise ValueError("Unsupported file format")

            # Enhance data with rich metadata and relationships
            enhanced_data = []
            relationship_graph = defaultdict(list)
            attribute_index = defaultdict(list)
            temporal_sequence = defaultdict(list)
            
            for item in data:
                # Add base metadata
                item['_source_file'] = file_name
                item['_data_type'] = data_type
                item['_id'] = str(uuid.uuid4())
                
                # Create attribute-based connections
                for key, value in item.items():
                    if not key.startswith('_'):
                        attribute_index[f"{key}:{value}"].append(item['_id'])
                        
                # Build relationship graph based on shared attributes
                for other_item in data:
                    if other_item != item:
                        shared_attrs = self._find_shared_attributes(item, other_item)
                        if shared_attrs:
                            relationship_graph[item['_id']].append({
                                'target_id': other_item.get('_id'),
                                'shared_attributes': shared_attrs,
                                'relationship_strength': len(shared_attrs)
                            })
                
                # Add temporal context if available
                if any(key in item for key in ['date', 'timeline', 'generation', 'evolution']):
                    temporal_key = next(key for key in ['date', 'timeline', 'generation', 'evolution'] 
                                     if key in item)
                    temporal_sequence[temporal_key].append(item['_id'])
                
                # Create multiple perspectives of the item
                perspectives = self._generate_item_perspectives(item)
                item['_perspectives'] = perspectives
                
                enhanced_data.append(item)
            
            # Store enhanced data
            self.data_store[data_type].extend(enhanced_data)
            
            # Create rich text embeddings that capture multiple perspectives
            texts = []
            metadatas = []
            
            for item in enhanced_data:
                # Generate multiple text representations for better embedding coverage
                text_representations = []
                
                # Basic context (keep minimal)
                text_representations.append(f"From {file_name} ({data_type} data):")
                
                # Add essential perspectives only
                if 'descriptive' in item['_perspectives']:
                    text_representations.append(f"\nDescription:\n{item['_perspectives']['descriptive']}")
                if 'functional' in item['_perspectives']:
                    text_representations.append(f"\nFunction:\n{item['_perspectives']['functional']}")
                
                # Add core relationships only
                if item['_id'] in relationship_graph:
                    strong_relationships = [
                        rel for rel in relationship_graph[item['_id']]
                        if rel['relationship_strength'] > 2  # Only strong relationships
                    ][:3]  # Limit to top 3
                    if strong_relationships:
                        text_representations.append("\nKey Relationships:")
                        for rel in strong_relationships:
                            text_representations.append(f"- Connected through {', '.join(rel['shared_attributes'][:3])}")
                
                # Combine all representations
                full_text = "\n".join(text_representations)
                texts.append(full_text)
                
                # Simplified metadata for retrieval
                metadata = {
                    "source_file": file_name,
                    "data_type": data_type,
                    "item_id": item['_id'],
                    **{k: v for k, v in item.items() if not k.startswith('_') 
                       and not isinstance(v, (list, dict))}  # Only include simple fields
                }
                metadatas.append(metadata)
            
            # Update vector store with rate limiting and chunking
            self._update_vector_store_with_rate_limit(texts, metadatas)
            
            # Store relationship data for cross-referencing
            self._store_relationships(relationship_graph, data_type)
            self._store_attribute_index(attribute_index, data_type)
            self._store_temporal_sequence(temporal_sequence, data_type)
            
            # Update schema with new perspectives
            if enhanced_data:
                self.data_schema[data_type] = self._analyze_enhanced_schema(enhanced_data[0])
            
            # Create or update QA chain
            if self.qa_chain is None:
                self.qa_chain = self._create_qa_chain()
            
            logger.info(f"Loaded {len(enhanced_data)} {data_type} entries with rich relationships from {file_name}")
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise

    def _find_shared_attributes(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> List[str]:
        """Find meaningful shared attributes between items."""
        shared = []
        for key in item1:
            if not key.startswith('_') and key in item2:
                if item1[key] == item2[key]:
                    shared.append(key)
        return shared

    def _generate_item_perspectives(self, item: Dict[str, Any]) -> Dict[str, str]:
        """Generate multiple perspectives of an item for rich embeddings."""
        perspectives = {}
        
        # Categorical perspective
        category_text = []
        for key, value in item.items():
            if not key.startswith('_') and isinstance(value, str):
                category_text.append(f"{key}: {value}")
        perspectives['categorical'] = "\n".join(category_text)
        
        # Descriptive perspective
        if 'description' in item:
            perspectives['descriptive'] = item['description']
        
        # Functional perspective (for items/abilities)
        functional_keys = ['abilities', 'effects', 'usage', 'function']
        functional_text = []
        for key in functional_keys:
            if key in item:
                functional_text.append(f"{key}: {item[key]}")
        if functional_text:
            perspectives['functional'] = "\n".join(functional_text)
        
        # Relational perspective
        relational_keys = ['related_to', 'evolves_from', 'evolves_to', 'requires']
        relational_text = []
        for key in relational_keys:
            if key in item:
                relational_text.append(f"{key}: {item[key]}")
        if relational_text:
            perspectives['relational'] = "\n".join(relational_text)
        
        # Statistical perspective (for numerical attributes)
        stats_text = []
        for key, value in item.items():
            if isinstance(value, (int, float)) and not key.startswith('_'):
                stats_text.append(f"{key}: {value}")
        if stats_text:
            perspectives['statistical'] = "\n".join(stats_text)
        
        return perspectives

    def _store_relationships(self, graph: Dict[str, List[Dict[str, Any]]], data_type: str):
        """Store relationship graph for cross-referencing."""
        if not hasattr(self, 'relationship_store'):
            self.relationship_store = defaultdict(dict)
        self.relationship_store[data_type].update(graph)

    def _store_attribute_index(self, index: Dict[str, List[str]], data_type: str):
        """Store attribute index for flexible querying."""
        if not hasattr(self, 'attribute_store'):
            self.attribute_store = defaultdict(dict)
        self.attribute_store[data_type].update(index)

    def _store_temporal_sequence(self, sequence: Dict[str, List[str]], data_type: str):
        """Store temporal relationships."""
        if not hasattr(self, 'temporal_store'):
            self.temporal_store = defaultdict(dict)
        self.temporal_store[data_type].update(sequence)

    def _analyze_enhanced_schema(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze schema including perspectives and relationships."""
        schema = self._analyze_schema(sample_data)
        
        # Add perspective schemas
        if '_perspectives' in sample_data:
            schema['_perspectives'] = {
                perspective: 'text'
                for perspective in sample_data['_perspectives']
            }
        
        # Add relationship schema
        schema['_relationships'] = {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'target_id': 'string',
                    'shared_attributes': 'array',
                    'relationship_strength': 'number'
                }
            }
        }
        
        return schema

    def _analyze_schema(self, sample_data: Dict[str, Any]) -> Dict[str, str]:
        """Analyze the schema of the data dynamically."""
        schema = {}
        for key, value in sample_data.items():
            if isinstance(value, (int, float)):
                schema[key] = 'number'
            elif isinstance(value, bool):
                schema[key] = 'boolean'
            elif isinstance(value, list):
                schema[key] = 'array'
            elif isinstance(value, dict):
                schema[key] = 'object'
            else:
                schema[key] = 'string'
        return schema

    def load_hatchy_data_from_file(self, file_path: str):
        """Load hatchy data from JSON, YAML, or CSV file."""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                data = df.to_dict('records')
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif file_path.endswith(('.yml', '.yaml')):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            else:
                raise ValueError("Unsupported file format. Use CSV, JSON, or YAML.")
                    
            self.load_hatchy_data(data)
            logger.info(f"Successfully loaded {len(data)} hatchy from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading hatchy data from {file_path}: {str(e)}")
            raise

    def load_text_file(self, file_path: str, data_type: str):
        """Load and process text files with rich context preservation."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_name = Path(file_path).name
            
            # Split content into meaningful sections
            sections = self._split_text_into_sections(content)
            
            # Process each section as a separate document
            enhanced_data = []
            for i, section in enumerate(sections):
                # Extract section title and clean content
                title = self._extract_section_title(section)
                clean_content = self._clean_section_content(section)
                
                section_data = {
                    '_id': str(uuid.uuid4()),
                    '_source_file': file_name,
                    '_data_type': data_type,
                    'content': clean_content,
                    'section_number': i + 1,
                    'title': title or f"Section {i + 1}",
                }
                
                # Extract key concepts and terms
                concepts = self._extract_key_concepts(clean_content)
                if concepts:
                    section_data['key_concepts'] = concepts
                    
                # Extract named entities and their descriptions
                entities = self._extract_named_entities(clean_content)
                if entities:
                    section_data['named_entities'] = entities
                
                # Add section type classification
                section_data['section_type'] = self._classify_section_type(clean_content)
                
                enhanced_data.append(section_data)
            
            # Store the enhanced data
            self.data_store[data_type].extend(enhanced_data)
            
            # Create rich embeddings for each section
            texts = []
            metadatas = []
            
            for section in enhanced_data:
                # Create rich text representation with better context
                text_parts = [
                    f"Source: {file_name} ({data_type})",
                    f"Section: {section['title']} (Type: {section['section_type']})"
                ]
                
                # Add content with proper context
                if section['section_type'] == 'lore':
                    text_parts.append("World Lore:")
                elif section['section_type'] == 'story':
                    text_parts.append("Story Content:")
                elif section['section_type'] == 'description':
                    text_parts.append("World Description:")
                
                text_parts.append(section['content'])
                
                # Add extracted information
                if 'key_concepts' in section:
                    text_parts.append("Key Concepts: " + ", ".join(section['key_concepts']))
                if 'named_entities' in section:
                    for entity, desc in section['named_entities'].items():
                        text_parts.append(f"{entity}: {desc}")
                
                full_text = "\n\n".join(text_parts)
                texts.append(full_text)
                
                # Enhanced metadata for better retrieval
                metadata = {
                    "source_file": file_name,
                    "data_type": data_type,
                    "section_title": section['title'],
                    "section_type": section['section_type'],
                    "item_id": section['_id'],
                    "key_concepts": section.get('key_concepts', []),
                    "named_entities": list(section.get('named_entities', {}).keys())
                }
                metadatas.append(metadata)
            
            # Update vector store with chunking and rate limiting
            self._update_vector_store_with_rate_limit(texts, metadatas)
            
            logger.info(f"Loaded {len(enhanced_data)} sections from {file_name}")
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            raise

    def _clean_section_content(self, section: str) -> str:
        """Clean and normalize section content."""
        # Remove markdown headers
        lines = []
        for line in section.split('\n'):
            if not line.strip().startswith('#'):
                lines.append(line)
        
        # Clean up whitespace
        content = '\n'.join(lines).strip()
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content

    def _classify_section_type(self, content: str) -> str:
        """Classify the type of section based on its content."""
        content_lower = content.lower()
        
        # Check for story indicators
        if any(word in content_lower for word in ['chapter', 'saga', 'story', 'tale', 'chronicle']):
            return 'story'
        
        # Check for lore indicators
        if any(word in content_lower for word in ['history', 'legend', 'myth', 'ancient', 'legacy', 'world']):
            return 'lore'
        
        # Check for descriptive content
        if any(word in content_lower for word in ['describe', 'detail', 'feature', 'characteristic']):
            return 'description'
        
        return 'general'

    def _extract_named_entities(self, text: str) -> Dict[str, str]:
        """Extract named entities and their descriptions from text."""
        entities = {}
        
        # Look for entity definitions (Entity: description)
        definition_pattern = r'([A-Z][a-zA-Z\s]+):\s*([^.\n]+)'
        for match in re.finditer(definition_pattern, text):
            entity, description = match.groups()
            entities[entity.strip()] = description.strip()
        
        # Look for entities in quotes with nearby context
        quote_pattern = r'"([^"]+)"([^.!?\n]+[.!?])'
        for match in re.finditer(quote_pattern, text):
            entity, context = match.groups()
            if entity[0].isupper():  # Only consider capitalized entities
                entities[entity.strip()] = context.strip()
        
        # Look for capitalized multi-word entities with context
        lines = text.split('\n')
        for line in lines:
            words = line.split()
            i = 0
            while i < len(words):
                if words[i][0].isupper() and len(words[i]) > 1:
                    # Check for multi-word entity
                    entity_words = [words[i]]
                    j = i + 1
                    while j < len(words) and words[j][0].isupper():
                        entity_words.append(words[j])
                        j += 1
                    
                    if len(entity_words) > 1:
                        entity = ' '.join(entity_words)
                        # Get surrounding context
                        start = max(0, i - 5)
                        end = min(len(words), j + 5)
                        context = ' '.join(words[start:end])
                        entities[entity] = context
                    i = j
                else:
                    i += 1
        
        return entities

    def _split_text_into_sections(self, text: str) -> List[str]:
        """Split text into logical sections based on multiple heuristics."""
        sections = []
        
        # Try splitting by markdown headers
        if any(line.startswith('#') for line in text.split('\n')):
            current_section = []
            for line in text.split('\n'):
                if line.startswith('#') and current_section:
                    sections.append('\n'.join(current_section))
                    current_section = [line]
                else:
                    current_section.append(line)
            if current_section:
                sections.append('\n'.join(current_section))
                
        # Try splitting by double newlines if no markdown headers
        elif '\n\n' in text:
            potential_sections = text.split('\n\n')
            for section in potential_sections:
                if len(section.strip()) > 50:  # Only keep substantial sections
                    sections.append(section.strip())
                    
        # Fallback to single chunk if no clear sections
        else:
            sections = [text]
            
        return sections

    def _extract_section_title(self, section: str) -> Optional[str]:
        """Extract a meaningful title from a section of text."""
        lines = section.split('\n')
        
        # Check for markdown headers
        for line in lines:
            if line.startswith('#'):
                return line.lstrip('#').strip()
        
        # Check for underlined headers
        if len(lines) > 1 and any(all(c in '=-' for c in lines[1].strip()) for c in lines[1]):
            return lines[0].strip()
        
        # Extract first sentence if short enough
        first_line = lines[0].strip()
        if len(first_line) < 100:
            return first_line
            
        return None

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts and terms from text."""
        concepts = set()
        
        # Look for capitalized terms (potential proper nouns)
        words = text.split()
        for i, word in enumerate(words):
            if word[0].isupper() and len(word) > 1:
                # Check for multi-word proper nouns
                phrase = [word]
                j = i + 1
                while j < len(words) and words[j][0].isupper():
                    phrase.append(words[j])
                    j += 1
                if len(phrase) > 1:
                    concepts.add(' '.join(phrase))
                else:
                    concepts.add(word)
        
        # Look for terms in quotes
        quotes = re.findall(r'"([^"]*)"', text)
        concepts.update(quotes)
        
        return list(concepts)

    def analyze_story_submission(self, story_text: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
        """
        Analyze a story submission for narrative quality and lore consistency.
        
        Args:
            story_text: The story text to analyze
            chat_history: Optional chat history for context
            
        Returns:
            Dict containing analysis results and feedback
        """
        try:
            # Perform narrative analysis
            narrative_analysis = self.validator.analyze_narrative_structure(story_text)
            
            # Get relevant lore context
            lore_context = self._get_relevant_lore_context(story_text)
            
            # Format narrative elements for the prompt
            narrative_elements = self._format_narrative_elements(narrative_analysis)
            
            # Generate analysis using LLM
            analysis_context = {
                "analysis_results": json.dumps(narrative_analysis, indent=2),
                "narrative_elements": narrative_elements,
                "lore_context": lore_context,
                "chat_history": chat_history or []
            }
            
            response = self.primary_llm.invoke(
                self.narrative_analysis_prompt.format(**analysis_context)
            )
            
            return {
                "analysis": response.content,
                "narrative_structure": narrative_analysis,
                "lore_alignment": self._check_lore_alignment(story_text, lore_context)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing story submission: {str(e)}")
            raise

    def _format_narrative_elements(self, analysis: Dict[str, Any]) -> str:
        """Format narrative elements for prompt display."""
        sections = []
        
        # Format characters
        if analysis["story_elements"]["characters"]:
            char_lines = ["Characters:"]
            for char in analysis["story_elements"]["characters"]:
                char_lines.append(f"- {char['name']} ({char['role']})")
            sections.append("\n".join(char_lines))
        
        # Format settings
        if analysis["story_elements"]["settings"]:
            setting_lines = ["Settings:"]
            for setting in analysis["story_elements"]["settings"]:
                attrs = setting["attributes"]
                setting_lines.append(
                    f"- {setting['name']} ({setting['type']})"
                    f"\n  Atmosphere: {', '.join(attrs['atmosphere'])}"
                    f"\n  Elements: {', '.join(attrs['elements'])}"
                )
            sections.append("\n".join(setting_lines))
        
        # Format plot points
        if analysis["story_elements"]["plot_points"]:
            plot_lines = ["Major Plot Points:"]
            for point in analysis["story_elements"]["plot_points"]:
                plot_lines.append(
                    f"- Type: {point['type']}"
                    f"\n  Significance: {', '.join(point['significance'])}"
                )
            sections.append("\n".join(plot_lines))
        
        # Format themes
        if analysis["story_elements"]["themes"]:
            theme_lines = ["Themes:"]
            for theme in analysis["story_elements"]["themes"]:
                theme_lines.append(f"- {theme['name']} (mentioned {theme['frequency']} times)")
            sections.append("\n".join(theme_lines))
        
        return "\n\n".join(sections)

    def _get_relevant_lore_context(self, story_text: str) -> str:
        """Get relevant lore context for the story."""
        # Extract key terms for search
        key_terms = set()
        
        # Add character names
        char_pattern = r'([A-Z][a-zA-Z\s]+)(?:\s+(?:is|was|appeared|stood|walked|said))'
        key_terms.update(match.group(1) for match in re.finditer(char_pattern, story_text))
        
        # Add location names
        loc_pattern = r'(?:in|at|near|through)\s+(?:the\s+)?([A-Z][a-zA-Z\s]+)'
        key_terms.update(match.group(1) for match in re.finditer(loc_pattern, story_text))
        
        # Search for relevant lore
        relevant_docs = []
        for term in key_terms:
            results = self.validator.search_knowledge_base(term, k=2)
            relevant_docs.extend(results)
        
        # Format context
        if relevant_docs:
            context_parts = []
            for doc in relevant_docs:
                if hasattr(doc, 'page_content'):
                    context_parts.append(doc.page_content)
                elif isinstance(doc, dict):
                    context_parts.append(doc.get('content', ''))
            
            return "\n\n".join(context_parts)
        
        return "No directly relevant lore found."

    def _check_lore_alignment(self, story_text: str, lore_context: str) -> Dict[str, Any]:
        """Check story alignment with existing lore."""
        alignment = {
            "is_aligned": True,
            "conflicts": [],
            "enhancements": []
        }
        
        # Check for element-ability conflicts
        element_check = self.validator._is_element_ability_valid(story_text)
        if not element_check[0]:
            alignment["is_aligned"] = False
            alignment["conflicts"].append({
                "type": "element_ability",
                "details": element_check[1]
            })
        
        # Check for location-element conflicts
        for element, invalid_locs in self.validator.invalid_locations.items():
            if any(loc.lower() in story_text.lower() for loc in invalid_locs):
                if any(kw.lower() in story_text.lower() 
                      for kw in self.validator.valid_combinations[element]):
                    alignment["is_aligned"] = False
                    alignment["conflicts"].append({
                        "type": "location_element",
                        "details": f"Invalid {element} element usage in {invalid_locs}"
                    })
        
        # Look for opportunities to enhance lore connections
        if lore_context != "No directly relevant lore found.":
            # Find mentioned but unexplained elements
            story_terms = set(re.findall(r'\b[A-Z][a-zA-Z]+\b', story_text))
            lore_terms = set(re.findall(r'\b[A-Z][a-zA-Z]+\b', lore_context))
            
            unexplained = story_terms - lore_terms
            if unexplained:
                alignment["enhancements"].append({
                    "type": "expand_lore",
                    "elements": list(unexplained),
                    "suggestion": "Consider expanding on these elements' connection to existing lore"
                })
        
        return alignment

    def _format_chat_history(self, chat_history: Optional[List[str]] = None) -> str:
        """Format the chat history for context."""
        if not chat_history:
            return "No previous conversation context."
        
        formatted_history = []
        for i, message in enumerate(chat_history):
            if i % 2 == 0:
                formatted_history.append(f"User: {message}")
            else:
                formatted_history.append(f"Assistant: {message}")
            
        return "\n".join(formatted_history)

    def _get_relevant_context(self, query: str, filters: Optional[Dict[str, Any]] = None) -> str:
        """Get relevant context for the query using the validator's knowledge base."""
        try:
            # Initialize empty context
            context_parts = []
            
            # Get relevant documents from vector store
            if self.validator and self.validator.vector_store:
                # Search with query
                results = self.validator.vector_store.similarity_search(
                    query,
                    k=5,  # Get top 5 relevant documents
                    filter=filters
                )
                
                # Extract and format content from results
                for doc in results:
                    if hasattr(doc, 'page_content'):
                        context_parts.append(doc.page_content)
                    elif isinstance(doc, dict):
                        context_parts.append(doc.get('content', ''))
            
            # If no context found through vector store, try data store
            if not context_parts and filters:
                for data_type, type_filters in filters.items():
                    matching_entities = self.get_specific_entities(data_type, type_filters)
                    for entity in matching_entities[:3]:  # Limit to top 3 matches
                        if isinstance(entity, dict):
                            # Format entity details
                            context_parts.append(self._format_entity_details([entity]))
            
            # Return formatted context or default message
            if context_parts:
                return "\n\n".join(context_parts)
            return "No directly relevant context found."
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {str(e)}")
            return "Error retrieving context." 