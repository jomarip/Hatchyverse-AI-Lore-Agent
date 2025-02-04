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
    
    async def ainvoke(self, input: str, **kwargs) -> List[Document]:
        """Async invoke implementation."""
        return await self._aget_relevant_documents(input)
    
    def invoke(self, input: str, **kwargs) -> List[Document]:
        """Enhanced invoke implementation with narrative context."""
        try:
            # Detect query type keywords
            character_keywords = ["character", "who", "person", "npc", "protagonist", "antagonist"]
            timeline_keywords = ["when", "timeline", "history", "event", "happened", "occurred"]
            story_keywords = ["story", "plot", "narrative", "tale", "saga", "chronicle"]
            
            query_lower = input.lower()
            is_character_query = any(kw in query_lower for kw in character_keywords)
            is_timeline_query = any(kw in query_lower for kw in timeline_keywords)
            is_story_query = any(kw in query_lower for kw in story_keywords)
            
            all_docs = []
            seen_contents = set()
            
            try:
                # Get character-specific results
                if is_character_query:
                    relevant_chars = self._get_relevant_characters(input)
                    for char in relevant_chars:
                        doc_content = (
                            f"Character: {char['name']}\n"
                            f"Type: {char.get('type', 'Unknown')}\n"
                            f"Affiliations: {', '.join(char.get('affiliations', []))}\n"
                            f"Abilities: {', '.join(char.get('abilities', []))}\n"
                        )
                        if doc_content not in seen_contents:
                            all_docs.append(Document(
                                page_content=doc_content,
                                metadata={
                                    "type": "Character",
                                    "name": char["name"],
                                    **{k: v for k, v in char.items() if k not in ["name"]}
                                }
                            ))
                            seen_contents.add(doc_content)
                
                # Get timeline-specific results
                if is_timeline_query:
                    relevant_events = self._get_relevant_timeline_events(input)
                    for event in relevant_events:
                        doc_content = (
                            f"Event: {event['name']}\n"
                            f"When: {event['timestamp']}\n"
                            f"Location: {event['location']}\n"
                            f"Description: {event['description']}\n"
                        )
                        if doc_content not in seen_contents:
                            all_docs.append(Document(
                                page_content=doc_content,
                                metadata={
                                    "type": "Event",
                                    "name": event["name"],
                                    **{k: v for k, v in event.items() if k not in ["name", "description"]}
                                }
                            ))
                            seen_contents.add(doc_content)
                
                # Get story-specific results
                if is_story_query:
                    relevant_segments = self._get_relevant_story_segments(input)
                    for segment in relevant_segments:
                        doc_content = (
                            f"Story: {segment['title']}\n"
                            f"Arc: {segment['arc']}\n"
                            f"Characters: {', '.join(segment.get('characters', []))}\n"
                            f"Content: {segment['content']}\n"
                        )
                        if doc_content not in seen_contents:
                            all_docs.append(Document(
                                page_content=doc_content,
                                metadata={
                                    "type": "Story",
                                    "title": segment["title"],
                                    **{k: v for k, v in segment.items() if k not in ["title", "content"]}
                                }
                            ))
                            seen_contents.add(doc_content)
                
                # Get item-specific results (existing functionality)
                item_docs = self._get_relevant_items(input)
                for item in item_docs:
                    doc_content = (
                        f"Name: {item['name']}\n"
                        f"Type: {item.get('type', 'Unknown')}\n"
                        f"Element: {item.get('element', 'None')}\n"
                        f"Description: {item.get('description', '')}\n"
                        f"Effects: {', '.join(item.get('effects', []))}\n"
                    )
                    if doc_content not in seen_contents:
                        all_docs.append(Document(
                            page_content=doc_content,
                            metadata={
                                "type": "Item",
                                "name": item["name"],
                                **{k: v for k, v in item.items() if k not in ["name", "description"]}
                            }
                        ))
                        seen_contents.add(doc_content)
                
                # If we have results, return them
                if all_docs:
                    return all_docs[:4]  # Limit to top 4 results
                
            except Exception as e:
                logger.warning(f"Error in specialized search, falling back to general search: {e}")
            
            # Fallback to base retriever
            return self.base_retriever.invoke(input)
            
        except Exception as e:
            logger.error(f"Error in FilteredRetriever: {e}")
            return self.base_retriever.invoke(input)

class LoreChatbot:
    """Manages conversations and lore interactions with users."""
    
    def __init__(
        self,
        validator: LoreValidator,
        model_provider: str = "openai",
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7
    ):
        self.validator = validator
        self.model_provider = model_provider
        self.primary_model_name = model_name
        self.fallback_model_name = os.getenv("OPENAI_FALLBACK_MODEL")
        self.temperature = temperature
        
        # Initialize the appropriate LLM based on provider
        if model_provider == "openai":
            self.primary_llm = self._create_llm(self.primary_model_name)
            if self.fallback_model_name:
                self.fallback_llm = self._create_llm(self.fallback_model_name)
            else:
                self.fallback_llm = None
        elif model_provider == "anthropic":
            self.primary_llm = ChatAnthropic(
                model_name=model_name or "claude-3-sonnet-20240229",
                temperature=temperature
            )
            self.fallback_llm = None
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
        
        # Define the base prompt template
        self.base_prompt = ChatPromptTemplate.from_template(
            """You are a knowledgeable Hatchyverse Lorekeeper. Your role is to help users explore and 
            understand the Hatchyverse world while ensuring all new contributions maintain consistency 
            with existing lore.

            Use this context from the knowledge base:
            {context}

            Current conversation history:
            {chat_history}

            User's message: {question}

            Remember:
            - Be friendly and enthusiastic about Hatchyverse lore
            - ALWAYS mention specific items, locations, and creatures from the knowledge base by their EXACT names
            - When discussing items or equipment:
              * Start by listing ALL relevant items with their EXACT names from the knowledge base
              * For fire-type items, ALWAYS mention Flame Essence, flame gems, and fire essence crystals
              * For water-type items, ALWAYS mention Frost Crystal and ice shard
              * For dark-type items, ALWAYS mention Shadow Essence and dark crystal
              * Describe each item's abilities, powers, and attack capabilities
              * Explain how these items enhance combat and abilities
            - When discussing elements or types:
              * Describe their unique abilities and attack styles
              * Mention specific locations where they train or gather
              * Include details about their powers and combat capabilities
            - NEVER make up names - only use items/locations/creatures explicitly mentioned in the knowledge base
            - Include specific details about abilities, attacks, and powers
            - Be comprehensive in covering all relevant aspects (items, abilities, locations)
            - Use exact terminology from the knowledge base

            Format your response with clear sections:
            1. Direct answer to the query
            2. Available items/equipment (if relevant)
            3. Related abilities and powers
            4. Important locations
            5. Additional relevant details

            Assistant:"""
        )
        
        # Initialize the QA chain with primary model
        self.qa_chain = self._create_qa_chain(self.primary_llm)
        
    def _create_llm(self, model_name: str) -> ChatOpenAI:
        """Creates a ChatOpenAI instance with retry logic for rate limits."""
        @retry(
            stop=stop_after_attempt(int(os.getenv("MAX_RETRIES", 3))),
            wait=wait_exponential(
                multiplier=float(os.getenv("INITIAL_RETRY_DELAY", 1)),
                max=float(os.getenv("MAX_RETRY_DELAY", 60))
            )
        )
        def create_with_retry():
            return ChatOpenAI(
                model_name=model_name,
                temperature=self.temperature
            )
        return create_with_retry()
    
    def _create_qa_chain(self, llm: Any) -> ConversationalRetrievalChain:
        """Creates a ConversationalRetrievalChain with the specified LLM."""
        try:
            # Create the filtered retriever
            base_retriever = self.validator.vector_store.as_retriever()
            filtered_retriever = FilteredRetriever(
                base_retriever=base_retriever,
                vector_store=self.validator.vector_store
            )
            
            return ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=filtered_retriever,
                combine_docs_chain_kwargs={"prompt": self.base_prompt}
            )
        except Exception as e:
            logger.error(f"Error creating QA chain: {e}")
            # Fallback to simple retriever if filtered retriever fails
            return ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.validator.vector_store.as_retriever(),
                combine_docs_chain_kwargs={"prompt": self.base_prompt}
            )
    
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
                self.qa_chain = self._create_qa_chain(self.fallback_llm)
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
            
    def generate_response(
        self,
        user_input: str,
        chat_history: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generates a response to user input, handling both queries and submissions.
        
        Args:
            user_input: The user's message
            chat_history: Optional list of previous messages
            
        Returns:
            Dict containing response and any validation results
        """
        try:
            # Handle lore submissions (prefixed with [SUBMIT])
            if user_input.startswith("[SUBMIT]"):
                submission_content = user_input[8:].strip()  # Remove [SUBMIT] prefix
                return self._process_with_fallback(
                    self._process_lore_submission,
                    submission_content
                )
            
            # Handle normal queries
            return self._process_with_fallback(
                lambda: {
                    "response": self.qa_chain.invoke({
                        "question": user_input,
                        "chat_history": chat_history or []
                    })["answer"],
                    "validation": None
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise 