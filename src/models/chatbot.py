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
    """Custom retriever that filters and prioritizes items based on query type."""
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    base_retriever: BaseRetriever = Field(description="Base retriever to filter and enhance")
    vector_store: Any = Field(description="Vector store for additional filtering")
    item_store: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Dynamic item store containing categorized items and their metadata"
    )
    
    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_item_store()
    
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
        """Synchronous invoke implementation."""
        try:
            # Check if this is an item-related query
            item_keywords = ["item", "items", "equipment", "available", "use", "using"]
            ability_keywords = ["attack", "power", "ability", "abilities", "skill", "skills"]
            element_keywords = {
                "fire": ["fire", "flame", "burning", "heat", "volcanic"],
                "water": ["water", "aqua", "ice", "frost", "crystal"],
                "dark": ["dark", "shadow", "night", "mysterious"]
            }
            
            query_lower = input.lower()
            is_item_query = any(kw in query_lower for kw in item_keywords)
            is_ability_query = any(kw in query_lower for kw in ability_keywords)
            
            # Detect element type from query
            element_type = None
            for element, keywords in element_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    element_type = element
                    break
            
            all_docs = []
            seen_contents = set()
            search_kwargs = {"k": 5}
            
            try:
                # Get item-specific results if it's an item query
                if is_item_query:
                    # Get relevant items based on query and element type
                    filters = {"element": element_type} if element_type else None
                    relevant_items = self._get_relevant_items(input, filters)
                    
                    # Convert items to documents
                    for item in relevant_items:
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
                
                # Get ability-related results if it's an ability query or item query
                if is_ability_query or is_item_query:
                    ability_query = f"{input} abilities attacks powers"
                    if element_type:
                        ability_query = f"{ability_query} {element_type} element"
                    
                    ability_docs = self.vector_store.similarity_search(
                        ability_query,
                        **search_kwargs
                    )
                    
                    # Add unique ability documents
                    for doc in ability_docs:
                        if doc.page_content not in seen_contents:
                            all_docs.append(doc)
                            seen_contents.add(doc.page_content)
                
                # Get element-specific results without filtering
                if element_type:
                    element_query = f"{input} {element_type}"
                    element_docs = self.vector_store.similarity_search(
                        element_query,
                        **search_kwargs
                    )
                    
                    # Add unique element documents
                    for doc in element_docs:
                        if doc.page_content not in seen_contents:
                            all_docs.append(doc)
                            seen_contents.add(doc.page_content)
                
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