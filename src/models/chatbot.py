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
from .lore_validator import LoreValidator
import logging

logger = logging.getLogger(__name__)

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
            - Always mention specific items, locations, and creatures from the knowledge base when relevant
            - If suggesting connections between elements, explain the reasoning
            - For new submissions, check carefully against existing canon
            - Highlight any potential conflicts with established lore
            - Provide constructive feedback for improving submissions
            - When discussing items or equipment, reference their exact names from the knowledge base

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