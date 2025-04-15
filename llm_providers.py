"""
LLM Provider Management Module

This module manages the different LLM providers (OpenAI and Ollama) and provides
a unified interface for using them in the application.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Constants for model providers
OPENAI = "openai"
OLLAMA = "ollama"

# Default models for each provider
DEFAULT_MODELS = {
    OPENAI: "gpt-4o",
    OLLAMA: "llama3"  # Change this to the model you have available in Ollama
}

# Default URLs for providers
DEFAULT_URLS = {
    OPENAI: None,  # OpenAI uses the API key, no base URL needed
    OLLAMA: "http://localhost:11434"  # Default Ollama local URL
}

class LLMManager:
    """
    Manager class for handling different LLM providers.
    """
    
    def __init__(self):
        """
        Initialize the LLM manager with available providers.
        """
        self.current_provider = OPENAI  # Default provider
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        # Check if API key is available
        self.openai_available = bool(self.openai_api_key)
        
        # Will be checked when needed
        self.ollama_available = None
        self.ollama_url = DEFAULT_URLS[OLLAMA]
        
        # Default models
        self.models = {
            OPENAI: DEFAULT_MODELS[OPENAI],
            OLLAMA: DEFAULT_MODELS[OLLAMA]
        }
    
    def set_provider(self, provider):
        """
        Set the current LLM provider.
        
        Args:
            provider (str): Provider name ('openai' or 'ollama')
        
        Returns:
            bool: True if provider was set successfully, False otherwise
        """
        if provider not in [OPENAI, OLLAMA]:
            print(f"Error: Unknown provider '{provider}'")
            return False
        
        # Check if the provider is available
        if provider == OPENAI and not self.openai_available:
            print("Error: OpenAI API key is not available")
            return False
        
        if provider == OLLAMA:
            # Check Ollama connection if not already checked
            if self.ollama_available is None:
                self.check_ollama_connection()
            
            if not self.ollama_available:
                print("Error: Ollama is not available or not running")
                return False
        
        self.current_provider = provider
        print(f"LLM provider set to: {provider}")
        return True
    
    def check_ollama_connection(self):
        """
        Check if Ollama is running and available.
        
        Returns:
            bool: True if Ollama is available, False otherwise
        """
        try:
            # Try to create a simple Ollama client to check connection
            client = ChatOllama(base_url=self.ollama_url, model=self.models[OLLAMA])
            # Simple test to check if Ollama is running
            test_response = client.invoke("Hello")
            self.ollama_available = True
            print(f"Ollama connection successful. Using model: {self.models[OLLAMA]}")
            return True
        except Exception as e:
            print(f"Ollama connection failed: {str(e)}")
            self.ollama_available = False
            return False
    
    def set_model(self, provider, model_name):
        """
        Set the model for a specific provider.
        
        Args:
            provider (str): Provider name
            model_name (str): Model name
        
        Returns:
            bool: True if model was set successfully, False otherwise
        """
        if provider not in [OPENAI, OLLAMA]:
            print(f"Error: Unknown provider '{provider}'")
            return False
        
        self.models[provider] = model_name
        print(f"Model for {provider} set to: {model_name}")
        return True
    
    def set_ollama_url(self, url):
        """
        Set the Ollama base URL.
        
        Args:
            url (str): Base URL for Ollama API
        
        Returns:
            bool: True if URL was set successfully, False otherwise
        """
        self.ollama_url = url
        self.ollama_available = None  # Reset connection status to trigger a new check
        print(f"Ollama URL set to: {url}")
        return True
    
    def get_chat_model(self, temperature=0.0):
        """
        Get a LangChain chat model based on the current provider.
        
        Args:
            temperature (float): Temperature for the model
        
        Returns:
            LangChain model: A model that can be used in LangChain
        """
        if self.current_provider == OPENAI:
            return ChatOpenAI(
                api_key=self.openai_api_key,
                model=self.models[OPENAI],
                temperature=temperature
            )
        elif self.current_provider == OLLAMA:
            return ChatOllama(
                base_url=self.ollama_url,
                model=self.models[OLLAMA],
                temperature=temperature
            )
        else:
            # Fallback to OpenAI if available
            if self.openai_available:
                print(f"Unknown provider '{self.current_provider}', falling back to OpenAI")
                return ChatOpenAI(
                    api_key=self.openai_api_key,
                    model=self.models[OPENAI],
                    temperature=temperature
                )
            else:
                raise ValueError(f"No available LLM provider")
    
    def get_available_providers(self):
        """
        Get a list of available LLM providers.
        
        Returns:
            list: List of available provider names
        """
        available = []
        
        if self.openai_available:
            available.append(OPENAI)
        
        # Check Ollama if not already checked
        if self.ollama_available is None:
            self.check_ollama_connection()
        
        if self.ollama_available:
            available.append(OLLAMA)
        
        return available
    
    def get_available_models(self, provider):
        """
        Get available models for a specific provider.
        This is a simplified version that returns predefined models.
        
        Args:
            provider (str): Provider name
        
        Returns:
            list: List of available model names
        """
        if provider == OPENAI:
            return ["gpt-4o", "gpt-3.5-turbo"]
        elif provider == OLLAMA:
            # In a real implementation, you might query Ollama for available models
            # For simplicity, return some common Ollama models
            return ["llama3", "llama2", "mistral", "gemma", "codellama"]
        else:
            return []
    
    def get_current_provider(self):
        """
        Get the current LLM provider.
        
        Returns:
            str: Current provider name
        """
        return self.current_provider
    
    def get_current_model(self):
        """
        Get the current model for the active provider.
        
        Returns:
            str: Current model name
        """
        return self.models[self.current_provider]

# Create a singleton instance for the application
llm_manager = LLMManager()