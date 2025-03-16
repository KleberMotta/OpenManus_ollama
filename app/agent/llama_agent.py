from typing import Any, List, Optional
from llama_cpp import Llama
from app.config.llama_config import llama_config

class LlamaLocalAgent:
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        self.config = llama_config
        
        if model_path:
            self.config.model_path = model_path
        if temperature is not None:
            self.config.temperature = temperature
        if max_tokens is not None:
            self.config.max_tokens = max_tokens
        
        self.config.validate_model()
        
        self.model = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.max_tokens,
            verbose=False
        )
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        full_prompt = (system_prompt + "\n" if system_prompt else "") + prompt
        
        response = self.model(
            full_prompt, 
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=self.config.temperature,
            stop=["Human:", "Assistant:"]
        )
        
        return response.get('choices', [{}])[0].get('text', '')
