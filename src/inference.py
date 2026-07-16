import os
import httpx
from typing import Type, TypeVar
from pydantic import BaseModel

# Create a generic TypeVar bound to Pydantic's BaseModel
T = TypeVar('T', bound=BaseModel)

class UniversalInferenceEngine:
    def __init__(self):
        # Read configurations from the scratchpad's global environment setup
        self.backend_type = os.getenv("SCRATCHPAD_LLM_BACKEND", "ollama").lower()
        self.model_name = os.getenv("SCRATCHPAD_MODEL_NAME", "qwen2.5:14b")
        self.llamacpp_url = os.getenv("LLAMACPP_API_URL", "http://localhost:8080/completion")

    def generate_structured(self, prompt: str, system_prompt: str, response_schema: Type[T]) -> T:
        """
        Executes structural inference against the configured local backend, 
        guaranteeing the return of a fully validated Pydantic model.
        """
        if self.backend_type == "ollama":
            return self._execute_ollama(prompt, system_prompt, response_schema)
        elif self.backend_type == "llamacpp":
            return self._execute_llamacpp(prompt, system_prompt, response_schema)
        elif self.backend_type == "lmstudio":
            return self._execute_lmstudio(prompt, system_prompt, response_schema)
        elif self.backend_type == "transformers":
            return self._execute_transformers(prompt, system_prompt, response_schema)
        else:
            raise ValueError(f"Unsupported inference backend: {self.backend_type}")

    def _execute_ollama(self, prompt: str, system_prompt: str, response_schema: Type[T]) -> T:
        import ollama # Lazy-loaded to keep dependencies light if not used
        
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            # Ollama natively compiles the schema to grammar rules
            format=response_schema.model_json_schema(),
            options={"temperature": 0.1}
        )
        return response_schema.model_validate_json(response['message']['content'])

    def _execute_llamacpp(self, prompt: str, system_prompt: str, response_schema: Type[T]) -> T:
        # Construct the unified instruction prompt for llama.cpp format alignment
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        payload = {
            "prompt": formatted_prompt,
            "temperature": 0.1,
            # llama.cpp server uses token masking based directly on the passed JSON schema
            "json_schema": response_schema.model_json_schema()
        }
        
        response = httpx.post(self.llamacpp_url, json=payload, timeout=90.0)
        response.raise_for_status()
        
        raw_json_string = response.json()["content"]
        return response_schema.model_validate_json(raw_json_string)

    def _execute_lmstudio(self, prompt: str, system_prompt: str, response_schema: Type[T]) -> T:
        """
        LM Studio with OpenAI-compatible /v1/chat/completions endpoint.
        Uses structured output via response_format + json_schema (same as OpenAI SDK).
        """
        lmstudio_base = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1")
        lmstudio_model = os.getenv("SCRATCHPAD_MODEL_NAME", self.model_name)

        # Wrap raw Pydantic schema with the name + strict fields that
        # OpenAI-compatible endpoints require at the top level.
        raw_schema = response_schema.model_json_schema()
        schema_name = raw_schema.get("title", response_schema.__name__)
        wrapped_schema = {
            "name": schema_name,
            "strict": True,
            "schema": raw_schema
        }

        payload = {
            "model": lmstudio_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
            "response_format": {
                "type": "json_schema",
                "json_schema": wrapped_schema
            }
        }

        response = httpx.post(f"{lmstudio_base}/chat/completions", json=payload, timeout=120.0)
        response.raise_for_status()
        data = response.json()

        # LM Studio returns OpenAI-compatible response shape
        raw_json_string = data["choices"][0]["message"]["content"]
        return response_schema.model_validate_json(raw_json_string)

    def _execute_transformers(self, prompt: str, system_prompt: str, response_schema: Type[T]) -> T:
        # Handled inside the package using Instructor patched transformers
        global _patched_hf_client
        if '_patched_hf_client' not in globals():
            import torch
            import instructor
            from transformers import pipeline
            
            hf_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            _patched_hf_client = instructor.from_hf(hf_pipeline)
            
        formatted_prompt = f"System: {system_prompt}\nUser: {prompt}"
        
        return _patched_hf_client.chat.completions.create(
            model="", 
            response_model=response_schema,
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=2048
        )
