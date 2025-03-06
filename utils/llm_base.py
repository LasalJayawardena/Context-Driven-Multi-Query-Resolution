# Files contains logic for LLM Models

## Import Libraries
import os
import requests
from abc import ABC, abstractmethod
from typing import *
from functools import lru_cache
import tiktoken
from openai import OpenAI


## Main Logic 

class BaseAIModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BaseAIModel with a configuration dictionary.

        :param config: Configuration dictionary for the model.
        """
        self.config = config
        self._validate_config()
        self.tokenizer = self.get_tokenizer()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate the configuration dictionary.
        Must be implemented by subclasses to enforce required keys in the config.
        """
        pass

    @abstractmethod
    def get_tokenizer(self) -> Any:
        """
        Get the tokenizer.
        Should be implemented by subclasses to return the appropriate tokenizer.
        """
        pass

    @abstractmethod
    def inference(self, prompt: str, args: Dict[str, Any]) -> str:
        """
        Generate a response from the model based on the prompt and args.
        Must be implemented by subclasses.

        :param prompt: The input prompt for the model.
        :param args: Additional arguments for the model's inference method.
        :return: The generated text from the model.
        """
        pass

    def get_num_tokens(self, text: str) -> int:
        """
        Get the number of tokens in the text using the tokenizer.

        :param text: The input text to tokenize.
        :return: The number of tokens in the text.
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        raise NotImplementedError("Tokenization is not implemented for this model type.")

    def get_token_ids(self, text: str) -> List[int]:
        """
        Return the ordered ids of the tokens in a text using the tokenizer.

        :param text: The input text to tokenize.
        :return: The list of token ids in the text.
        """
        if self.tokenizer:
            return self.tokenizer.encode(text)
        raise NotImplementedError("Tokenization is not implemented for this model type.")

class LocalAIModel(BaseAIModel):
    def _validate_config(self) -> None:
        """
        Validate the configuration dictionary for LocalAIModel.
        Ensures 'model_path' is present in the config.
        """
        required_keys = ["model_path"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def get_tokenizer(self) -> tiktoken.Encoding:
        """
        Get the tokenizer for the LocalAIModel.
        Loads the tokenizer from the specified model path in the config.

        :return: An instance of tiktoken.Encoding.
        """
        try:
            return tiktoken.get_encoding(self.config["model_path"])
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")

    def inference(self, prompt: str, args: Dict[str, Any]) -> str:
        """
        Perform inference using the local model.

        :param prompt: The input prompt for the model.
        :param args: Additional arguments for the model's generate method.
        :return: The generated text from the model.
        """
        if not prompt:
            raise ValueError("Prompt is required for inference.")
        
        model = self.config.get("model")
        if model is None:
            raise ValueError("Local model not provided in config.")

        inputs = self.tokenizer.encode(prompt)
        outputs = model.generate(inputs, **args)
        return self.tokenizer.decode(outputs[0])

class APIAIModel(BaseAIModel):
    def _validate_config(self) -> None:
        """
        Validate the configuration dictionary for APIAIModel.
        Ensures 'api_url' and 'api_key' are present in the config.
        """
        required_keys = ["api_url", "api_key"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def get_tokenizer(self) -> None:
        """
        Get the tokenizer for the APIAIModel.
        API models do not require a tokenizer, so this returns None.

        :return: None
        """
        return None

    def inference(self, prompt: str, args: Dict[str, Any]) -> str:
        """
        Perform inference using the API model.

        :param prompt: The input prompt for the model.
        :param args: Additional arguments for the API request.
        :return: The generated text from the API response.
        """
        if not prompt:
            raise ValueError("Prompt is required for inference.")

        response = requests.post(
            self.config["api_url"],
            headers={"Authorization": f"Bearer {self.config['api_key']}"},
            json={"prompt": prompt, **args}
        )

        if response.status_code != 200:
            raise RuntimeError(f"API request failed with status code {response.status_code}: {response.text}")

        return response.json()["generated_text"]


class OpenAI_Model(BaseAIModel):
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the OpenAI_Model with a configuration dictionary.
        Set default values for optional parameters.

        :param config: Configuration dictionary for the model.
        """
        self.model: str = config.get("model", "gpt-3.5-turbo")
        self.api_url: str = config.get("api_url", "https://api.openai.com/v1/chat/completions")
        self.api_key: str = config.get("api_key", os.getenv("OPENAI_API_KEY"))
        if not self.api_key:
            raise ValueError("API key is required.")
        self.temperature: float = config.get("temperature", 1.0)
        self.max_tokens: int = config.get("max_tokens", 100)
        self.n: int = config.get("n", 1)
        self.stop: Optional[List[str]] = config.get("stop", None)
        self.frequency_penalty: float = config.get("frequency_penalty", 0.0)
        self.presence_penalty: float = config.get("presence_penalty", 0.0)
        
        super().__init__(config)

    def _validate_config(self) -> None:
        """
        Validate the configuration dictionary for OpenAI_Model.
        Ensures required keys are present in the config and checks the validity of parameter values.
        """
        required_keys = ["api_url", "api_key", "model"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        if not (0 <= self.temperature <= 2):
            raise ValueError("Temperature must be between 0 and 2.")
        if not (0 <= self.frequency_penalty <= 2):
            raise ValueError("Frequency penalty must be between -2 and 2.")
        if not (0 <= self.presence_penalty <= 2):
            raise ValueError("Presence penalty must be between -2 and 2.")
        if not (0 <= self.max_tokens <= 4096):  # Example max token limit for GPT-3.5-turbo
            raise ValueError("Max tokens must be between 0 and the model's context length limit.")

    def get_tokenizer(self) -> tiktoken.Encoding:
        """
        Get the tokenizer for the OpenAI_Model.
        Uses tiktoken to get the tokenizer corresponding to the specified model.

        :return: An instance of tiktoken.Encoding.
        """
        try:
            return tiktoken.encoding_for_model(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to load tokenizer: {e}")

    @lru_cache(maxsize=32)
    def inference(self, prompt: str, args: Dict[str, Any] = {}, json_mode: bool = False) -> str:
        """
        Perform inference using the GPT-3.5 model with caching.

        :param prompt: The input prompt for the model.
        :param args: Additional arguments for the API request.
        :param json_mode: If true, requests a JSON-formatted response.
        :return: The generated text from the API response.
        """
        if not prompt:
            raise ValueError("Prompt is required for inference.")

        request_data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.get("temperature", self.temperature),
            "max_tokens": args.get("max_tokens", self.max_tokens),
            "n": args.get("n", self.n),
            "stop": args.get("stop", self.stop),
            "frequency_penalty": args.get("frequency_penalty", self.frequency_penalty),
            "presence_penalty": args.get("presence_penalty", self.presence_penalty)
        }

        if json_mode:
            request_data["response_format"] = {"type": "json_object"}

        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=request_data
        )

        if response.status_code != 200:
            raise RuntimeError(f"API request failed with status code {response.status_code}: {response.text}")

        return response.json()["choices"][0]["message"]["content"]

    @lru_cache(maxsize=32)
    def get_full_response(self, prompt: str, args: Dict[str, Any] = {}, json_mode: bool = False) -> Dict[str, Any]:
        """
        Perform inference using the GPT-3.5 model and return the full response.

        :param prompt: The input prompt for the model.
        :param args: Additional arguments for the API request.
        :param json_mode: If true, requests a JSON-formatted response.
        :return: The full response from the API.
        """
        if not prompt:
            raise ValueError("Prompt is required for inference.")

        request_data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.get("temperature", self.temperature),
            "max_tokens": args.get("max_tokens", self.max_tokens),
            "n": args.get("n", self.n),
            "stop": args.get("stop", self.stop),
            "frequency_penalty": args.get("frequency_penalty", self.frequency_penalty),
            "presence_penalty": args.get("presence_penalty", self.presence_penalty)
        }

        if json_mode:
            request_data["response_format"] = {"type": "json_object"}

        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=request_data
        )

        if response.status_code != 200:
            raise RuntimeError(f"API request failed with status code {response.status_code}: {response.text}")

        return response.json()

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update the configuration parameters and revalidate them.

        :param new_config: Dictionary containing the new configuration parameters.
        """
        self.config.update(new_config)
        self._validate_config()

    def validate_prompt(self, prompt: List[Dict[str, str]]) -> None:
        """
        Validate the format of the prompt.

        :param prompt: The input prompt to validate.
        :raises ValueError: If the prompt format is invalid.
        """
        if not isinstance(prompt, list):
            raise ValueError("Prompt must be a list of dictionaries.")
        
        valid_roles = {"system", "user", "assistant"}
        
        for message in prompt:
            if not isinstance(message, dict):
                raise ValueError("Each message in the prompt must be a dictionary.")
            if "role" not in message or "content" not in message:
                raise ValueError("Each message must have 'role' and 'content' keys.")
            if message["role"] not in valid_roles:
                raise ValueError(f"Invalid role '{message['role']}'. Role must be one of {valid_roles}.")


class GPT4o_Model(OpenAI_Model):
    """
    GPT-4o Model class extending the OpenAI_Model class.
    Utilizes the OpenAI client call method for API interaction.
    Inherits from OpenAI_Model and overrides the inference method for GPT-4 specific usage.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GPT4o_Model with the configuration dictionary.
        Uses OpenAI client for API interaction with additional API settings.

        :param config: Configuration dictionary for the model.
        """
        # Set up default model and API key details, and initialize the client
        self.model = config.get("model", "gpt-4")
        api_key = config.get("api_key", os.getenv("OPENAI_API_KEY"))
        helicone_key = os.getenv("HELICONE_API_KEY")

        # If no API key is provided in config or environment, raise an error
        if not api_key:
            raise ValueError("API key is required for GPT-4 model initialization.")

        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://oai.hconeai.com/v1",
            default_headers={"Helicone-Auth": f"Bearer {helicone_key}"}
        )

        self.temperature = config.get("temperature", 1.0)
        self.max_tokens = config.get("max_tokens", None)
        self.n = config.get("n", 1)
        self.stop = config.get("stop", None)
        self.frequency_penalty = config.get("frequency_penalty", 0.0)
        self.presence_penalty = config.get("presence_penalty", 0.0)

        super().__init__(config)

    def _validate_config(self) -> None:
        """
        Override config validation for GPT-4o Model to check client initialization and parameters.
        Ensures required keys are present and valid client initialization.

        :raises ValueError: If required keys or configurations are invalid.
        """
        required_keys = ["model"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        # Validate temperature
        if not (0 <= self.temperature <= 2):
            raise ValueError("Temperature must be between 0 and 2.")

        # Validate frequency penalty
        if not (-2 <= self.frequency_penalty <= 2):
            raise ValueError("Frequency penalty must be between -2 and 2.")

        # Validate presence penalty
        if not (-2 <= self.presence_penalty <= 2):
            raise ValueError("Presence penalty must be between -2 and 2.")

        # Validate max_tokens only if provided
        if self.max_tokens is not None and not (0 <= self.max_tokens <= 4096):
            raise ValueError("Max tokens must be between 0 and 4096 for GPT-4.")

    def generate_api_request(self, prompt_text: str, images_dict: Dict[str, str], max_tokens: Optional[int] = None) -> str:
        """
        Function to call the GPT-4o API with a prompt and associated images in base64 format.

        :param prompt_text: The input text prompt for the model.
        :param images_dict: A dictionary where keys are image numbers and values are base64-encoded images.
        :param max_tokens: Optional maximum number of tokens for the response. Only included if provided.
        :return: The generated text from the GPT-4 API.
        """
        # Construct message content with the prompt and base64-encoded images
        message_content = [{"type": "text", "text": prompt_text}]
        for image_num, base64_image in images_dict.items():
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        # API call to the OpenAI GPT-4 model
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            # Only pass max_tokens if provided
            **({"max_tokens": max_tokens} if max_tokens else {})
        )

        # Return the generated text response
        return response

    def generate_api_request_with_json(self, prompt_text: str, images_dict: Dict[str, str], max_tokens: Optional[int] = None) -> str:
        """
        Function to call the GPT-4o API with a prompt and associated images in base64 format.

        :param prompt_text: The input text prompt for the model.
        :param images_dict: A dictionary where keys are image numbers and values are base64-encoded images.
        :param max_tokens: Optional maximum number of tokens for the response. Only included if provided.
        :return: The generated text from the GPT-4 API.
        """
        # Construct message content with the prompt and base64-encoded images
        message_content = [{"type": "text", "text": prompt_text}]
        for image_num, base64_image in images_dict.items():
            message_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        # API call to the OpenAI GPT-4 model
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            # Only pass max_tokens if provided
            **({"max_tokens": max_tokens} if max_tokens else {}),
            response_format = {"type": "json_object"}
        )

        # Return the generated text response
        return response

    def inference(self, prompt_text: str, images_dict: Dict[str, str] = {}, args: Dict[str, Any] = {}) -> str:
        """
        Perform inference using the GPT-4 model via the OpenAI client call method.

        :param prompt_text: The input text prompt for the model.
        :param images_dict: Dictionary where keys are image numbers and values are base64-encoded image data.
        :param args: Additional arguments, including max_tokens, temperature, etc.
        :return: The generated text response from the GPT-4 API.
        """
        max_tokens = args.get("max_tokens", None)  # Optional max_tokens argument

        # Generate the API request using the prompt and images
        return self.generate_api_request(prompt_text, images_dict, max_tokens)

    def inference_with_json(self, prompt_text: str, images_dict: Dict[str, str] = {}, args: Dict[str, Any] = {}) -> str:
        """
        Perform inference using the GPT-4 model via the OpenAI client call method.

        :param prompt_text: The input text prompt for the model.
        :param images_dict: Dictionary where keys are image numbers and values are base64-encoded image data.
        :param args: Additional arguments, including max_tokens, temperature, etc.
        :return: The generated text response from the GPT-4 API.
        """
        max_tokens = args.get("max_tokens", None)  # Optional max_tokens argument

        # Generate the API request using the prompt and images
        return self.generate_api_request_with_json(prompt_text, images_dict, max_tokens)