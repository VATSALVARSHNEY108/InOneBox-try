import os
import json
import streamlit as st
from typing import Optional, Dict, Any, List
import requests
import base64
from api_config import APIConfig
# Using OpenAI integration for image generation - the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user
from openai import OpenAI


class AIClient:
    """Unified AI client using direct API calls"""

    def __init__(self):
        self.api_config = APIConfig()

    def _init_clients(self):
        """No initialization needed for direct API calls"""
        pass

    def generate_text(self, prompt: str, model: str = "gemini", max_tokens: int = 1000) -> str:
        """Generate text using direct Gemini API"""
        try:
            # Adjust prompt based on requested "model" style
            if model == "openai":
                enhanced_prompt = f"Act like GPT/OpenAI model. {prompt}"
            elif model == "anthropic":
                enhanced_prompt = f"Act like Claude/Anthropic model. {prompt}"
            else:
                enhanced_prompt = prompt

            # Use direct API call
            return APIConfig.generate_text_direct(enhanced_prompt)

        except Exception as e:
            return f"Error generating text: {str(e)}"

    def analyze_image(self, image_data: bytes, prompt: str = "Analyze this image", model: str = "gemini") -> str:
        """Analyze image using direct Gemini API"""
        try:
            return "Image analysis feature requires library integration. Please use text generation tools."
        except Exception as e:
            return f"Error analyzing image: {str(e)}"

    def generate_image(self, prompt: str, model: str = "dall-e-3") -> Optional[bytes]:
        """Generate image using OpenAI DALL-E"""
        try:
            # Get OpenAI API key
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                st.error("OpenAI API key not found. Please configure your API key.")
                return None

            # Initialize OpenAI client
            client = OpenAI(api_key=openai_api_key)

            # Generate image using DALL-E 3
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024",
                quality="standard"
            )

            # Get the image URL
            image_url = response.data[0].url

            # Download the image and return as bytes
            image_response = requests.get(image_url)
            image_response.raise_for_status()

            return image_response.content

        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
            return None

    def analyze_sentiment(self, text: str, model: str = "gemini") -> Dict[str, Any]:
        """Analyze sentiment using direct Gemini API"""
        try:
            # Adjust prompt based on requested "model" style
            base_prompt = f"""
            Analyze the sentiment of the following text and provide:
            1. Overall sentiment (positive/negative/neutral)
            2. Confidence score (0-1)
            3. Key emotional indicators
            4. Brief explanation

            Text: {text}

            Respond in JSON format.
            """

            if model == "openai":
                prompt = f"Act like GPT/OpenAI sentiment analysis model. {base_prompt}"
            elif model == "anthropic":
                prompt = f"Act like Claude sentiment analysis model. {base_prompt}"
            else:
                prompt = base_prompt

            # Use direct API call
            response_text = APIConfig.generate_text_direct(prompt)

            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                return {"error": "Invalid JSON response from API"}

        except Exception as e:
            return {"error": f"Sentiment analysis failed: {str(e)}"}

    def translate_text(self, text: str, target_language: str, model: str = "gemini") -> str:
        """Translate text to target language"""
        try:
            prompt = f"Translate the following text to {target_language}: {text}"
            return self.generate_text(prompt, model=model)
        except Exception as e:
            return f"Translation failed: {str(e)}"

    def summarize_text(self, text: str, max_sentences: int = 3, model: str = "gemini") -> str:
        """Summarize text to specified number of sentences"""
        try:
            prompt = f"Summarize the following text in {max_sentences} sentences: {text}"
            return self.generate_text(prompt, model=model)
        except Exception as e:
            return f"Summarization failed: {str(e)}"

    def get_available_models(self) -> List[str]:
        """Get list of available AI models - All powered by single Gemini API"""
        # All models use the same Gemini API - just different prompting styles
        return ["gemini", "openai", "anthropic"] if APIConfig.has_gemini() else []


# Global AI client instance
ai_client = AIClient()
