import os
import requests
import json
import streamlit as st
from typing import Optional, Dict, Any


class APIConfig:
    """Direct Gemini API configuration and usage"""

    @staticmethod
    def get_gemini_key() -> Optional[str]:
        """Get Google Gemini API key from session state, environment, or Streamlit secrets"""
        # Check session state first (for user-entered keys)
        if 'gemini_api_key' in st.session_state and st.session_state.gemini_api_key:
            return st.session_state.gemini_api_key

        # Check environment variables
        key = os.getenv('GEMINI_API_KEY')
        if key:
            return key

        # Check Streamlit secrets
        try:
            key = st.secrets.get("GEMINI_API_KEY")
            if key:
                return key
        except:
            pass

        return None

    @staticmethod
    def has_gemini() -> bool:
        """Check if Gemini API key is available"""
        return bool(APIConfig.get_gemini_key())

    @staticmethod
    def generate_text_direct(prompt: str, model: str = "gemini-2.5-flash") -> str:
        """Generate text using Gemini API directly"""
        api_key = APIConfig.get_gemini_key()
        if not api_key:
            return "Gemini API not available. Please check your API key."

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        headers = {
            'Content-Type': 'application/json',
        }

        data = {
            'contents': [{
                'parts': [{
                    'text': prompt
                }]
            }]
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            if 'candidates' in result and len(result['candidates']) > 0:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                return "No response generated"

        except Exception as e:
            return f"Error generating text: {str(e)}"

    @staticmethod
    def get_available_services() -> list[str]:
        """Get list of available API services - Gemini only"""
        services = []
        if APIConfig.has_gemini():
            services.append('Gemini')
        return services

    @staticmethod
    def validate_keys() -> dict[str, bool]:
        """Validate Gemini API key"""
        return {
            'gemini': APIConfig.has_gemini()
        }