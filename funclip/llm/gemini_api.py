"""
Gemini API integration for LLM-based video clipping.
Supports Gemini 2.5 Pro and Gemini 2.5 Flash models.
"""

import logging

import google.generativeai as genai


def gemini_call(
    apikey: str,
    model: str = "gemini-2.5-flash",
    user_content: str = "如何做西红柿炖牛腩？",
    system_content: str | None = None,
) -> str:
    """
    Call Google Gemini API for text generation.

    Args:
        apikey: Google AI API key
        model: Model name (gemini-2.5-pro, gemini-2.5-flash, etc.)
        user_content: User message content
        system_content: Optional system instruction

    Returns:
        Generated text response
    """
    genai.configure(api_key=apikey)

    # Build generation config
    # Use large max_output_tokens to avoid truncation for long SRT content
    generation_config = genai.GenerationConfig(
        temperature=0.7,
        max_output_tokens=81920,
    )

    # Initialize model with optional system instruction
    if system_content and system_content.strip():
        gemini_model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            system_instruction=system_content,
        )
    else:
        gemini_model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
        )

    # Generate response
    response = gemini_model.generate_content(user_content)
    logging.info(f"Gemini model ({model}) inference done.")

    return response.text


if __name__ == "__main__":
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        result = gemini_call(
            apikey=api_key,
            model="gemini-2.5-flash",
            user_content="Hello, what's your name?",
        )
        print(result)
    else:
        print("Please set GEMINI_API_KEY environment variable")
