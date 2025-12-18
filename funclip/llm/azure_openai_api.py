"""
Azure OpenAI API integration for LLM-based video clipping.
This module is reserved for future Azure OpenAI support.

TODO: Implement Azure OpenAI integration when needed.
Required environment variables:
    - AZURE_OPENAI_API_KEY
    - AZURE_OPENAI_ENDPOINT
    - AZURE_OPENAI_DEPLOYMENT_NAME
"""


def azure_openai_call(
    apikey: str,
    endpoint: str,
    deployment: str,
    user_content: str = "如何做西红柿炖牛腩？",
    system_content: str | None = None,
) -> str:
    """
    Call Azure OpenAI API for text generation.

    NOTE: This function is not yet implemented.
    Placeholder for future Azure OpenAI support.

    Args:
        apikey: Azure OpenAI API key
        endpoint: Azure OpenAI endpoint URL
        deployment: Deployment name
        user_content: User message content
        system_content: Optional system instruction

    Returns:
        Generated text response

    Raises:
        NotImplementedError: This function is not yet implemented
    """
    raise NotImplementedError(
        "Azure OpenAI support is not yet implemented. Please use Gemini models for now."
    )


# Future implementation reference:
# from openai import AzureOpenAI
#
# def azure_openai_call(...):
#     client = AzureOpenAI(
#         api_key=apikey,
#         api_version="2024-02-15-preview",
#         azure_endpoint=endpoint,
#     )
#
#     messages = []
#     if system_content:
#         messages.append({"role": "system", "content": system_content})
#     messages.append({"role": "user", "content": user_content})
#
#     response = client.chat.completions.create(
#         model=deployment,
#         messages=messages,
#     )
#     return response.choices[0].message.content
