import openai as OpenAI
import os
import textwrap
import requests
import tkinter as tk

# OpenAI API Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

def generate_poem(topic, style, tone, length):
    """
    Generate a poem based on user-defined parameters using DeepSeek API.
    
    :param topic: The subject of the poem (e.g., nature, technology).
    :param style: The poetic form (e.g., haiku, sonnet, free verse).
    :param tone: The emotional tone (e.g., melancholic, inspirational).
    :param length: Desired length (e.g., short, long).
    :return: AI-generated poem as a string.
    """
    
    # Craft the prompt for the AI
    prompt = (
        f"\n\U0001F4DD Write a {style} about {topic} in a {tone} style. "
        f"Use vivid imagery and metaphors. Keep it {length}.\n"
    )

    headers = {
        'Authorization': f'Bearer {client.api_key}',
        'Content-Type': 'application/json'
    }

    data = {
        "prompt": prompt,
        "max_tokens": 150 if length.lower() == "short" else 300,
        "temperature": 0.7  # Controls creativity. Higher values = more creative.
    }

    try:
        response = requests.post(client.base_url, headers=headers, json=data)
        response.raise_for_status()  # Raise an error for bad status codes
        result = response.json()
        poem = result.get('text', '').strip()
        return textwrap.fill(poem, width=70)
    
    except requests.exceptions.RequestException as e:
        return f"\U0001F6A8 An error occurred: {e}"
    except Exception as e:
        return f"\U0001F6A8 An unexpected error occurred: {e}"

if __name__ == "__main__":
    # Welcome banner
    print("""
    \U0001F3B6====================================\U0001F3B6
        WELCOME TO THE AI POETRY GENERATOR!
    \U0001F3B6====================================\U0001F3B6
    """)

    # User input
    topic = input("\U0001F4D6 Enter the topic of the poem (e.g., nature, love, technology): ")
    style = input("\U0001F58B️ Enter the poetic style (e.g., haiku, sonnet, free verse): ")
    tone = input("\U0001F4AC Enter the tone (e.g., melancholic, inspirational, humorous): ")
    length = input("\U0001F4C4 Enter the length (short or long): ")

    # Generating poem
    print("\n\U0001F3A8 Generating your personalized poem...\n")
    poem = generate_poem(topic, style, tone, length)

    # Displaying poem
    print(f"\U0001F4D6 Your AI-Generated Poem \U0001F4D6\n")
    print(poem)

    # Option to save the poem to a file
    save_option = input("\n\U0001F4BE Would you like to save this poem? (yes/no): ").strip().lower()
    if save_option == "yes":
        with open("generated_poem.txt", "w") as file:
            file.write(poem)
        print("\U0001F4E5 Poem saved as 'generated_poem.txt'! Enjoy your masterpiece! \U0001F3A8")
    else:
        print("\n\U0001F3B5 Thank you for using the AI Poetry Generator! Keep creating! \U0001F3B5")