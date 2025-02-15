from openai import OpenAI
import requests
import textwrap

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
    prompt = (
        f"\n\U0001F4DD Write a {style} about {topic} in a {tone} style. "
        f"Use vivid imagery and metaphors. Keep it {length}.\n"
    )

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://openrouter.ai/settings/keys",  # Optional. Site URL for rankings on openrouter.ai.
                "X-Title": "Openrounter.ai",  # Optional. Site title for rankings on openrouter.ai.
            },
            extra_body={},
            model="deepseek/deepseek-r1:free",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        poem = completion.choices[0].message.content.strip()
        return textwrap.fill(poem, width=70)
    
    except Exception as e:
        return f"\U0001F6A8 An error occurred: {e}"


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