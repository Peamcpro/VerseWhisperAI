import random
import textwrap
import nltk
from nltk.corpus import cmudict
from transformers import pipeline
from openai import OpenAI

# Load Sentiment Analysis Model
sentiment_analyzer = pipeline("sentiment-analysis")

# Load CMU Pronouncing Dictionary for Rhyming
nltk.download('cmudict')
pronouncing_dict = cmudict.dict()

# OpenAI API Client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",  # Replace with your API key
)

def count_syllables(word):
    """Count syllables in a word using CMU Pronouncing Dictionary."""
    return [len([y for y in x if y[-1].isdigit()]) for x in pronouncing_dict.get(word.lower(), [[word]])][0]

def generate_poem(topic, style, tone, length):
    """Generate an AI-enhanced poem with rhyme and sentiment control."""

    # Adjust tone based on sentiment analysis
    sentiment_result = sentiment_analyzer(tone)
    if sentiment_result[0]['label'] == 'NEGATIVE':
        tone = "melancholic"
    elif sentiment_result[0]['label'] == 'POSITIVE':
        tone = "joyful"

    # Advanced AI Prompt
    prompt = (
        f"Write a {style} poem about {topic} in a {tone} style. "
        f"Ensure it has strong imagery, a rhythmic flow, and natural line breaks. "
        f"Use literary devices like metaphors, personification, and similes. "
        f"Follow appropriate rhyming patterns if applicable. Keep it {length}.\n\n"
    )

    try:
        # Generate poem using OpenAI API
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://openrouter.ai/settings/keys",
                "X-Title": "AI Poetry Generator",
            },
            model="deepseek/deepseek-r1:free",
            messages=[{"role": "user", "content": prompt}],
        )

        poem = completion.choices[0].message.content.strip()
        
        # Enhance readability
        formatted_poem = "\n".join(textwrap.wrap(poem, width=60))

        return formatted_poem
    
    except Exception as e:
        return f"❌ An error occurred: {e}"
    
def user_input_with_guidance():
    """Guide the user through the input process with explanations."""
    print("""
    🎵====================================
         WELCOME TO THE AI POETRY GENERATOR!
    🎵====================================
    """)

    print("""
    📚 USER GUIDE 📚
    1️⃣ Enter the topic of the poem (e.g., nature, love, technology).
    2️⃣ Choose the poetic style (e.g., haiku, sonnet, free verse).
    3️⃣ Specify the tone (e.g., melancholic, inspirational, humorous).
    4️⃣ Indicate the length (short or long).
    5️⃣ Optionally, generate a haiku after the main poem.
    6️⃣ Save the generated poem if desired.
    """)

    # User Input with Explanations
    topic = input("📖 Enter the topic of the poem (e.g., nature, love, technology): ").strip()
    print("ℹ️ Topic is the main subject or theme of the poem. Example: 'love', 'space', 'adventure'.")

    style_options = ["haiku", "sonnet", "free verse"]
    style = input("✒️ Enter the poetic style (haiku, sonnet, free verse): ").strip().lower()
    while style not in style_options:
        print("❌ Invalid style. Please choose from: haiku, sonnet, free verse.")
        style = input("✒️ Enter the poetic style again: ").strip().lower()

    tone = input("💬 Enter the tone (e.g., melancholic, inspirational, humorous): ").strip()
    print("ℹ️ Tone is the mood of the poem. Example: 'melancholic' (sad), 'inspirational' (uplifting).")

    length_options = ["short", "long"]
    length = input("📜 Enter the length (short or long): ").strip().lower()
    while length not in length_options:
        print("❌ Invalid length. Please choose 'short' or 'long'.")
        length = input("📜 Enter the length again: ").strip().lower()

    return topic, style, tone, length

if __name__ == "__main__":
    topic, style, tone, length = user_input_with_guidance()

    print("\n🎨 Generating your enhanced AI poem...\n")
    
    poem = generate_poem(topic, style, tone, length)

    print("\n📖 Your AI-Enhanced Poem 📖\n")
    print(poem)

    save_option = input("\n💾 Would you like to save this poem? (yes/no): ").strip().lower()
    if save_option == "yes":
        with open("generated_poem.txt", "w") as file:
            file.write(poem)
        print("📥 Poem saved as 'generated_poem.txt'! Enjoy your masterpiece! 🎨")
    else:
        print("\n🎵 Thank you for using the AI Poetry Generator! Keep creating! 🎵")
