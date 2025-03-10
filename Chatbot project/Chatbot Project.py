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

def generate_haiku(topic):
    """Generate a haiku poem about a given topic."""
    prompt = (
        f"Write a haiku about {topic}. "
        f"Ensure it follows the 5-7-5 syllable structure and captures the essence of the topic.\n\n"
    )

    try:
        # Generate haiku using OpenAI API
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://openrouter.ai/settings/keys",
                "X-Title": "AI Haiku Generator",
            },
            model="deepseek/deepseek-r1:free",
            messages=[{"role": "user", "content": prompt}],
        )

        haiku = completion.choices[0].message.content.strip()
        
        # Enhance readability
        formatted_haiku = "\n".join(textwrap.wrap(haiku, width=60))

        return formatted_haiku
    
    except Exception as e:
        return f"❌ An error occurred: {e}"

def generate_sonnet(topic):
    """Generate a sonnet poem about a given topic."""
    prompt = (
        f"Write a sonnet about {topic}. "
        f"Ensure it follows the traditional 14-line structure with an ABABCDCDEFEFGG rhyme scheme.\n\n"
    )

    try:
        # Generate sonnet using OpenAI API
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://openrouter.ai/settings/keys",
                "X-Title": "AI Sonnet Generator",
            },
            model="deepseek/deepseek-r1:free",
            messages=[{"role": "user", "content": prompt}],
        )

        sonnet = completion.choices[0].message.content.strip()
        
        # Enhance readability
        formatted_sonnet = "\n".join(textwrap.wrap(sonnet, width=60))

        return formatted_sonnet
    
    except Exception as e:
        return f"❌ An error occurred: {e}"

def generate_free_verse(topic):
    """Generate a free verse poem about a given topic."""
    prompt = (
        f"Write a free verse poem about {topic}. "
        f"Ensure it has strong imagery, a rhythmic flow, and natural line breaks. "
        f"Use literary devices like metaphors, personification, and similes.\n\n"
    )

    try:
        # Generate free verse using OpenAI API
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://openrouter.ai/settings/keys",
                "X-Title": "AI Free Verse Generator",
            },
            model="deepseek/deepseek-r1:free",
            messages=[{"role": "user", "content": prompt}],
        )

        free_verse = completion.choices[0].message.content.strip()
        
        # Enhance readability
        formatted_free_verse = "\n".join(textwrap.wrap(free_verse, width=60))

        return formatted_free_verse
    
    except Exception as e:
        return f"❌ An error occurred: {e}"


if __name__ == "__main__":
    print("""
    🎵====================================
         WELCOME TO THE AI POETRY GENERATOR!
    🎵====================================
    """)

    # User Input
    topic = input("📖 Enter the topic of the poem (e.g., nature, love, technology): ").strip()
    style = input("✒️ Enter the poetic style (e.g., haiku, sonnet, free verse): ").strip()
    tone = input("💬 Enter the tone (e.g., melancholic, inspirational, humorous): ").strip()
    length = input("📜 Enter the length (short or long): ").strip()

    print("\n🎨 Generating your enhanced AI poem...\n")
    
    if style.lower() == "sonnet":
        poem = generate_sonnet(topic)
    elif style.lower() == "free verse":
        poem = generate_free_verse(topic)
    else:
        poem = generate_poem(topic, style, tone, length)

    print("\n📖 Your AI-Enhanced Poem 📖\n")
    print(poem)

    haiku_option = input("\n🌸 Would you like to generate a haiku? (yes/no): ").strip().lower()
    if haiku_option == "yes":
        haiku_topic = input("📖 Enter the topic of the haiku (e.g., nature, love, technology): ").strip()
        print("\n🎨 Generating your haiku...\n")
        haiku = generate_haiku(haiku_topic)
        print("\n📖 Your Haiku 📖\n")
        print(haiku)
    else:
        save_option = input("\n💾 Would you like to save this poem? (yes/no): ").strip().lower()
        if save_option == "yes":
            with open("generated_poem.txt", "w") as file:
                file.write(poem)
            print("📥 Poem saved as 'generated_poem.txt'! Enjoy your masterpiece! 🎨")
        else:
            print("\n🎵 Thank you for using the AI Poetry Generator! Keep creating! 🎵")
