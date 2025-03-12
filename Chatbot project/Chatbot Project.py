import random
import textwrap
import nltk
from nltk.corpus import cmudict
from transformers import pipeline
from openai import OpenAI
from datasets import load_dataset  # Import the datasets library

# Load Sentiment Analysis Model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Load CMU Pronouncing Dictionary for Rhyming
nltk.download('cmudict')
pronouncing_dict = cmudict.dict()

# Load dataset for training
dataset = load_dataset('json', data_files='C:/Users/Peam/OneDrive/เอกสาร/GitHub/VerseWhisperAI/Dataset/poetry_dataset.json')

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

def check_poem_accuracy(poem, style, tone):
    """Check the accuracy of the generated poem based on the given style and tone."""
    accuracy_report = []

    # Check style
    if style == "haiku":
        lines = poem.split('\n')
        if len(lines) == 3:
            syllable_counts = [count_syllables(line) for line in lines]
            if syllable_counts == [5, 7, 5]:
                accuracy_report.append("✅ Haiku structure is correct (5-7-5 syllables).")
            else:
                accuracy_report.append(f"❌ Haiku structure is incorrect. Syllable counts: {syllable_counts}")
        else:
            accuracy_report.append(f"❌ Haiku should have exactly 3 lines. Found: {len(lines)} lines.")
    elif style == "sonnet":
        lines = poem.split('\n')
        if len(lines) == 14:
            accuracy_report.append("✅ Sonnet structure is correct (14 lines).")
        else:
            accuracy_report.append(f"❌ Sonnet should have exactly 14 lines. Found: {len(lines)} lines.")
    elif style == "free verse":
        accuracy_report.append("✅ Free verse has no specific structure.")

    # Check tone
    sentiment_result = sentiment_analyzer(poem)
    poem_tone = sentiment_result[0]['label'].lower()
    if tone.lower() in poem_tone:
        accuracy_report.append(f"✅ Tone matches the expected tone: {tone}.")
    else:
        accuracy_report.append(f"❌ Tone does not match. Expected: {tone}, Found: {poem_tone}.")

    return "\n".join(accuracy_report)

if __name__ == "__main__":
    topic, style, tone, length = user_input_with_guidance()

    print("\n🎨 Generating your enhanced AI poem...\n")
    
    poem = generate_poem(topic, style, tone, length)

    print("\n📖 Your AI-Enhanced Poem 📖\n")
    print(poem)

    # Check accuracy
    accuracy_report = check_poem_accuracy(poem, style, tone)
    print("\n🔍 Accuracy Report 🔍\n")
    print(accuracy_report)

    save_option = input("\n💾 Would you like to save this poem? (yes/no): ").strip().lower()
    if save_option == "yes":
        with open("generated_poem.txt", "w") as file:
            file.write(poem)
        print("📥 Poem saved as 'generated_poem.txt'! Enjoy your masterpiece! 🎨")
    else:
        print("\n🎵 Thank you for using the AI Poetry Generator! Keep creating! 🎵")
