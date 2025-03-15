import os
import random
import textwrap
import nltk
from nltk.corpus import cmudict
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import openai 
from datasets import load_dataset
import tkinter as tk
from tkinter import Button, filedialog, Label
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import requests

# Force the use of CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Update logging settings
logging.getLogger('transformers').setLevel(logging.ERROR)

def update_cmudict():
    """Update the CMU Pronouncing Dictionary."""
    nltk.download('cmudict', force=True)

# Update CMU Pronouncing Dictionary
update_cmudict()
pronouncing_dict = cmudict.dict()

# ✅ Load Sentiment Analysis Model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)  # ✅ Use CPU

# ✅ OpenAI API Client
client = openai.OpenAI(  # If using OpenAI SDK v1+
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

def count_syllables(word):
    """Count syllables in a word using CMU Pronouncing Dictionary."""
    return [len([y for y in x if y[-1].isdigit()]) for x in pronouncing_dict.get(word.lower(), [[word]])][0]

def find_rhymes(word):
    """Find rhyming words using CMU Pronouncing Dictionary."""
    word = word.lower()
    if word in pronouncing_dict:
        word_pronunciation = pronouncing_dict[word]
        rhymes = []
        for key, value in pronouncing_dict.items():
            if key != word and any(pron[-2:] == word_pron[-2:] for pron in value for word_pron in word_pronunciation):
                rhymes.append(key)
        return rhymes
    return []

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

# Model Performance and Evaluation Methods

def evaluate_model_performance():
    """Evaluate the model performance using various metrics."""
    # Example evaluation logic
    true_labels = ["positive", "negative", "positive"]
    predicted_labels = ["positive", "negative", "negative"]

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

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

    # Evaluate model performance
    evaluate_model_performance()
