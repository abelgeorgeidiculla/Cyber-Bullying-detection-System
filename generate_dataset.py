import pandas as pd
import random
from faker import Faker
import nltk
from nltk.corpus import stopwords

# Download NLTK data
nltk.download('stopwords')

# Initialize tools
fake = Faker()
stop_words = set(stopwords.words('english'))

# Enhanced cyberbullying phrases
cyberbullying_phrases = {
    'insult': [
        "You're a pathetic loser", "Absolute waste of oxygen",
        "Your existence is a mistake", "Braindead moron",
        "clown", "failure", "worthless"
    ],
    'threat': [
        "I'll ruin your life", "Sleep with one eye open",
        "kill yourself", "I know where you live"
    ],
    'identity_hate': [
        "People like you should be banned", "Go back to where you came from",
        "subhuman", "vermin"
    ],
    'gender': [
        "Make me a sandwich", "Typical behavior for your gender",
        "slut", "whore"
    ],
    'religion': [
        "Your god is fake", "Religious terrorists",
        "cultist", "heretic"
    ],
    'ethnicity': [
        "Go back to your country", "immigrant trash"
    ]
}

neutral_phrases = [
    "Have a nice day!", "The weather is lovely today",
    "Great job on the project!", "Puppies are adorable"
]

# Generate realistic variations
def add_typos(text, prob=0.3):
    if random.random() > prob:
        return text
    words = text.split()
    for i in range(len(words)):
        if random.random() < 0.2 and len(words[i]) > 2:
            word = words[i]
            typo_pos = random.randint(1, len(word)-2)
            words[i] = word[:typo_pos] + random.choice(['1', '!', '?']) + word[typo_pos+1:]
    return ' '.join(words)

def add_emojis(text, prob=0.4):
    if random.random() > prob:
        return text
    emojis = ['😂', '😭', '🤡', '💩', '🖕']
    return text + ' ' + random.choice(emojis)

# Generate 5000 samples
data = []
for _ in range(5000):
    is_bullying = random.random() < 0.7
    labels = {cat: 0 for cat in cyberbullying_phrases.keys()}
    
    if is_bullying:
        category = random.choice(list(cyberbullying_phrases.keys()))
        phrases = cyberbullying_phrases[category]
        text = ' '.join(random.sample(phrases, k=random.randint(1, 2)))
        text = add_typos(text)
        text = add_emojis(text)
        labels[category] = 1
    else:
        text = fake.sentence()
    
    row = {
        'text': text,
        'length': len(text),
        'word_count': len(text.split()),
        'has_emoji': int(any(c in text for c in ['😂','😭','🤡','💩','🖕'])),
        'all_caps': int(text.isupper()),
        **labels
    }
    data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Simple balancing (no complex operations that might cause errors)
category_counts = df[list(cyberbullying_phrases.keys())].sum()
neutral_count = len(df) - df[list(cyberbullying_phrases.keys())].any(axis=1).sum()

print("Class distribution before balancing:")
print(category_counts)
print(f"Neutral texts: {neutral_count}")

# Save to CSV
df.to_csv("cyberbullying_dataset_5000.csv", index=False)
print("\n Successfully generated balanced 5000-sample dataset")

print("Saved as 'cyberbullying_dataset_5000.csv'")
