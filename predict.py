import joblib

# Load your trained pipeline
pipeline = joblib.load("pipeline.pkl")

# Test sentences (a mix of good, bad, threatening, hateful, and neutral)
texts = [
    "You're a dumb idiot and everyone hates you.",
    "I'll kill you if I see you again.",
    "Hope you're doing well today!",
    "Typical woman can't even drive.",
    "That religion is just a cult of fools.",
    "Your skin color makes you inferior.",
    "Hey, just checking in! You're amazing 💖",
    "You're a fat pig, go to the gym.",
    "All Muslims are terrorists.",
    "Go die in a hole.",
    "Your ethnicity is disgusting.",
    "I hope you fall and break your neck.",
    "You're such a good friend, I love your vibes.",
    "Why are you so slow? Must be your genes.",
    "You're too stupid to understand anything.",
    "Happy birthday! Wish you all the best 🎂🎉",
    "Your religion is a plague.",
    "You're ugly and no one likes you.",
    "I admire how hardworking you are!",
    "I'm going to beat you up after school.",
    "You're just a poor trash from that low caste.",
    "Hey there! You're doing great. Keep going!",
    "You are such a snake, no one can trust you.",
    "You should be ashamed of your gender.",
    "You’re killing it with your skills! 🔥"
]

# Predict labels
predictions = pipeline.predict(texts)

# Display results
for text, labels in zip(texts, predictions):
    print("\n📝 Text:", text)
    print("💡 Prediction:", labels)
