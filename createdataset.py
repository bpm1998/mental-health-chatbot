import pandas as pd
import random

suicidal_templates = [
    "I don't see any reason to live anymore.",
    "Everything feels hopeless and empty.",
    "I'm thinking about ending my life.",
    "Life hurts too much.",
    "I want all the pain to stop.",
    "There’s no escape from this sadness.",
    "I feel like a burden to everyone.",
    "No one will miss me if I'm gone.",
    "Death seems like the only option.",
    "I wish I could disappear forever."
]

non_suicidal_templates = [
    "I had a great day with my friends!",
    "Looking forward to my vacation next week.",
    "I love spending time with my family.",
    "Feeling really productive and motivated today.",
    "I’m enjoying my new hobby.",
    "Excited to start my new job!",
    "Today was a peaceful day.",
    "I'm feeling calm and happy.",
    "Everything is going well lately.",
    "Grateful for the little things in life."
]

# Generate 100 of each with random selection
suicidal_data = [random.choice(suicidal_templates) + " " + str(i) for i in range(100)]
non_suicidal_data = [random.choice(non_suicidal_templates) + " " + str(i) for i in range(100)]

# Create dataframe
df_synthetic = pd.DataFrame({
    "clean_selftext": suicidal_data + non_suicidal_data,
    "suicidal_intent": ["suicidal"] * 100 + ["not suicidal"] * 100
})

# Save to CSV
df_synthetic.to_csv("synthetic_suicide_data.csv", index=False)
print("✅ Saved synthetic_suicide_data.csv with", len(df_synthetic), "records")
