import os
import pandas as pd
import openai
from prompts.templates import FEW_SHOT_CLASSIFIER

def classify_message(client, message_text):
    """Refactored classification function using templates."""
    prompt = FEW_SHOT_CLASSIFIER.format(message_text=message_text)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def process_messages(input_path, output_path="output/classified_messages.csv"):
    """Processes messages and saves results to CSV."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, "r") as f:
        messages = [line.strip() for line in f.readlines() if line.strip()]
        
    results = []
    print(f"Processing {len(messages)} messages...")
    
    for msg in messages:
        classification = classify_message(client, msg)
        results.append({"Raw Message": msg, "Classification": classification})

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Part 1 Complete: Saved to {output_path}")