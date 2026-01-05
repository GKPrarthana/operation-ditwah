import pandas as pd
import os
from openai import OpenAI
import dotenv
import logging
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not set")
    raise ValueError("OPENAI_API_KEY is not set in the environment variables")

client = openai.OpenAI(api_key=api_key)

def classify_message(message_text):
    """
    Uses Few-Shot prompting to classify a single message.
    """

    few_shot_prompt = f"""
    You are a crisis response classifier. 
    Classify the message based on: District, Intent (Rescue, Supply, Info, Other), and Priority (High, Low).
    
    Examples:
    Input: "Breaking News: Kelani River level at 9m."
    Output: District: Colombo | Intent: Info | Priority: Low
    
    Input: "We are trapped on the roof with 3 kids!"
    Output: District: None | Intent: Rescue | Priority: High
    
    Input: "Gampaha hospital is requesting drinking water for patients."
    Output: District: Gampaha | Intent: Supply | Priority: High
    
    Input: "I lost my ID card in the flood."
    Output: District: None | Intent: Other | Priority: Low

    Message to classify: "{message_text}"
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": few_shot_prompt}],
        temperature=0
    )
    return response.choices[0].message.content

input_path = "data/Sample Messages.txt"
output_path = "output/classified_messages.xlsx"

results = []

if os.path.exists(input_path):
    with open(input_path, "r") as f:
        messages = [line.strip() for line in f.readlines() if line.strip()]
        
    print(f"Processing {len(messages)} messages...")
    
    for msg in messages:
        classification = classify_message(msg)
        results.append({"Raw Message": msg, "Classification": classification})

    df = pd.DataFrame(results)
    os.makedirs("output", exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"Success! Results saved to {output_path}")
else:
    print("Error: Input file not found.")