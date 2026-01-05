import pandas as pd
import os
from openai import OpenAI
import dotenv
import logging
import openai
import tiktoken
from pydantic import BaseModel, Field, ValidationError
from typing import Literal, Optional
import json

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
output_path = "output/classified_messages.csv"

results = []

# if os.path.exists(input_path):
#     with open(input_path, "r") as f:
#         messages = [line.strip() for line in f.readlines() if line.strip()]
        
#     print(f"Processing {len(messages)} messages...")
    
#     for msg in messages:
#         classification = classify_message(msg)
#         results.append({"Raw Message": msg, "Classification": classification})

#     df = pd.DataFrame(results)
#     os.makedirs("output", exist_ok=True)
#     df.to_csv(output_path, index=False)
#     print(f"Success! Results saved to {output_path}")
# else:
#     print("Error: Input file not found.")

def run_stability_test(scenario_text, temp):
    """
    Runs Chain of Thought reasoning on a scenario at specified temperature.
    """
    cot_prompt = f"""
    Analyze this crisis scenario step-by-step. 
    1. Identify all life threats.
    2. Identify medical emergencies.
    3. Identify resource needs.
    4. Provide a final priority recommendation.

    Scenario: {scenario_text}
    """
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": cot_prompt}],
        temperature=temp
    )
    return response.choices[0].message.content

def run_stability_experiment():
    """
    Runs the temperature stability test on all scenarios.
    """
    scenarios_path = "data/Scenarios.txt"
    
    with open(scenarios_path, "r") as f:
        content = f.read()
        scenarios = content.split("SCENARIO")[1:]  

    for i, text in enumerate(scenarios):
        print(f"\n--- TESTING SCENARIO {i+1} ---")
        print(f"Scenario text: {text.strip()[:100]}...")
        
        print(f"\n[CHAOS MODE - TEMP 1.0]")
        chaos_outputs = []
        for run in range(3):
            output = run_stability_test(text, 1.0)
            chaos_outputs.append(output)
            print(f"Run {run+1} Output: {output[:200]}...") 
            
        print(f"\n[SAFE MODE - TEMP 0.0]")
        safe_output = run_stability_test(text, 0.0)
        print(f"Final Output: {safe_output}")
        
        print(f"\n[ANALYSIS]")
        print("Looking for drift in Chaos Mode outputs:")
        for j in range(1, len(chaos_outputs)):
            if chaos_outputs[0][:100] != chaos_outputs[j][:100]:
                print(f"  DRIFT DETECTED between Run 1 and Run {j+1}")
        
        print("Comparing Safe vs Chaos modes:")
        if safe_output[:100] != chaos_outputs[0][:100]:
            print("  DIFFERENCE between Safe and Chaos modes")

def run_logistics_commander():
    """
    Combines CoT scoring and ToT routing to find the optimal rescue path.
    """
    with open("data/Incidents.txt", "r") as f:
        incidents_data = f.read()

    scoring_prompt = f"""
    Analyze these incidents row by row and assign a Priority Score (1-10):
    Rules:
    - Base Score: 5
    - +2 if Age > 60 or < 5 [cite: 188]
    - +3 if Need is "Rescue" (Life Threat) [cite: 189]
    - +1 if Need is "Insulin/Medicine" [cite: 190]
    
    Incidents:
    {incidents_data}
    
    Output the ID and the Final Score for each.
    """
    
    scores_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": scoring_prompt}],
        temperature=0
    )
    scores_output = scores_response.choices[0].message.content
    print("\n--- STEP A: PRIORITY SCORES ---")
    print(scores_output)

    strategy_prompt = f"""
    Act as a Logistics Commander. We have ONE boat at Ragama.
    
    SCORING DATA:
    {scores_output}
    
    CONSTRAINTS:
    - Ragama to Ja-Ela: 10 mins
    - Ja-Ela to Gampaha: 40 mins
    - Ragama to Gampaha: 50 mins (via Ja-Ela)
    
    TASK: Explore 3 branches to maximize priority score saved in shortest time[cite: 105]:
    Branch 1: Highest Score First (Greedy) 
    Branch 2: Closest First (Speed) 
    Branch 3: Furthest First (Logistics) 
    
    Select the optimal route and justify it.
    """
    
    strategy_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": strategy_prompt}],
        temperature=0
    )
    print("\n--- STEP B: OPTIMAL ROUTE (ToT) ---")
    print(strategy_response.choices[0].message.content)

def count_tokens(text, model="gpt-4o-mini"):
    """Counts the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def budget_keeper_gatekeeper(message_text):
    """
    Checks if a message is too long. 
    If > 150 tokens, it truncates and marks as BLOCKED/TRUNCATED.
    """
    token_count = count_tokens(message_text)
    
    if token_count > 150:
        truncated_text = message_text[:100] + "..." 
        print(f"!!! [BLOCKED/TRUNCATED] Message is {token_count} tokens. Reducing costs.")
        return f"TRUNCATED SPAM: {truncated_text}"
    
    return message_text

long_spam = "HELP " * 200
clean_message = budget_keeper_gatekeeper(long_spam)
print(f"Gatekeeper Result: {clean_message}")

class CrisisEvent(BaseModel):
    district: Literal["Colombo", "Gampaha", "Kandy", "Kalutara", "Galle", "Kegalle", "Ratnapura", "Matara", "Badulla", "Nuwara Eliya"]
    flood_level_meters: Optional[float] = None
    victim_count: int = Field(default=0)
    main_need: str
    status: Literal["Critical", "Warning", "Stable"]

def extract_structured_data(text):
    """Refined json_extract.v1 to match CrisisEvent schema exactly."""
    extract_prompt = f"""
    Extract data into JSON. You MUST use these exact keys:
    - "district": Must be one of [Colombo, Gampaha, Kandy, Kalutara, Galle, Kegalle, Ratnapura, Matara, Badulla, Nuwara Eliya]
    - "flood_level_meters": float or null
    - "victim_count": integer
    - "main_need": string (if none, put "None")
    - "status": One of [Critical, Warning, Stable]
    
    News Item: "{text}"
    
    Respond ONLY with the raw JSON object.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": extract_prompt}],
        response_format={ "type": "json_object" },
        temperature=0
    )
    return response.choices[0].message.content

def run_news_pipeline():
    news_path = "data/News Feed.txt"
    valid_events = []

    with open(news_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Starting Extraction Pipeline for {len(lines)} items...")

    for line in lines:
        try:
            raw_json = extract_structured_data(line)
            event = CrisisEvent.model_validate_json(raw_json)
            valid_events.append(event.model_dump())
        except Exception as e:
            print(f"!!! Skipping invalid data: {line[:30]}... | Error: {e}")

    print(f"Extracted {len(valid_events)} valid events")
    df = pd.DataFrame(valid_events)
    df.to_csv("output/flood_report.csv", index=False)
    print("Pipeline Complete! Structured report saved to output/flood_report.csv")

#run_stability_experiment()
#run_logistics_commander()
run_news_pipeline()