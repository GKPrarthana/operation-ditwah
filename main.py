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

run_stability_experiment()