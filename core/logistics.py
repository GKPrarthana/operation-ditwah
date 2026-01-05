import os
import openai
from prompts.templates import LOGISTICS_SCORING, LOGISTICS_STRATEGY

def run_commander(incidents_path):
    """Executes the logistics planning steps."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    with open(incidents_path, "r") as f:
        incidents_data = f.read()

    scores_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": LOGISTICS_SCORING.format(data=incidents_data)}],
        temperature=0
    )
    scores_output = scores_response.choices[0].message.content
    print("\n--- PRIORITY SCORES ---")
    print(scores_output)

    strategy_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": LOGISTICS_STRATEGY.format(scores=scores_output)}],
        temperature=0
    )
    print("\n--- OPTIMAL ROUTE (ToT) ---")
    print(strategy_response.choices[0].message.content)