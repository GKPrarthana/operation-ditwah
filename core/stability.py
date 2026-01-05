import os
import openai

def run_stability_test(client, scenario_text, temp):
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
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
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
            output = run_stability_test(client, text, 1.0)
            chaos_outputs.append(output)
            print(f"Run {run+1} Output: {output[:200]}...") 
            
        print(f"\n[SAFE MODE - TEMP 0.0]")
        safe_output = run_stability_test(client, text, 0.0)
        print(f"Final Output: {safe_output}")
        
        print(f"\n[ANALYSIS]")
        print("Looking for drift in Chaos Mode outputs:")
        for j in range(1, len(chaos_outputs)):
            if chaos_outputs[0][:100] != chaos_outputs[j][:100]:
                print(f"  DRIFT DETECTED between Run 1 and Run {j+1}")
        
        print("Comparing Safe vs Chaos modes:")
        if safe_output[:100] != chaos_outputs[0][:100]:
            print("  DIFFERENCE between Safe and Chaos modes")
