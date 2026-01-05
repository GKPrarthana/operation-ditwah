import os
import pandas as pd
import openai
from models.schemas import CrisisEvent

def extract_structured_data(client, text):
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
    """Processes news feed and extracts structured crisis data."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    news_path = r"data/News Feed.txt"
    valid_events = []

    with open(news_path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Starting Extraction Pipeline for {len(lines)} items...")

    for line in lines:
        try:
            raw_json = extract_structured_data(client, line)
            event = CrisisEvent.model_validate_json(raw_json)
            valid_events.append(event.model_dump())
        except Exception as e:
            print(f"!!! Skipping invalid data: {line[:30]}... | Error: {e}")

    print(f"Extracted {len(valid_events)} valid events")
    df = pd.DataFrame(valid_events)
    df.to_csv("output/flood_report.csv", index=False)
    print("Part 5 Complete: Structured report saved to output/flood_report.csv")
