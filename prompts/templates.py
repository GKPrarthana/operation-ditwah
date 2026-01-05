FEW_SHOT_CLASSIFIER = """
You are a crisis response classifier. Provide District, Intent, and Priority.
Examples:
Input: "Breaking News: Kelani River level at 9m."
Output: District: Colombo | Intent: Info | Priority: Low

Message: {message_text}
"""

LOGISTICS_SCORING = """
Analyze these incidents and assign scores (1-10).
{data}
"""

LOGISTICS_STRATEGY = """
Explore 3 rescue branches (Greedy, Speed, Logistics).
Scores: {scores}
"""