from utils.token_utils import count_tokens

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

def run_budget_demo():
    """Demonstrates the budget keeper functionality."""
    long_spam = "HELP " * 200
    clean_message = budget_keeper_gatekeeper(long_spam)
    print(f"Gatekeeper Result: {clean_message}")
    print("Part 4 Complete: Budget Keeper demonstrated token economics")
