import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

def generate_ai_text(prompt):
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing in the .env file.")
        
    with genai.Client(api_key=GEMINI_API_KEY) as client:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=700,
                system_instruction=(
                    "You are a careful pension assistance guide. "
                    "Use simple language. Never claim to access a bank "
                    "or government portal. Never invent official rules. "
                    "Do not request OTPs, passwords, Aadhaar numbers, "
                    "or full bank credentials."
                )
            )
        )
        
        if not response.text:
            raise RuntimeError("Gemini returned no text. Please try again.")
            
        return response.text

def explain_rejection(pensioner_name, rejection_reason):
    prompt = f"""
    Explain this demo Life Certificate rejection in simple English.
    Pensioner name: {pensioner_name}
    Rejection reason: {rejection_reason}
    
    Include:
    1. Why the submission failed.
    2. What the mismatch means.
    3. What the pensioner should check.
    4. How to make a corrected resubmission.
    5. When to contact the bank or pension authority.
    
    This is only a software demonstration. Do not guarantee approval.
    Do not ask for confidential information.
    """
    return generate_ai_text(prompt)

def generate_pension_guidance(question):
    prompt = f"""
    A pensioner asks the following general question:
    {question}
    
    Answer in plain, respectful language. If the answer depends on
    official rules, a bank account, PPO status or a government portal,
    tell the person to confirm with their bank or pension authority.
    Do not invent a rule or claim to have checked their records.
    """
    return generate_ai_text(prompt)