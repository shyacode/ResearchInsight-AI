import ollama

def analyze_paper(text):

    prompt = f"""
    Analyze this research paper and provide:

    1. Problem statement
    2. Key contributions
    3. Methodology explained simply
    4. Strengths
    5. Limitations
    6. Possible future work

    Paper:
    {text}
    """

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]