import ollama

def ask_ai(prompt):
    """
    Sends a prompt to the AI model and returns the response.
    """
    try:
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"
