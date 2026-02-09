import ollama

SYSTEM_PROMPT = """Du bist ein hilfreicher Assistent für AllCo Mitarbeiter. 
Deine Aufgabe ist es, Fragen zu internen Personalabteilungs-Prozessen und -Verfahren zu beantworten.

Wichtige Regeln:
- Antworte NUR basierend auf den bereitgestellten Dokumenten-Ausschnitten
- Wenn die Antwort nicht in den Dokumenten steht, sage ehrlich "Diese Information finde ich nicht in den verfügbaren Dokumenten"
- Antworte immer auf Deutsch
- Sei präzise und professionell
- Wenn du dir unsicher bist, gib das zu
"""

def generate_answer(question, search_results, top_n=3):
    # Extract chunks inside the function
    chunks = search_results['documents'][0][:top_n]  # Top n chunks
    # Format context nicely
    context = "Verfügbare Dokument-Ausschnitte:\n\n"
    for i, chunk in enumerate(chunks, 1):
        context += f"Ausschnitt {i}:\n{chunk}\n\n"
    # Build user message with context + question
    user_message = context + f"Frage: {question}"
    # Call LLM
    response = ollama.chat(
    model='llama3.1',
    messages=[
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_message}
        ]
    )
    return response['message']['content']  # Return only the content of the response message




