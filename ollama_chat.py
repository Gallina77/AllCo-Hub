from backend.llm_generation import generate_answer
from backend.vector_store import create_client, search_in_chromadb

# # Optional: Test search
collection_name = "hr_procedures"
client = create_client()  # Create client once at startup
#query = "Was müssen wir halbjährlich machen?"
query = "Wie melde ich mich krank?"  # ← New query for testing
print(f"\nSearching for: '{query}'")
search_results = search_in_chromadb(query, n_results=5, collection_name=collection_name, client=client)

answer = generate_answer(query, search_results)
print("Ollama reply:", answer)  