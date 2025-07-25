from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
import sys

# Add the project root to the sys.path to allow importing rag_code
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from rag_code.llm import BengaliRAGSystem
    from rag_code.db import BengaliVectorStore
except ImportError as e:
    print(f"Failed to import RAG modules: {e}")
    print(f"Python path: {sys.path}")

# Initialize the RAG system globally to avoid re-loading on each request
# Adjust the path to your vector database as needed
try:
    vector_store_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'bengali_vector_db'))
    bengali_vector_store = BengaliVectorStore().load_local(vector_store_path)
    rag_system = BengaliRAGSystem(bengali_vector_store)
    rag_system_initialized = True
except Exception as e:
    print(f"Error initializing RAG system: {e}")
    rag_system_initialized = False
    rag_system = None

@csrf_exempt
def chat(request):
    if not rag_system_initialized:
        return JsonResponse({'error': 'RAG system not initialized. Check server logs.'}, status=500)

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message')
            if message:
                try:
                    response = rag_system.ask(message)
                    return JsonResponse({'response': response['answer'], 'sources': response['sources']})
                except Exception as e:
                    return JsonResponse({'error': f'Error processing message with RAG system: {e}'}, status=500)
            else:
                return JsonResponse({'error': 'No message provided'}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
    else:
        return JsonResponse({'error': 'Only POST requests are accepted'}, status=405)