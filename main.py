# backend/main.py

import os
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from typing import Dict, Any, List
from backend.tools.search_job_offers import search_job_offers
from backend.tools.provide_cv_advice import provide_cv_advice
from backend.tools.get_career_advice import get_career_advice
from backend.tools.generate_cover_letter import generate_cover_letter
from backend.config import settings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import ToolException

# Initialisation de Flask
app = Flask(__name__)

# Initialisation du modèle GPT-4 via LangChain
model = ChatOpenAI(model="gpt-4")

# Liste des outils créés et importés
tools = [
    search_job_offers,
    provide_cv_advice,
    get_career_advice,
    generate_cover_letter
]

# Liaison des outils au modèle
llm_with_tools = model.bind_tools(tools)

# Définition du template de prompt avec instructions personnalisées et outils
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Vous êtes un assistant virtuel de Pôle Emploi. Vous aidez les chercheurs d'emploi à trouver des offres, à rédiger des CV et des lettres de motivation, et à fournir des conseils de carrière. Utilisez les outils disponibles pour améliorer vos réponses."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Définition du trimmer pour gérer l'historique des messages
from langchain_core.messages import trim_messages

trimmer = trim_messages(
    max_tokens=1000,  # Ajustez selon les besoins et la limite de tokens du modèle
    strategy="last",
    token_counter=len,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Définition de l'état des conversations
from typing import Sequence
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# Définition de la fonction de rappel avec trimmer et prompt
def call_model_with_tools(state: State):
    # Trimer les messages pour respecter la limite de tokens
    trimmed_messages = trimmer.invoke(state["messages"])
    # Créer la chaîne de prompt avec les messages trimmés
    chain = prompt | llm_with_tools
    # Invoker la chaîne LangChain avec les messages trimmés
    response = chain.invoke({"messages": trimmed_messages, "language": state.get("language", "French")})
    return {"messages": [response]}

# Définition du workflow avec LangGraph
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model_with_tools")
workflow.add_node("model_with_tools", call_model_with_tools)

# Ajout de la mémoire avec MemorySaver
memory = MemorySaver()
app_langchain = workflow.compile(checkpointer=memory)

# Fonction pour exécuter les outils appelés par LangChain
def execute_tool(tool_call: Dict[str, Any]) -> List[str]:
    try:
        tool_name = tool_call['name']
        args = tool_call['args']
        
        if tool_name == 'search_job_offers':
            # Extraire les paramètres nécessaires
            return search_job_offers(**args)
        
        elif tool_name == 'provide_cv_advice':
            return [provide_cv_advice(**args)]
        
        elif tool_name == 'get_career_advice':
            return [get_career_advice(**args)]
        
        elif tool_name == 'generate_cover_letter':
            return [generate_cover_letter(**args)]
        
        else:
            return [f"Aucun outil défini pour le nom {tool_name}."]
    
    except ToolException as te:
        return [str(te)]
    except Exception as e:
        return [f"Une erreur est survenue lors de l'exécution de l'outil: {e}"]

# Route principale pour le chat
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    thread_id = data.get('thread_id', 'pole_emploi_default')
    language = data.get('language', 'French')
    user_name = data.get('user_name', 'Utilisateur')  # Optionnel

    if not query:
        return jsonify({"error": "Le champ 'query' est requis."}), 400

    # Créer les messages d'entrée
    input_messages = [HumanMessage(content=query)]
    state = {"messages": input_messages, "language": language}
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Invoker LangChain avec l'état et la configuration
        output = app_langchain.invoke(state, config)
        ai_message = output["messages"][-1].content

        # Traitement des appels aux outils
        tool_responses = []
        last_message = output["messages"][-1]
        if hasattr(last_message, 'tool_calls'):
            for tool_call in last_message.tool_calls:
                result = execute_tool(tool_call)
                tool_responses.extend(result)

        return jsonify({
            "response": ai_message,
            "tools": tool_responses
        }), 200

    except ToolException as te:
        return jsonify({"error": str(te)}), 500
    except Exception as e:
        return jsonify({"error": f"Une erreur inattendue est survenue: {e}"}), 500

# Point d'entrée de l'application
if __name__ == '__main__':
    # Configuration pour lire les variables d'environnement depuis .env
    from dotenv import load_dotenv
    load_dotenv()

    # Vérifier que les clés API sont définies
    if not settings.FRANCETRAVAIL_API_KEY:
        raise ValueError("La clé API Francetravail n'est pas définie. Veuillez la définir dans les variables d'environnement.")

    if not settings.OPENAI_API_KEY:
        raise ValueError("La clé API OpenAI n'est pas définie. Veuillez la définir dans les variables d'environnement.")

    # Lancer l'application Flask
    app.run(host='0.0.0.0', port=5000)
