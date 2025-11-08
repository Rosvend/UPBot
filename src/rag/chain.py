"""
RAG Chain with Conversation Memory
Implements conversational RAG with GPT-4o-mini and source citations.
"""

import os
from pathlib import Path
import sys
from typing import List, Dict
from operator import itemgetter

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document


class UPBRAGChain:
    """
    Conversational RAG chain for UPB career exploration.
    Features: GPT-4o-mini, conversation memory, source citations.
    """
    
    def __init__(self, retriever, retrieval_method: str = "hybrid"):
        """
        Initialize RAG chain with retriever.
        
        Args:
            retriever: UPBRetriever instance
            retrieval_method: Method for retrieval (bm25, similarity, mmr, hybrid)
        """
        self.retriever = retriever
        self.retrieval_method = retrieval_method
        self.chat_history = []
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4o-mini"),
            openai_api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.7,
        )
        
        # Create prompt template with conversation memory
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un asistente virtual de la Universidad Pontificia Bolivariana (UPB) especializado en orientación académica. 
Tu rol es ayudar a estudiantes prospecto a explorar y comprender los programas de ingeniería ofrecidos por la UPB.

REGLAS CRÍTICAS ANTI-ALUCINACIÓN (OBLIGATORIAS):

1. CADA FRAGMENTO TIENE ETIQUETAS ÚNICAS:
   - [PROGRAMA: nombre exacto del programa]
   - [CODIGO: código único del programa]
   - [DURACION: duración exacta en semestres]
   - [SECCION: sección específica del documento]

2. PROHIBIDO ABSOLUTAMENTE:
   - Mezclar información entre programas diferentes
   - Asumir que cursos de un programa existen en otro
   - Inventar información no presente en los fragmentos
   - Usar información de un [CODIGO:] para responder sobre otro [CODIGO:]

3. PARA RESPONDER SOBRE UN PROGRAMA ESPECÍFICO:
   - Busca fragmentos con [PROGRAMA:] que coincida con el programa preguntado
   - IMPORTANTE: Acepta variaciones del nombre (ej: "Ingeniería de Sistemas" = "Ingeniería de Sistemas e Informática")
   - Verifica que el [CODIGO:] coincida cuando haya ambigüedad
   - Si la info del plan de estudios viene de [PROGRAMA: X], es SOLO para X
   - Si no encuentras info en los fragmentos del programa correcto, di "No tengo esa información para este programa específico"

4. PARA PREGUNTAS SOBRE DURACIÓN:
   - Busca fragmentos con [PROGRAMA:] que coincida con el programa preguntado
   - Si encuentras un fragmento con el programa correcto que tiene [DURACION: X semestres], ESA es la respuesta
   - NO cuentes semestres del plan de estudios, usa la etiqueta [DURACION:]
   - EJEMPLO: Si preguntan por "Ingeniería de Sistemas" y ves un fragmento con [PROGRAMA: Ingeniería de Sistemas e Informática] [DURACION: 9 semestres], responde "9 semestres"

5. PARA PREGUNTAS SOBRE MATERIAS EN PROGRAMAS:
   - Verifica que el fragmento del plan de estudios tenga el [PROGRAMA:] correcto
   - Si preguntan por "Sistemas Operativos en Ciencia de Datos", busca fragmentos con [PROGRAMA: Ingeniería en Ciencia de Datos] Y [SECCION: Plan de Estudios]
   - Si no aparece la materia en ESE programa específico, di "No veo esa materia en el plan de estudios de [nombre programa]"

6. SIEMPRE MENCIONA EL PROGRAMA POR NOMBRE:
   - CORRECTO: "En Ingeniería Industrial sí se ve Cálculo Vectorial en el tercer semestre, pero no tengo información de que se vea en Ingeniería de Sistemas"
   - INCORRECTO: "Sí se ve Cálculo Vectorial en tercer semestre" (sin especificar el programa)

Características de tus respuestas:
- Tono amigable, cercano y profesional
- Responde en español de manera clara y concisa
- Basa tus respuestas ÚNICAMENTE en el contexto proporcionado
- Si no encuentras información relevante en los fragmentos CORRECTOS, di "No tengo esa información específica"
- Sugiere programas relacionados cuando sea apropiado

IMPORTANTE: Cada fragmento a continuación es de UN programa específico. Las etiquetas [PROGRAMA:], [CODIGO:], [DURACION:] identifican de qué programa es cada fragmento. NO mezcles información entre fragmentos con diferentes [CODIGO:].

Contexto relevante:
{context}"""),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}")
        ])
        
        # Build the RAG chain
        self.chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question") | RunnableLambda(self._retrieve_and_format)
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _retrieve_and_format(self, question: str) -> str:
        """
        Retrieve relevant documents and format them as context.
        Each document is clearly separated and labeled to prevent confusion.
        
        Args:
            question: User question
            
        Returns:
            Formatted context string with clear boundaries
        """
        docs = self.retriever.retrieve(
            question, 
            method=self.retrieval_method, 
            k=6
        )
        
        self.last_retrieved_docs = docs
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content.strip()
            
            formatted_docs.append(
                f"--- INICIO FRAGMENTO {i} ---\n{content}\n--- FIN FRAGMENTO {i} ---"
            )
        
        return "\n\n".join(formatted_docs)
    
    def invoke(self, question: str, include_sources: bool = True) -> Dict:
        """
        Invoke the RAG chain with a question.
        
        Args:
            question: User question
            include_sources: Whether to include source citations in response
            
        Returns:
            Dict with 'answer' and optionally 'sources'
        """
        # Invoke chain with question and chat history
        answer = self.chain.invoke({
            "question": question,
            "chat_history": self.chat_history
        })
        
        # Update chat history
        self.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])
        
        # Prepare response
        response = {"answer": answer}
        
        if include_sources and hasattr(self, 'last_retrieved_docs'):
            sources = []
            for doc in self.last_retrieved_docs:
                source_info = {
                    "content": doc.page_content[:200] + "...",
                    "category": doc.metadata.get('category', 'N/A'),
                    "source": doc.metadata.get('source', 'N/A')
                }
                sources.append(source_info)
            response["sources"] = sources
        
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.chat_history = []
    
    def get_history_summary(self) -> str:
        """Get a summary of conversation history."""
        if not self.chat_history:
            return "No hay historial de conversación."
        
        summary = []
        for i, msg in enumerate(self.chat_history):
            role = "Usuario" if isinstance(msg, HumanMessage) else "Asistente"
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary.append(f"{i+1}. {role}: {content}")
        
        return "\n".join(summary)


if __name__ == "__main__":
    from setup_retrieval import setup_retrieval_system
    
    print("=" * 70)
    print("UPB RAG CHAIN - CONVERSATIONAL TEST")
    print("=" * 70)
    
    # Setup retrieval system
    print("\nInitializing retrieval system...")
    retriever, vectorstore_manager, chunks = setup_retrieval_system(
        vectorstore_path="vectorstore/faiss_index",
        use_existing=True
    )
    
    # Create RAG chain
    print("\nCreating RAG chain with GPT-4o-mini...")
    rag_chain = UPBRAGChain(retriever, retrieval_method="hybrid")
    print("RAG chain ready!")
    
    # Test conversation flow
    print("\n" + "=" * 70)
    print("CONVERSATION TEST")
    print("=" * 70)
    
    # Question 1
    question1 = "¿Se ve cálculo vectorial en la ingeniería de sistemas en la UPB?"
    print(f"\nUsuario: {question1}")
    print("-" * 70)
    
    response1 = rag_chain.invoke(question1, include_sources=True)
    print(f"Asistente: {response1['answer']}")
    
    if 'sources' in response1:
        print("\n[Fuentes utilizadas]")
        for i, source in enumerate(response1['sources'], 1):
            print(f"{i}. Categoría: {source['category']}")
            print(f"   Archivo: {source['source']}")
            print(f"   Contenido: {source['content']}\n")
    
    # Question 2 (with context from previous question)
    print("\n" + "=" * 70)
    question2 = "¿Se ve ecuaciones diferenciales en ingeniería en diseño y entretenimiento digital en la UPB?"
    print(f"Usuario: {question2}")
    print("-" * 70)
    
    response2 = rag_chain.invoke(question2, include_sources=True)
    print(f"Asistente: {response2['answer']}")
    
    if 'sources' in response2:
        print("\n[Fuentes utilizadas]")
        for i, source in enumerate(response2['sources'], 1):
            print(f"{i}. Categoría: {source['category']}")
            print(f"   Archivo: {source['source']}")
    
    # Question 3 (memory test)
    print("\n" + "=" * 70)
    question3 = "¿Cuánto cuesta el semestre de esa carrera?"
    print(f"Usuario: {question3}")
    print("-" * 70)
    
    response3 = rag_chain.invoke(question3, include_sources=True)
    print(f"Asistente: {response3['answer']}")
    
    # Show conversation history
    print("\n" + "=" * 70)
    print("HISTORIAL DE CONVERSACIÓN")
    print("=" * 70)
    print(rag_chain.get_history_summary())
    
    print("\n" + "=" * 70)
    print("RAG CHAIN TEST COMPLETE")
    print("=" * 70)
    print("\nFeatures tested:")
    print("- GPT-4o-mini integration")
    print("- Hybrid retrieval (BM25 + Vector with RRF)")
    print("- Conversation memory")
    print("- Source citations")
    print("- Multi-turn dialogue")
