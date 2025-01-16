#!/usr/bin/env python
# coding: utf-8

from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage

# Cargar variables de entorno
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Definir la clase State
class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

# Inicializar el modelo
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Definir los nodos de procesamiento
def classification_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories: News, Blog, Physics, Chemistry, TIC or Other.\n\nText:{text}\n\nCategory:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}

def entity_extraction_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location, Chemical Molecules) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}

def summarization_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in 300 words using Markdown.\n\nText:{text}\n\nSummary:\n\n"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}

# Crear y configurar el flujo de trabajo
workflow = StateGraph(State)

# Agregar nodos al grafo
workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)

# Agregar bordes al grafo
workflow.set_entry_point("classification_node")
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

# Compilar el grafo
app = workflow.compile()

# Texto de ejemplo
sample_text = """
Tipos de enlace químico
Existen tres tipos de enlace químico conocidos, dependiendo de la naturaleza de los átomos involucrados:

Enlace covalente. Ocurre entre átomos no metálicos y de cargas electromagnéticas semejantes (por lo general altas), que se unen y comparten algunos pares de electrones de su capa de valencia. Es el tipo de enlace predominante en las moléculas orgánicas y puede ser de tres tipos: simple (A-A), doble (A=A) y triple (A≡A), dependiendo de la cantidad de electrones compartidos.
Enlace iónico. Consiste en la atracción electrostática entre partículas con cargas eléctricas de signos contrarios llamadas iones (partícula cargada eléctricamente, que puede ser un átomo o molécula que ha perdido o ganado electrones, es decir, que no es neutro).
Enlace metálico. Se da únicamente entre átomos metálicos de un mismo elemento, que por lo general constituyen estructuras sólidas, sumamente compactas. Es un enlace fuerte, que une los núcleos atómicos entre sí, rodeados de sus electrones como en una nube.
Ejemplos de enlace químico
Algunos ejemplos de compuestos con enlace covalente:

Benceno (C6H6)
Metano (CH4)
Glucosa (C6H12O6)
Amoníaco (NH3)
Freón (CFC)
En todas las formas del carbono (C): carbón, diamantes, grafeno, etc.
Algunos ejemplos de compuestos con enlace iónico:

Óxido de magnesio (MgO)
Sulfato de cobre (CuSO4)
Ioduro de potasio (KI)
Cloruro de manganeso (MnCl2)
Carbonato de calcio (CaCO3)
Sulfuro de hierro (Fe2S3)
Algunos ejemplos de compuestos con enlace metálico:

Barras de hierro (Fe)
Yacimientos de cobre (Cu)
Barras de oro puro (Au)
Barras de plata pura (Ag)


Fuente: https://concepto.de/enlace-quimico/#ixzz8xQjCywVK
"""

# Ejecutar el pipeline
if __name__ == "__main__":
    state_input = {"text": sample_text}
    result = app.invoke(state_input)
    
    print("Classification:", result["classification"])
    print("\nEntities:", result["entities"])
    print("\nSummary:", result["summary"])
