import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
import os
from dotenv import load_dotenv
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import PyPDF2

# Configuración inicial de Streamlit
st.set_page_config(
    page_title="Análisis de Texto con LangGraph",
    page_icon="📊",
    layout="wide"
)

# Configuración del estado de la sesión
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Título principal
st.title("📊 Analizador de Texto con LangGraph")

# Cargar variables de entorno
load_dotenv()

# Verificar la clave API
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.error("⚠️ No se ha configurado la clave API de OpenAI. Por favor, configura la variable de entorno OPENAI_API_KEY.")
    st.stop()

# Definir la clase State
class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

# Inicializar el modelo con manejo de errores
try:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
except Exception as e:
    st.error(f"Error al inicializar el modelo: {str(e)}")
    st.stop()

# Definir los nodos de procesamiento
def classification_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories: Harvard Case, News, Blog, Science, TIC or Other.\n\nText:{text}\n\nCategory:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}

def entity_extraction_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location, Discipline) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}

def summarization_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in 500 words using Markdown, in Spanish.\n\nText:{text}\n\nSummary:\n\n"
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

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        return None

# Interfaz de usuario mejorada
st.markdown("### 📁 Sube tu archivo PDF")
uploaded_file = st.file_uploader("Elige un archivo PDF", type="pdf", key='pdf_uploader')

if uploaded_file is not None:
    try:
        with st.spinner('Procesando el archivo...'):
            # Extraer texto del PDF
            text_content = extract_text_from_pdf(uploaded_file)
            
            if text_content:
                # Mostrar el texto extraído
                with st.expander("Ver texto extraído"):
                    st.text(text_content)
                
                # Procesar el texto
                state_input = {"text": text_content}
                result = app.invoke(state_input)
                st.session_state.processed = True
                
                # Mostrar resultados en columnas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("### 📋 Clasificación")
                    st.info(result["classification"])
                
                with col2:
                    st.markdown("### 🏷️ Entidades")
                    for entity in result["entities"]:
                        st.write(f"- {entity}")
                
                with col3:
                    st.markdown("### 📝 Resumen")
                    st.markdown(result["summary"])
                
    except Exception as e:
        st.error(f"Error durante el procesamiento: {str(e)}")