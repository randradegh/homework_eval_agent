# Analizador de Texto Académico con LangGraph

## 📚 Descripción
Este proyecto es una herramienta en desarrollo diseñada para evaluar trabajos académicos de estudiantes de maestría en negocios. Utiliza tecnologías de procesamiento de lenguaje natural para analizar documentos PDF y proporcionar información estructurada sobre su contenido.

## 🎯 Funcionalidades Principales
- Clasificación automática del tipo de documento
- Extracción de entidades relevantes
- Generación de resúmenes
- Procesamiento de archivos PDF
- Interfaz web intuitiva

## 🛠️ Tecnologías Utilizadas
- Streamlit para la interfaz de usuario
- LangGraph para el flujo de trabajo de análisis
- OpenAI GPT para el procesamiento de lenguaje natural
- PyPDF2 para la extracción de texto de PDFs
- LangChain para la orquestación de modelos de lenguaje

## 📋 Requisitos Previos
- Python 3.8 o superior
- Clave API de OpenAI
- Dependencias listadas en `requirements.txt`

## 🚀 Instalación

1. Clonar el repositorio:
```bash
git clone [URL-del-repositorio]
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar variables de entorno:
Crear un archivo `.env` con:
```
OPENAI_API_KEY=tu-clave-api
```

## 💻 Uso
Para ejecutar la aplicación:
```bash
streamlit run st_langgraph_analisis.py
```

## 🔍 Características del Análisis
El sistema realiza tres tipos de análisis principales:
1. **Clasificación del Documento**: Identifica el tipo de documento (Harvard Case, News, Blog, Science, TIC, etc.)
2. **Extracción de Entidades**: Identifica personas, organizaciones, ubicaciones y disciplinas relevantes
3. **Generación de Resumen**: Crea un resumen estructurado del contenido

## 🔄 Flujo de Trabajo
El proceso de análisis sigue un flujo definido mediante LangGraph:

```79:90:st_langgraph_analisis.py
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
```


## 📊 Ejemplo de Uso
1. Acceder a la interfaz web
2. Subir un archivo PDF
3. Esperar el procesamiento automático
4. Revisar los resultados en las tres columnas:
   - Clasificación del documento
   - Entidades identificadas
   - Resumen generado

## ⚠️ Estado del Proyecto
Este proyecto está en desarrollo activo y se están implementando nuevas funcionalidades para mejorar la evaluación de trabajos académicos.

## 🤝 Contribuciones
Las contribuciones son bienvenidas. Por favor, crear un issue o pull request para sugerencias.

## 📝 Licencia
Este proyecto está bajo la Licencia MIT.
