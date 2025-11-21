import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Neo4jVector
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
import streamlit as st
import tempfile
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import operator

# Define the state for our graph
class GraphState(TypedDict):
    question: str
    cypher_query: str
    result: str
    error: str
    schema: str

def main():
    st.set_page_config(
        layout="wide",
        page_title="Graphy",
        page_icon=""
    )
    
    if os.path.exists('logo.png'):
        st.sidebar.image('logo.png', use_column_width=True)
    
    with st.sidebar.expander("About This App"):
        st.markdown("""
        **Graphy** - Built with LangGraph + Groq
        
        This application allows you to:
        - Upload PDF files
        - Extract content into Neo4j graph database
        - Query using natural language
        - Powered by Llama3 via Groq API
        - Uses LangGraph for workflow orchestration
        """)
    
    st.title("Graphy: GraphRAG with LangGraph")
    
    load_dotenv()

    # Set Groq API key
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        st.sidebar.subheader("Groq API Key")
        groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type='password')
    
    if groq_api_key:
        os.environ['GROQ_API_KEY'] = groq_api_key
        
        # Initialize Groq LLM with Llama3 if not already in session state or if key changed
        if 'llm' not in st.session_state or st.session_state.get('GROQ_API_KEY') != groq_api_key:
            st.session_state['GROQ_API_KEY'] = groq_api_key
            
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                groq_api_key=groq_api_key
            )
            
            # Use HuggingFace embeddings (free alternative)
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            st.session_state['embeddings'] = embeddings
            st.session_state['llm'] = llm
        else:
            llm = st.session_state['llm']
            embeddings = st.session_state['embeddings']
    else:
        st.warning("Please enter your Groq API Key in the sidebar.")
        st.stop()

    # Initialize Neo4j connection
    graph = None
    
    if 'neo4j_connected' not in st.session_state:
        neo4j_url = os.getenv('NEO4J_URI')
        neo4j_username = os.getenv('NEO4J_USERNAME')
        neo4j_password = os.getenv('NEO4J_PASSWORD')
        
        connect_button = False
        if not (neo4j_url and neo4j_username and neo4j_password):
            st.sidebar.subheader("🗄️ Connect to Neo4j Database")
            neo4j_url = st.sidebar.text_input("Neo4j URL:", value="neo4j+s://<your-neo4j-url>")
            neo4j_username = st.sidebar.text_input("Neo4j Username:", value="neo4j")
            neo4j_password = st.sidebar.text_input("Neo4j Password:", type='password')
            connect_button = st.sidebar.button("Connect")
        else:
            # Auto-connect if env vars are present
            connect_button = True
        
        if connect_button and neo4j_password:
            try:
                graph = Neo4jGraph(
                    url=neo4j_url,
                    username=neo4j_username,
                    password=neo4j_password
                )
                st.session_state['graph'] = graph
                st.session_state['neo4j_connected'] = True
                st.session_state['neo4j_url'] = neo4j_url
                st.session_state['neo4j_username'] = neo4j_username
                st.session_state['neo4j_password'] = neo4j_password
                if not (os.getenv('NEO4J_URI') and os.getenv('NEO4J_USERNAME') and os.getenv('NEO4J_PASSWORD')):
                    st.sidebar.success("✅ Connected to Neo4j database.")
            except Exception as e:
                st.error(f"❌ Failed to connect to Neo4j: {e}")
    else:
        graph = st.session_state['graph']
        neo4j_url = st.session_state['neo4j_url']
        neo4j_username = st.session_state['neo4j_username']
        neo4j_password = st.session_state['neo4j_password']

    # File uploader
    if graph is not None:
        uploaded_file = st.file_uploader("📄 Please select a PDF file", type="pdf")

        if uploaded_file is not None and 'graph_workflow' not in st.session_state:
            with st.spinner("🔄 Processing the PDF..."):
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # Load and split PDF
                loader = PyMuPDFLoader(tmp_file_path)
                pages = loader.load_and_split()

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=200,
                    chunk_overlap=40
                )
                docs = text_splitter.split_documents(pages)

                lc_docs = [
                    Document(
                        page_content=doc.page_content.replace("\n", ""),
                        metadata={'source': uploaded_file.name}
                    ) for doc in docs
                ]

                # Clear graph database
                graph.query("MATCH (n) DETACH DELETE n;")

                # Define schema
                allowed_nodes = ["Patient", "Disease", "Medication", "Test", "Symptom", "Doctor"]
                allowed_relationships = [
                    "HAS_DISEASE", "TAKES_MEDICATION", "UNDERWENT_TEST",
                    "HAS_SYMPTOM", "TREATED_BY"
                ]

                # Transform to graph
                transformer = LLMGraphTransformer(
                    llm=llm,
                    allowed_nodes=allowed_nodes,
                    allowed_relationships=allowed_relationships,
                    node_properties=False,
                    relationship_properties=False
                )

                graph_documents = transformer.convert_to_graph_documents(lc_docs)
                graph.add_graph_documents(graph_documents, include_source=True)

                # Create vector index
                index = Neo4jVector.from_existing_graph(
                    embedding=embeddings,
                    url=neo4j_url,
                    username=neo4j_username,
                    password=neo4j_password,
                    database="neo4j",
                    node_label="Patient",
                    text_node_properties=["id", "text"],
                    embedding_node_property="embedding",
                    index_name="vector_index",
                    keyword_index_name="entity_index",
                    search_type="hybrid"
                )

                # Build LangGraph workflow
                workflow = create_langgraph_workflow(llm, graph)
                st.session_state['graph_workflow'] = workflow
                st.session_state['schema'] = graph.get_schema
                
                st.success(f"✅ {uploaded_file.name} processed successfully!")
                os.unlink(tmp_file_path)
    else:
        st.warning("⚠️ Please connect to Neo4j database first.")

    # Question answering interface
    if 'graph_workflow' in st.session_state:
        st.subheader("💬 Ask a Question")
        
        with st.form(key='question_form'):
            question = st.text_input("Enter your question:")
            submit_button = st.form_submit_button(label='🚀 Submit')

        if submit_button and question:
            with st.spinner("🤔 Generating answer..."):
                try:
                    workflow = st.session_state['graph_workflow']
                    schema = st.session_state['schema']
                    
                    # Execute workflow
                    initial_state = {
                        "question": question,
                        "schema": schema,
                        "cypher_query": "",
                        "result": "",
                        "error": ""
                    }
                    
                    final_state = workflow.invoke(initial_state)
                    
                    if final_state.get("error"):
                        st.error(f"❌ Error: {final_state['error']}")
                    else:
                        st.write("**🔍 Generated Cypher Query:**")
                        st.code(final_state.get("cypher_query", ""), language="cypher")
                        st.write("**✨ Answer:**")
                        st.write(final_state.get("result", "No result returned."))
                        
                except Exception as e:
                    st.error(f"❌ Error processing question: {str(e)}")


def create_langgraph_workflow(llm, graph):
    """Create a LangGraph workflow for Cypher query generation and execution"""
    
    def generate_cypher(state: GraphState) -> GraphState:
        """Generate Cypher query from natural language question"""
        prompt = f"""Task: Generate a Cypher statement to query the graph database.

Instructions:
- Use only relationship types and properties provided in schema
- Do not use other relationship types or properties that are not provided
- Return ONLY the Cypher query, no explanations or markdown formatting

Schema:
{state['schema']}

Question: {state['question']}

Cypher Query:"""
        
        try:
            response = llm.invoke(prompt)
            cypher_query = response.content.strip()
            
            # Clean up the query
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
            
            return {
                **state,
                "cypher_query": cypher_query,
                "error": ""
            }
        except Exception as e:
            return {
                **state,
                "error": f"Error generating Cypher: {str(e)}"
            }
    
    def execute_cypher(state: GraphState) -> GraphState:
        """Execute the generated Cypher query"""
        if state.get("error"):
            return state
            
        try:
            result = graph.query(state["cypher_query"])
            
            # Format result
            if result:
                formatted_result = format_graph_result(result)
            else:
                formatted_result = "No results found."
                
            return {
                **state,
                "result": formatted_result,
                "error": ""
            }
        except Exception as e:
            return {
                **state,
                "error": f"Error executing Cypher: {str(e)}"
            }
    
    def format_result(state: GraphState) -> GraphState:
        """Format the final result for presentation"""
        if state.get("error"):
            return state
        
        # If result needs additional formatting with LLM
        if state.get("result") and not state.get("error"):
            prompt = f"""Given the following query result, provide a clear and concise answer to the question.

Question: {state['question']}

Query Result: {state['result']}

Answer:"""
            
            try:
                response = llm.invoke(prompt)
                return {
                    **state,
                    "result": response.content.strip()
                }
            except Exception as e:
                # If formatting fails, return original result
                return state
        
        return state
    
    # Build the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("generate_cypher", generate_cypher)
    workflow.add_node("execute_cypher", execute_cypher)
    workflow.add_node("format_result", format_result)
    
    # Add edges
    workflow.set_entry_point("generate_cypher")
    workflow.add_edge("generate_cypher", "execute_cypher")
    workflow.add_edge("execute_cypher", "format_result")
    workflow.add_edge("format_result", END)
    
    # Compile the graph
    return workflow.compile()


def format_graph_result(result):
    """Format graph query results for display"""
    if not result:
        return "No results found."
    
    formatted = []
    for record in result:
        if isinstance(record, dict):
            formatted.append(", ".join(f"{k}: {v}" for k, v in record.items()))
        else:
            formatted.append(str(record))
    
    return "\n".join(formatted)


if __name__ == "__main__":
    main()