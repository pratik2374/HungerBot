import streamlit as st
import pandas as pd
from openai import OpenAI
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core import Document
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, AgentStream
from llama_index.core.workflow import Context
import os
import asyncio
import matplotlib.pyplot as plt
from llama_index.core.tools import FunctionTool
import seaborn as sns
import io
import re
import datetime
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import glob
import logging
logging.getLogger("httpx").setLevel(logging.CRITICAL)
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("torch._classes").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module="torch")
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import rapidfuzz
from rapidfuzz import process as rapidfuzz_process 

from graph import gen_plot

# Set OpenAI API key from secrets
os.environ['OPENAI_API_KEY'] = st.secrets["openai_api_key"]
client = OpenAI()

if "history_count" not in st.session_state:
    st.session_state.history_count = 0

# Set page config
st.set_page_config(
    page_title="Data Analysis Chatbot",
    page_icon="🤖",
    layout="centered"  # Options are: "centered", "wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [] # For streamlit chat interface
    st.session_state.chat_history = [] # For keeping chat history

# Title and description
st.title("📊 HungerBot v1806")
st.markdown("Ask questions about your data and get instant insights!")

# Sidebar instructions
with st.sidebar:
    st.header("📁 Data Upload")
    st.markdown("""
    ### Instructions:
    1. Upload your CSV or Excel file using the file uploader below
    2. Supported formats: CSV (.csv) and Excel (.xls, .xlsx)
    """)

    st.sidebar.title("💡 Example Prompts")
    example_prompts = [
        "List the top five vendors by orders",
        "Make a bar graph on sales with order date",
        "Peak sales hours of April 28, 2025",
        "Make a pie chart with vendor_id and sales",
        "Give me the least sold item name by orders"
    ]

    for prompt in example_prompts:
        st.sidebar.code(prompt, language='')  # empty language = plain text

    file = st.file_uploader(
        "Upload your data file",
        type=["csv", "xls", "xlsx"],
        help="Upload a CSV or Excel file to analyze"
    )
if "df_loaded" not in st.session_state:
    st.session_state.df_loaded = False
# --- NEW: List available datasets in the 'dataset' folder ---
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)
dataset_files = glob.glob(os.path.join(DATASET_DIR, "*.csv")) + glob.glob(os.path.join(DATASET_DIR, "*.xlsx")) + glob.glob(os.path.join(DATASET_DIR, "*.xls"))

if (not dataset_files) and (file is None):
    st.info("No datasets found in the 'dataset' folder. Please upload a dataset to continue.")
else : 
    def preview_data(df):
        st.sidebar.success(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        st.sidebar.markdown("### Data Preview")
        st.sidebar.dataframe(df.head(3))
    def save_csv(df):
        if not st.session_state.df_loaded:
            now_str = datetime.datetime.now().strftime("%d%m%H%M")
            save = f"dataset/HBdata_{now_str}.csv"
            df.to_csv(save, index=False)

    def load_csv():
        """
        Load a CSV or Excel file into a pandas DataFrame.
        If an uploaded file is provided, use it; otherwise, use a file from the dataset directory.
        """
        uploaded_file = file
        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file, low_memory=False)
                # Save to dataset folder
                save_csv(df)
                preview_data(df)
            elif file_extension in ['xls', 'xlsx']:
                df = pd.read_excel(uploaded_file)
                # Save to dataset folder
                save_csv(df)
                preview_data(df)
            else:
                st.error("Unsupported uploaded file format.")
                st.stop()
            st.session_state.df_loaded = True
            df = preprocess_datetime_columns(df)
            return df
        else:
            # Let user select a dataset if more than one
            if len(dataset_files) == 1:
                selected_file = dataset_files[0]
            else:
                selected_file = st.selectbox("Select a dataset to analyze", dataset_files)
            file_extension = selected_file.split('.')[-1].lower()
            if file_extension == 'csv':
                df = pd.read_csv(selected_file, low_memory=False)
                preview_data(df)
            elif file_extension in ['xls', 'xlsx']:
                df = pd.read_excel(selected_file)
                preview_data(df)
            else:
                st.error("Unsupported file format in dataset folder.")
                st.stop()
            st.session_state.df_loaded = True
            df = preprocess_datetime_columns(df)
            return df
            
    # --- END: Load the selected dataset ---

    # ==== TO DETECT DATETIME column ==========
    def preprocess_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically detect and convert datetime columns to proper datetime format.
        """
        df_processed = df.copy()
        datetime_keywords = ['date', 'time', 'created', 'updated', 'modified', 'timestamp']
        
        for col in df.columns:
            # Check if column name suggests it's a datetime
            if any(keyword in col.lower() for keyword in datetime_keywords):
                try:
                    # Try to convert to datetime
                    df_processed[col] = pd.to_datetime(df[col], errors='coerce')
                    if df_processed[col].notna().any():  # If conversion successful for some values
                        print(f"✅ Converted '{col}' to datetime format")
                    else:
                        print(f"⚠️ Could not convert '{col}' to datetime - keeping original format")
                        df_processed[col] = df[col]  # Revert if conversion failed
                except Exception as e:
                    print(f"⚠️ Could not convert '{col}' to datetime: {str(e)}")
                    df_processed[col] = df[col]  # Keep original if conversion fails
        
        return df_processed

    # def data_profiling(df ):
    #     main_df = preprocess_datetime_columns(df)

    df = load_csv()
    # -- NEW: Initialize the PandasQueryEngine and vector Store Engine with the loaded DataFrame --
    query_engine = PandasQueryEngine(
        df=df,
        verbose=True,
        synthesize_response=True,
        llm=OpenAI(model="gpt-4o")
        )

    def panda_retriver(query: str) -> str:
        """
            Run a natural language query on the DataFrame using PandasQueryEngine.

            Steps:
            1. Convert the query to pandas operations
            2. Run them on df
            3. Return a readable response

            Args:
                query (str): A question about the data (e.g., stats, filters)

            Returns:
                str: A summary with results and explanation

            Note:
            - The query should be related to the data in the loaded DataFrame
        """

        ## Todo : Subquestion Query to retrive from all
        response = query_engine.query(query)
        return str(response)


    tools = [panda_retriver]

    def get_column_summary(df: pd.DataFrame, n_examples: int = 2) -> str:
        """Generate a summary of columns, types, and example values for prompt."""
        summary = []
        for col in df.columns:
            col_type = str(df[col].dtype)
            examples = df[col].dropna().unique()[:n_examples]
            summary.append(
                f'- "{col}": type={col_type}, examples={list(examples)}'
            )
        return "\n".join(summary)

    def get_column_names(df: pd.DataFrame) -> str:
        """Return a comma-separated string of column names."""
        return ", ".join([f'"{col}"' for col in df.columns])

    def is_visualization_query(query: str) -> bool:
        """Detect if the query is asking for a visualization."""
        viz_keywords = [
            "plot", "chart", "graph", "visualize", "visualization", "draw", "histogram", "bar", "line", "scatter", "pie", "heatmap"
        ]
        return any(word in query.lower() for word in viz_keywords)

    def get_entity_related_columns(df: pd.DataFrame):
        """
        Dynamically map common entities to their related columns based on column names.
        """
        entity_map = {}
        colnames = [col.lower() for col in df.columns]
        
        # Example for product
        product_cols = [col for col in df.columns if any(key in col.lower() for key in ["product", "item"])]
        if product_cols:
            entity_map["product"] = product_cols
            
        # Example for vendor
        vendor_cols = [col for col in df.columns if "vendor" in col.lower()]
        if vendor_cols:
            entity_map["vendor"] = vendor_cols
            
        # Enhanced customer/user detection
        customer_cols = [col for col in df.columns if any(key in col.lower() for key in ["customer", "user", "buyer", "client"])]
        if customer_cols:
            entity_map["customer/user"] = customer_cols
            
        # Example for order with time columns
        order_cols = [col for col in df.columns if "order" in col.lower()]
        if order_cols:
            entity_map["order"] = order_cols
            
        # Time-related columns for peak analysis
        time_cols = [col for col in df.columns if any(key in col.lower() for key in ["time", "created", "date", "timestamp"])]
        if time_cols:
            entity_map["time"] = time_cols
            entity_map["date"] = time_cols
            
        # Example for location
        location_cols = [col for col in df.columns if "location" in col.lower()]
        if location_cols:
            entity_map["location"] = location_cols
            
        return entity_map

    # In your main prompt, dynamically describe the data:
    def get_system_prompt(df: pd.DataFrame) -> str:
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entity_map = get_entity_related_columns(df)
        entity_instructions = ""
        for entity, cols in entity_map.items():
            entity_instructions += (
                f"- When a user asks about a {entity}, always include these related columns in your answer: {', '.join([f'\"{c}\"' for c in cols])}. "
                f"If you need to look up an ID or code, always resolve it to its full details using all related columns in a single answer. "
                f"If the answer requires multiple steps (e.g., find the most sold {entity}, then fetch its details), perform all steps and return a complete answer in one response.\n"
            )
        return f"""
    You are HungerBox's advanced Data Analysis Assistant, designed to help users analyze and gain insights from complex sales and operational data. You are integrated into a Retrieval-Augmented Generation (RAG) system and have access to the user's uploaded data as a pandas DataFrame named df.

    Current date and time: {current_datetime}

    Your core responsibilities:
    - *Data Fidelity:* Always use the provided DataFrame df for all analysis, calculations, and visualizations. Never create or load a new DataFrame or use sample data.
    - *Column Awareness:* Only use the actual column names and data types present in df. If you are unsure about a column name, refer to the provided list of columns.
    - *Entity Context:*\n {entity_instructions}\n
    - *Ambiguity Handling:*
    - If a user asks about a product, vendor, customer, or any entity and the name is ambiguous or matches multiple items, always return a table listing *all similar or matching items* (not just the top match).
    - For each matching item, provide all relevant details: product/vendor/customer ID, name, total sales (in ₹), percentage of total sales, and any other associated columns (e.g., quantity sold, number of orders, etc.).
    - *If the entity name (e.g., item name) is enclosed in quotation marks (""), perform an exact match. If not, use case-insensitive substring matching to find similar names.*
    - Always include a summary explaining the results and their business significance.

    ✅ *Example:*
    - User query: Show me sales for Samosa
        - Match all item names that *contain* the word “Samosa”, such as:
        - Veg Samosa
        - Cheese Samosa
        - Samosa Chat
        - Samosa
    - User query: Show me sales for "Samosa"
        - Match only the item *exactly* named Samosa.

    - *Business Context:* Assume the data is related to sales, orders, transactions, customers, vendors, products, payments, and other business operations typical for a food-tech company like HungerBox. However, always adapt to the actual columns and data provided.
    - *Directness:* Always answer the user's question directly and concisely. If the question is ambiguous or cannot be answered, ask for clarification or explain why.
    

    - *Details:* Always answer the user's question directly, *with a detailed, step-by-step explanation* of your reasoning and findings. For every response, include a clear summary of what the result means in a business context, and how the user can interpret or act on it.

    - *Visualization:* - If a query involves to make a graph then just provide summary with the help of pandas query engine tool, no need to draw any visualization or write any code for that

    - *Analysis:*
    - Support a wide range of queries: aggregations, trends, comparisons, filtering, segmentation, cohort analysis, time series, outlier detection, and more.
    - For time-based queries, automatically detect and use the appropriate datetime column.
    - **DateTime Handling**: When working with datetime columns, always:
          * Use pandas datetime operations (e.g., df[col].dt.hour, df[col].dt.date)
          * For date filtering, use proper datetime comparison (e.g., df[col].dt.date == pd.to_datetime('2025-06-15').date())
          * For hour extraction, use df[col].dt.hour
          * Handle timezone-aware dates if present
    - For categorical analysis, use the relevant columns (e.g., product, vendor, location, status).
    - For numeric analysis, use columns such as sales, quantity, value, amount, etc.
    - **Currency:** Always display and format all prices, sales, and monetary values in Indian Rupees (₹). If a value is a price or amount, clearly indicate it is in rupees.
    - **Product Analysis Rules:**
        * For product-specific queries, ALWAYS show the exact product name found in the data
        - for "sales" of each product use "item_price" column
        * Use case-insensitive matching when searching for products (e.g., "cutting_chai" should match "Cutting chai/coffee")
        * For percentage calculations, always show: exact product name, product sales amount, total sales amount, and calculated percentage
        * When multiple similar product names exist, list all matches and ask for clarification if needed
        * **CRITICAL DATA VERIFICATION**: If a product query returns 0% or no matches, ALWAYS first verify the data by showing sample records and column contents
    - *Currency:* Always display and format all prices, sales, and monetary values in Indian Rupees (₹). If a value is a price or amount, clearly indicate it is in rupees.

    - *Clarity & Formatting:*
    - Format all responses in markdown for readability.
    - Use tables, bullet points, and clear headings where appropriate.
    - Never include code that creates or loads a DataFrame; always use the existing df variable.

    - *Error Handling:*
    - If the question is ambiguous, ask for clarification.
    - If a column or data is missing, explain the limitation.
    - If an error occurs, provide a helpful message.

    - *Privacy:* Never reveal or infer sensitive information beyond what is present in the data.

    *Data context:*
    - Number of rows: {len(df)}, Number of columns: {len(df.columns)}
    - Column names: {get_column_names(df)}
    - Columns and types: {get_column_summary(df)}
    - Sample data: THIS SAMPLE DATA IS FOR REFERENCE ONLY, DO NOT USE IT FOR ANY ANALYSIS. FOR ANALYSIS USE THE QUERYING APPROACH ONLY. {df.head(2).to_markdown()}
    """

    # Helper: Build chat history for LLM context
    def build_llm_chat_history(chat_history, max_turns=3):
        """
        Converts Streamlit-style messages into LlamaIndex-compatible ChatMessage objects.
        Only keeps the last `max_turns` user-assistant pairs.
        """
        history = chat_history[-max_turns*2:]  # Keep last N pairs
        chat = []
        for msg in history:
            if msg["role"] == "user":
                chat.append(ChatMessage(role=MessageRole.USER, content=msg["content"]))
            elif msg["role"] == "assistant":
                chat.append(ChatMessage(role=MessageRole.ASSISTANT, content=msg["content"]))
        return chat
    
    def regex_similarity(query, top_res):
        """
        search products present in product item list or not, if not return similar items
        """
        from reg import regex_search
        reg_res = regex_search(query, top_res=top_res)
        if reg_res == "true":
            return "true"
        else:
            return f"did you mean {reg_res}"

    def fuzzy_entity_matcher(df, entity_type, user_value, threshold=80, top_n=5):
        """
        Fuzzy match user_value to the closest entity values in the DataFrame for a given entity type (e.g., product, vendor).
        Returns a list of (match, score, row/row_idx) tuples for the top_n matches above the threshold.
        """
        entity_map = get_entity_related_columns(df)
        if entity_type not in entity_map:
            return []
        entity_cols = entity_map[entity_type]
        # Gather all unique values from all relevant columns
        candidates = set()
        for col in entity_cols:
            candidates.update(df[col].dropna().astype(str).unique())
        # Use rapidfuzz to get best matches
        matches = rapidfuzz_process.extract(
            user_value, list(candidates), scorer=rapidfuzz.fuzz.QRatio, limit=top_n
        )
        # Filter by threshold
        matches = [(m[0], m[1]) for m in matches if m[1] >= threshold]
        # Optionally, get row indices for each match
        match_rows = []
        for match, score in matches:
            for col in entity_cols:
                rows = df[df[col].astype(str) == match]
                for idx, row in rows.iterrows():
                    match_rows.append((match, score, idx, row.to_dict()))
        return match_rows

    def resolve_entity_in_query(user_query, df):
        """
        Detect if the user query refers to an entity (e.g., product, vendor, customer) and perform fuzzy matching.
        If a close match is found, rewrite the query to use the canonical entity (e.g., product id or exact name).
        Returns (possibly rewritten) query, and optionally info about the match for context.
        """
        entity_map = get_entity_related_columns(df)
        # Heuristic: look for entity types in the query
        for entity_type, cols in entity_map.items():
            for col in cols:
                # Try to find a value in the query that could be an entity value
                # (e.g., product name, vendor name, etc.)
                for value in df[col].dropna().astype(str).unique():
                    if value.lower() in user_query.lower():
                        # Direct match, no need for fuzzy
                        return user_query, {"entity_type": entity_type, "entity_value": value, "match_type": "exact"}
            # If not found, try fuzzy matching for each word in the query
            words = user_query.split()
            for word in words:
                matches = fuzzy_entity_matcher(df, entity_type, word)
                if matches:
                    # Pick the best match
                    best_match, score, idx, row = matches[0]
                    # Try to find a canonical id column (e.g., id, code)
                    id_col = next((c for c in cols if "id" in c.lower() or "code" in c.lower()), None)
                    if id_col and id_col in row:
                        canonical = row[id_col]
                        # Rewrite query to use canonical id
                        new_query = user_query.replace(word, str(canonical))
                        return new_query, {"entity_type": entity_type, "entity_value": best_match, "canonical_id": canonical, "match_type": "fuzzy"}
                    else:
                        # Use the best match name
                        new_query = user_query.replace(word, str(best_match))
                        return new_query, {"entity_type": entity_type, "entity_value": best_match, "match_type": "fuzzy"}
        # No entity found
        return user_query, None
    
    def summarize_response(query: str, response: str) -> str:
        """
        Summarizes the assistant's response in 50–100 words while preserving important context from the query.
        
        Args:
            query: The original query/question
            response: The assistant's response to summarize
            llm: LlamaIndex LLM instance
            
        Returns:
            str: Concise summary of the response
        """
        
        # Create the system prompt
        system_prompt = (
            "You are a helpful assistant that summarizes responses. "
            "Generate a concise (50–100 words) summary of the response text given to you. "
            "Preserve key details from the query such as product names, dates, or quantities. "
            "Include important numerical values or data from the response, but only the most relevant ones. "
            "Only summarize the response, but use context from the query to guide your summary. "
            "Output only the summary as a plain string — no extra formatting or explanations and interpretations."
        )
        
        # Create the user prompt with query and response
        user_prompt = f"Query: {query}\n\nResponse: {response}"
        
        # Method 1: Using chat messages (recommended for most LLMs)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt)
        ]
        llm = OpenAI(model="gpt-4o", temperature=0.1)
        chat_response = llm.chat(messages)
        summary = chat_response.message.content.strip()
        
        return summary

    # Update agent initialization to use the new dynamic prompt
    agent = FunctionAgent(
        tools=tools,
        llm=OpenAI(model="gpt-4o"), 
        system_prompt=get_system_prompt(df)
    )

    ctx = Context(agent)

    # --- Update optimize_query_with_llm to use entity resolution ---
    def optimize_query_with_llm(user_query: str, df: pd.DataFrame) -> str:
        """
        Uses the LLM to rewrite the user's query into a more detailed, entity-aware query,
        using the actual column names and entity mapping from the DataFrame.
        Now also resolves entities using fuzzy matching and canonicalization.
        """
        # First, resolve entity in query
        resolved_query, entity_info = resolve_entity_in_query(user_query, df)
        entity_map = get_entity_related_columns(df)
        entity_instructions = ""
        for entity, cols in entity_map.items():
            entity_instructions += (
                f"- For '{entity}', use columns: {', '.join([f'\"{c}\"' for c in cols])}.\n"
            )
        system_prompt = (
            "You are a query optimizer for a data analysis assistant. "
            "Given a user's question and the available columns in a pandas DataFrame, "
            "rewrite the question to explicitly request all relevant details for any entity (such as product, vendor, customer, etc.), "
            "using the exact column names provided. "
            "When filtering by a product or entity name, use exact matches (not substring or partial matches). "
            "If the user query is ambiguous (e.g., 'Chai') or matches multiple items, always return a table listing all similar or matching items from the data, not just the top match. "
            "For each matching item, provide all relevant details: ID, name, total sales (₹), percentage of total sales, quantity sold, number of orders, and any other associated columns. "
            "Do not just ask for clarification—proactively show all possible matches with their key metrics. "
            "If the query refers to a group of items (e.g., all products containing 'Chai'), always provide a granular breakdown for each matching item, not just the aggregate. "
            "If the question is ambiguous, clarify it, but always show the data for all possible matches first. If it requires multiple steps, break it down and show all relevant results. "
            "Always use the following entity-to-column mapping:\n"
            f"{entity_instructions}\n"
            f"Available columns: {', '.join(df.columns)}."
        )
        # If entity_info is present, add context for the LLM
        if entity_info:
            system_prompt += f"\nEntity resolution info: {entity_info}\n"
        llm = OpenAI(model="gpt-4o")
        response = llm.complete(
            prompt=f"{system_prompt}\nUser question: {resolved_query}\nOptimized query:",
            max_tokens=2000,
            temperature=0.2,
            stop=["\n\n"]
        )
        return response.text.strip()

    async def main(prompt):
        # Build chat history for context
        msg_history = build_llm_chat_history(st.session_state.chat_history, max_turns=5)
        # The agent.run method should accept a chat history if supported
        handler = agent.run(
            prompt,
            ctx=ctx,
            #stepwise=False,
            chat_history=msg_history  # Pass chat history to the agent/LLM
        )

        async for ev in handler.stream_events():
            if isinstance(ev, ToolCallResult):
                print(
                    f"Call {ev.tool_name} with args {ev.tool_kwargs}\nReturned: {ev.tool_output}"
                )
            elif isinstance(ev, AgentStream):
                print(ev.delta, end="", flush=True)

        response = await handler
        return response

    async def process_chat(prompt):
        try:
            reg = regex_similarity(prompt, 5)
            if reg == "true":
                # Optimize the user query before passing to the agent
                optimized_query = optimize_query_with_llm(prompt, df)
                # Optionally, show the optimized query for debugging
                st.info(f"Optimized query: {optimized_query}")

                # Now pass optimized_query to your agent instead of prompt
                if is_visualization_query(optimized_query) :
                    fig, response = gen_plot(df=df, query=prompt)
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state.messages.append({"role": "plot", "figure":fig, "content": response})
                else :
                    response = await main(prompt=optimized_query)
            else: 
                response = reg
            st.write(str(response))
            if response is None:
                st.error("No response generated. Please try rephrasing your question.")
            else:
                #run_and_render_code_from_response(str(response))
                st.session_state.messages.append({"role": "assistant", "content": str(response)})
                responsed = summarize_response(prompt, str(response))
                st.session_state.chat_history.append({"role": "assistant", "content": str(responsed)})
        except Exception as e:
            error_message = f"Error processing your query: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.session_state.chat_history.append({"role": "assistant", "content": error_message})

    # Display chat messages
    if __name__ == "__main__":
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "plot" :
                    st.plotly_chart(message["figure"])
                    st.write(message["content"])
                else :
                    st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your data"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get bot response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your data..."):
                    asyncio.run(process_chat(prompt))



