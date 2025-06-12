import streamlit as st
import pandas as pd
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

# Set OpenAI API key from secrets
os.environ['OPENAI_API_KEY'] = st.secrets["openai_api_key"]

# Set page config
st.set_page_config(
    page_title="Data Analysis Chatbot",
    page_icon="🤖",
    layout="centered"  # Options are: "centered", "wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("📊 Data Analysis Chatbot")
st.markdown("Ask questions about your data and get instant insights!")

# Sidebar instructions
with st.sidebar:
    st.header("📁 Data Upload")
    st.markdown("""
    ### Instructions:
    1. Upload your CSV or Excel file using the file uploader below
    2. Supported formats: CSV (.csv) and Excel (.xls, .xlsx)
    3. Make sure your data is clean and properly formatted
    4. Maximum file size: 200MB
    
    ### Tips:
    - For best results, ensure your data has clear column headers
    - Remove any unnecessary columns before uploading
    - Check for missing values in your dataset
    """)

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
            return df
            
    # Load the data and index
    @st.cache_data
    def load_index(df):
        dirs = glob.glob("./storage/sample_*")

        if not dirs:
            # Create indexing and vector store if none exist
            text = df.to_csv(index=False)
            documents = Document(text=text)
            index = VectorStoreIndex.from_documents([documents])

            # Timestamped directory name
            now_str = datetime.datetime.now().strftime("%d%m%H%M")
            persist_dir = f"./storage/sample_{now_str}"
            index.storage_context.persist(persist_dir=persist_dir)

            return index

        else:
            # Always find the latest modified one
            persist_dir = max(dirs, key=os.path.getmtime)
            print("Latest directory:", persist_dir)

        try:
            st.session_state.storage_context = StorageContext.from_defaults(
                persist_dir=persist_dir
            )
            print("Loading index from latest storage context:")
            st.session_state.supa_index = load_index_from_storage(st.session_state.storage_context)
            return st.session_state.supa_index

        except Exception as e:
            st.error(
                f"No existing vector index found in '{persist_dir}'. Please create the index and embeddings first.\nError: {e}"
            )
            st.stop()
    # --- END: Load the selected dataset ---

    df = load_csv()

    # -- NEW: Initialize the PandasQueryEngine and vector Store Engine with the loaded DataFrame --
    query_engine = PandasQueryEngine(
        df=df,
        verbose=False,
        synthesize_response=True,
        )
    st.session_state.supa_index = load_index(df)
    Supa_Engine = st.session_state.supa_index.as_query_engine(similarity_top_k=3)


    # -- VECTOR STORE ENGINE --
    query_engine_tools = [
        QueryEngineTool.from_defaults(
            query_engine=Supa_Engine,
            name="Sales-Analyser",
            description=(
                # "Use this tool to answer questions about the uploaded data which are more textual in nature, "
                # "this tool only returns the most relevant text-based insights from the data, "
                # "So, use it where we require a more narrative or descriptive answer, "
                # "such as understanding customer sentiments, product reviews, feedbacks or general trends in the data. "
                # "And it won't perform well on whole data having numerical, so use it with textual data like taking feedbacks"
                "Use this tool to get key insights from uploaded data that's mostly text. It's best for descriptive questions like "
                "customer feedback, reviews, or sentiment. Avoid using it on fully numerical data — it's meant for understanding written content."
            ),
        )
    ]

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


    tools = query_engine_tools + [panda_retriver]

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
            "plot", "chart", "graph", "visualize", "visualization", "draw", "histogram", "bar", "line", "scatter", "pie"
        ]
        return any(word in query.lower() for word in viz_keywords)

    def render_visualization(df: pd.DataFrame, query: str):
        """
        Generate a visualization based on the query using pandas/seaborn/matplotlib.
        Handles numeric and categorical data robustly and asks for clarification if needed.
        """
        try:
            # Bar plot: try to find a categorical x and numeric y, or count if y is not numeric
            if "bar" in query or "bar plot" in query or "bar chart" in query:
                # Try to find a categorical column for x
                x_col = next((col for col in df.columns if df[col].dtype == 'object' or 'cat' in str(df[col].dtype)), df.columns[0])
                # Try to find a numeric column for y
                y_col = next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])), None)

                if y_col:
                    # If numeric y, aggregate by sum
                    agg_df = df.groupby(x_col)[y_col].sum().reset_index()
                    plt.figure(figsize=(8, 4))
                    sns.barplot(x=agg_df[x_col], y=agg_df[y_col])
                    plt.ylabel(y_col)
                    plt.xlabel(x_col)
                    plt.title(f"{y_col} by {x_col}")
                    plt.xticks(rotation=45)
                else:
                    # If no numeric y, ask for clarification before plotting counts
                    st.info(f"You asked for a bar plot of '{x_col}', but there is no numeric column to plot. "
                            "Would you like to see a count plot (number of occurrences for each category)?")
                    # Optionally, you could add a button for the user to confirm
                    if st.button(f"Show count plot for {x_col}"):
                        counts = df[x_col].value_counts()
                        plt.figure(figsize=(8, 4))
                        sns.barplot(x=counts.index, y=counts.values)
                        plt.ylabel("Count")
                        plt.xlabel(x_col)
                        plt.title(f"Count by {x_col}")
                        plt.xticks(rotation=45)
                    else:
                        return
            elif "hist" in query or "histogram" in query:
                col = next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])), None)
                if col:
                    plt.figure(figsize=(8, 4))
                    sns.histplot(df[col].dropna(), kde=True)
                    plt.title(f"Histogram of {col}")
                else:
                    st.error("No numeric column found for histogram.")
                    return
            elif "scatter" in query or "scatter plot" in query:
                num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                if len(num_cols) >= 2:
                    plt.figure(figsize=(8, 4))
                    sns.scatterplot(data=df, x=num_cols[0], y=num_cols[1])
                    plt.title(f"Scatter plot of {num_cols[0]} vs {num_cols[1]}")
                else:
                    st.error("Not enough numeric columns for scatter plot.")
                    return
            else:
                st.info("Visualization type not recognized or not explicitly requested. Please specify the type of plot you want.")
                return
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png")
            st.image(buf)
            plt.close()
        except Exception as e:
            st.error(f"Could not generate visualization: {e}")

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
        # Example for customer
        customer_cols = [col for col in df.columns if "customer" in col.lower() or "user" in col.lower()]
        if customer_cols:
            entity_map["customer"] = customer_cols
        # Example for order
        order_cols = [col for col in df.columns if "order" in col.lower()]
        if order_cols:
            entity_map["order"] = order_cols
        # Example for location
        location_cols = [col for col in df.columns if "location" in col.lower()]
        if location_cols:
            entity_map["location"] = location_cols
        # Add more entities as needed
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
    You are HungerBox's advanced Data Analysis Assistant, designed to help users analyze and gain insights from complex sales and operational data. You are integrated into a Retrieval-Augmented Generation (RAG) system and have access to the user's uploaded data as a pandas DataFrame named `df`.

    Current date and time: {current_datetime}

    Your core responsibilities:
    - **Data Fidelity:** Always use the provided DataFrame `df` for all analysis, calculations, and visualizations. Never create or load a new DataFrame or use sample data.
    - **Column Awareness:** Only use the actual column names and data types present in `df`. If you are unsure about a column name, refer to the provided list of columns.
    - **Entity Context:**  
    {entity_instructions}
    - **Business Context:** Assume the data is related to sales, orders, transactions, customers, vendors, products, payments, and other business operations typical for a food-tech company like HungerBox. However, always adapt to the actual columns and data provided.
    - **Directness:** Always answer the user's question directly and concisely. If the question is ambiguous or cannot be answered, ask for clarification or explain why.
    - **Details:** Always answer the user's question directly, **with a detailed, step-by-step explanation** of your reasoning and findings. For every response, include a clear summary of what the result means in a business context, and how the user can interpret or act on it.
    - **Visualization:** 
        - Only generate a visualization if the user explicitly requests it (e.g., plot, chart, graph, visualization, histogram, bar, line, scatter, pie) or if the question strongly implies a need for visual representation (e.g., trend, distribution, compare, relationship).
        - If a visualization is required, return only the Python code (in a code block) that generates the requested plot using matplotlib or seaborn, and then provide a concise explanation of what the chart shows. Do not explain the code itself.
        - If the user does not ask for a visualization, provide only a direct, concise, and complete answer to their question, with no code block.
        - If a visualization is not possible, explain why.
    - **Analysis:** 
        - Support a wide range of queries: aggregations, trends, comparisons, filtering, segmentation, cohort analysis, time series, outlier detection, and more.
        - For time-based queries, automatically detect and use the appropriate datetime column.
        - For categorical analysis, use the relevant columns (e.g., product, vendor, location, status).
        - For numeric analysis, use columns such as sales, quantity, value, amount, etc.
    - **Currency:** Always display and format all prices, sales, and monetary values in Indian Rupees (₹). If a value is a price or amount, clearly indicate it is in rupees.
    - **Clarity & Formatting:** 
        - Format all responses in markdown for readability.
        - Use tables, bullet points, and clear headings where appropriate.
        - Never include code that creates or loads a DataFrame; always use the existing `df` variable.
    - **Error Handling:** 
        - If the question is ambiguous, ask for clarification.
        - If a column or data is missing, explain the limitation.
        - If an error occurs, provide a helpful message.
    - **Privacy:** Never reveal or infer sensitive information beyond what is present in the data.

    **Data context:**
    - Number of rows: {len(df)}, Number of columns: {len(df.columns)}
    - Column names: {get_column_names(df)}
    - Columns and types:
    {get_column_summary(df)}
    - Sample data:
    {df.head(2).to_markdown()}
    """

    def run_and_render_code_from_response(response: str):
        """
        Detects Python code blocks in the response, executes them, and renders any matplotlib plots.
        Shows the rest of the response as markdown.
        """
        code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", response)
        non_code = re.sub(r"```(?:python)?\s*[\s\S]*?```", "", response).strip()
        
        # Run each code block
        for code in code_blocks:
            try:
                plt.close('all')
                exec_globals = {
                    "plt": plt,
                    "sns": sns,
                    "pd": pd,
                    "df": df,
                }
                exec(code, exec_globals)
                buf = io.BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format="png")
                st.image(buf)
                plt.close()
            except Exception as e:
                st.error(f"Error running code snippet: {e}")
        
        if non_code:
            st.markdown(non_code)

    # Helper: Build chat history for LLM context
    def build_llm_chat_history(messages, max_turns=5):
        """
        Converts Streamlit-style messages into LlamaIndex-compatible ChatMessage objects.
        Only keeps the last `max_turns` user-assistant pairs.
        """
        history = messages[-max_turns*2:]  # Keep last N pairs
        chat = []
        for msg in history:
            if msg["role"] == "user":
                chat.append(ChatMessage(role=MessageRole.USER, content=msg["content"]))
            elif msg["role"] == "assistant":
                chat.append(ChatMessage(role=MessageRole.ASSISTANT, content=msg["content"]))
        return chat

    # Update agent initialization to use the new dynamic prompt
    agent = FunctionAgent(
        tools=tools,
        llm=OpenAI(model="gpt-4o"), 
        system_prompt=get_system_prompt(df)
    )

    st.session_state.ctx = Context(agent)

    def optimize_query_with_llm(user_query: str, df: pd.DataFrame) -> str:
        """
        Uses the LLM to rewrite the user's query into a more detailed, entity-aware query.
        """
        # Use a simple prompt for the LLM to expand the query
        system_prompt = (
            "You are a query optimizer for a data analysis assistant. "
            "Given a user's question and the available columns in a pandas DataFrame, "
            "rewrite the question to explicitly request all relevant details for any entity (such as product, vendor, customer, etc.), "
            "including IDs, names, dates, and values. "
            "If the question is ambiguous, make it specific and multi-step if needed. "
            "Columns available: " + ", ".join(df.columns) + "."
        )
        # Use your OpenAI LLM (or any LLM you have) to rewrite the query
        llm = OpenAI(model="gpt-4o")  # or your preferred model
        response = llm.complete(
            prompt=f"{system_prompt}\nUser question: {user_query}\nOptimized query:",
            max_tokens=128,
            temperature=0.2,
            stop=["\n\n"]
        )
        return response.text.strip()

    async def main(prompt):
        # Build chat history for context
        chat_history = build_llm_chat_history(st.session_state.messages, max_turns=5)
        # The agent.run method should accept a chat history if supported
        handler = agent.run(
            prompt,
            ctx=st.session_state.ctx,
            #stepwise=False,
            chat_history=chat_history  # Pass chat history to the agent/LLM
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
            # Optimize the user query before passing to the agent
            optimized_query = optimize_query_with_llm(prompt, df)
            # Optionally, show the optimized query for debugging
            st.info(f"Optimized query: {optimized_query}")

            # Now pass optimized_query to your agent instead of prompt
            response = await main(prompt=optimized_query)
            if response is None:
                st.error("No response generated. Please try rephrasing your question.")
            else:
                run_and_render_code_from_response(str(response))
                st.session_state.messages.append({"role": "assistant", "content": str(response)})
        except Exception as e:
            error_message = f"Error processing your query: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

    # Display chat messages
    if __name__ == "__main__":
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your data"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get bot response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your data..."):
                    asyncio.run(process_chat(prompt))



