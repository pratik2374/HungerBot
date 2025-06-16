import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go
import json
import re
import contextlib
import io
load_dotenv()

class LLMQueryProcessor:
    """Handles LLM integration for query processing"""
    
    def __init__(self, llm_provider: str = "openai"):
        self.provider = llm_provider
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup LLM client based on provider"""
        if self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                print("OpenAI package not installed. Run: pip install openai")
            except:
                print("OpenAI API key not configured. Add to Streamlit secrets.")
                
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            except ImportError:
                print("Anthropic package not installed. Run: pip install anthropic")
            except:
                print("Anthropic API key not configured. Add to Streamlit secrets.")
            
    
    def analyze_query_with_llm(self, query: str, df_info: Dict) -> Dict:
        """Use LLM to analyze query and generate visualization code"""
        
        # Create context about the dataframe
        df_context = self._create_dataframe_context(df_info)
        
        # Create the prompt
        prompt = self._create_analysis_prompt(query, df_context)
        
        try:
            if self.provider == "openai" and self.client:
                response = self._query_openai(prompt)
            elif self.provider == "anthropic" and self.client:
                response = self._query_anthropic(prompt)
            else:
                # Fallback to rule-based
                return self._fallback_analysis(query, df_info)
            
            return self._parse_llm_response(response)
            
        except Exception as e:
            print(f"LLM Error: {e}")
            return self._fallback_analysis(query, df_info)
    
    def _create_dataframe_context(self, df_info: Dict) -> str:
        """Create context string about dataframe structure"""
        context = "Dataset Information:\n"
        context += f"Columns and their types:\n"
        
        for col, info in df_info.items():
            context += f"- {col}: {info['type']} ({info['unique_count']} unique values)\n"
            if info['sample_values']:
                context += f"  Sample values: {info['sample_values']}\n"
        
        return context
    
    def _create_analysis_prompt(self, query: str, df_context: str) -> str:
        """Create structured prompt for LLM"""
        return f"""
You are a data visualization expert. Analyze the user's query and provide a structured response for creating the appropriate chart.

{df_context}

User Query: "{query}"

Please provide a JSON response with the following structure:
{{
    "chart_type": "one of: bar, pie, line, scatter, histogram, box, heatmap",
    "columns": ["list", "of", "column", "names"],
    "aggregation": "sum/count/mean/median/none",
    "reasoning": "brief explanation of your choice",
    "code_suggestion": "brief pandas/plotly code suggestion"
}}

Choose the most appropriate visualization based on:
1. Data types of columns mentioned
2. The intent of the query (comparison, distribution, trend, relationship)
3. Best practices for data visualization

Response (JSON only):
"""
    
    def _query_openai(self, prompt: str) -> str:
        """Query OpenAI API"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3
        )
        return response.choices[0].message.content
    
    def _query_anthropic(self, prompt: str) -> str:
        """Query Anthropic API"""
        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response and extract structured data"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                return result
            else:
                raise ValueError("No JSON found in response")
        except:
            # Fallback parsing
            return {
                "chart_type": "bar",
                "columns": [],
                "aggregation": "count",
                "reasoning": "Failed to parse LLM response",
                "code_suggestion": ""
            }
    
    def _fallback_analysis(self, query: str, df_info: Dict) -> Dict:
        """Fallback rule-based analysis when LLM fails"""
        query_lower = query.lower()
        
        # Chart type detection
        if any(word in query_lower for word in ['pie', 'proportion', 'share']):
            chart_type = 'pie'
        elif any(word in query_lower for word in ['line', 'trend', 'time', 'over time']):
            chart_type = 'line'
        elif any(word in query_lower for word in ['scatter', 'relationship', 'vs']):
            chart_type = 'scatter'
        elif any(word in query_lower for word in ['histogram', 'distribution']):
            chart_type = 'histogram'
        else:
            chart_type = 'bar'
        
        # Column detection
        columns = []
        for col in df_info.keys():
            if col.lower() in query_lower:
                columns.append(col)
        
        return {
            "chart_type": chart_type,
            "columns": columns,
            "aggregation": "count",
            "reasoning": "Rule-based fallback analysis",
            "code_suggestion": ""
        }
    
def execute_plotly_code(code: str):
    """
    Executes LLM-generated Plotly code and returns the figure object.
    
    Args:
        code (str): The Python code that creates a Plotly graph and stores it in a variable named `fig`.

    Returns:
        plotly.graph_objects.Figure or str: The resulting Plotly figure, or error message.
    """
    local_vars = {}
    global_vars = {
        "go": go,
        "px": px
    }
    
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, global_vars, local_vars)
    except Exception as e:
        return f"❌ Error during execution: {e}\n\nCode:\n{code}"

    fig = local_vars.get("fig")
    if fig is None:
        return "❌ No variable named `fig` found in the code. Make sure the LLM returns code that ends with `fig = ...`"
    
    return fig

class AdvancedChartGenerator:
    """Enhanced chart generator with LLM insights"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def generate_chart_from_llm_analysis(self, analysis: Dict) -> go.Figure:
        """Generate chart based on LLM analysis"""
        chart_type = analysis.get('chart_type', 'bar')
        columns = analysis.get('columns', [])
        aggregation = analysis.get('aggregation', 'count')
        code = analysis.get("code_suggestion")
        # try :
        #     return execute_plotly_code(code)
        # except:
        #     print("error in exceuting generated on ")
        
        if not columns:
            return self._create_error_chart("No columns specified by LLM")
        
        try:
            if chart_type == 'bar':
                return self._create_smart_bar_chart(columns, aggregation)
            elif chart_type == 'pie':
                return self._create_smart_pie_chart(columns, aggregation)
            elif chart_type == 'line':
                return self._create_smart_line_chart(columns, aggregation)
            elif chart_type == 'scatter':
                return self._create_smart_scatter_chart(columns)
            elif chart_type == 'histogram':
                return self._create_smart_histogram(columns)
            elif chart_type == 'box':
                return self._create_smart_box_chart(columns)
            elif chart_type == 'heatmap':
                return self._create_heatmap(columns)
            else:
                return self._create_smart_bar_chart(columns, aggregation)
                
        except Exception as e:
            return self._create_error_chart(f"Error: {str(e)}")
    
    def _create_smart_bar_chart(self, columns: List[str], aggregation: str) -> go.Figure:
        """Create intelligent bar chart with proper aggregation"""
        if len(columns) == 1:
            col = columns[0]
            if aggregation == 'count':
                data = self.df[col].value_counts().head(20)
                fig = px.bar(x=data.values, y=data.index, orientation='h',
                           title=f'Count of {col}')
            else:
                # For single numeric column
                fig = px.bar(x=range(len(self.df[col])), y=self.df[col],
                           title=f'{col} Values')
        
        elif len(columns) >= 2:
            cat_col = columns[0]
            num_col = columns[1]
            
            # Smart column role detection
            if self.df[cat_col].nunique() > self.df[num_col].nunique():
                cat_col, num_col = num_col, cat_col
            
            if aggregation == 'sum':
                grouped_data = self.df.groupby(cat_col)[num_col].sum()
            elif aggregation == 'mean':
                grouped_data = self.df.groupby(cat_col)[num_col].mean()
            elif aggregation == 'count':
                grouped_data = self.df.groupby(cat_col)[num_col].count()
            else:
                grouped_data = self.df.groupby(cat_col)[num_col].sum()
            
            grouped_data = grouped_data.sort_values(ascending=False).head(20)
            fig = px.bar(x=grouped_data.index, y=grouped_data.values,
                        title=f'{aggregation.title()} of {num_col} by {cat_col}')
        
        fig.update_layout(height=500, showlegend=False)
        return fig
    
    def _create_smart_pie_chart(self, columns: List[str], aggregation: str) -> go.Figure:
        """Create intelligent pie chart"""
        col = columns[0]
        if len(columns) > 1:
            # Choose categorical column
            cat_cols = [c for c in columns if self.df[c].nunique() < 20]
            if cat_cols:
                col = cat_cols[0]
        
        if aggregation == 'count':
            data = self.df[col].value_counts().head(10)
        else:
            # For value-based pie chart
            data = self.df.groupby(col)[columns[1] if len(columns) > 1 else col].sum().head(10)
        
        fig = px.pie(values=data.values, names=data.index,
                    title=f'{aggregation.title()} of {col}')
        fig.update_layout(height=500)
        return fig
    
    def _create_smart_line_chart(self, columns: List[str], aggregation: str) -> go.Figure:
        """Create intelligent line chart"""
        if len(columns) == 1:
            col = columns[0]
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                # Time series
                data = self.df.groupby(col).size()
                fig = px.line(x=data.index, y=data.values,
                             title=f'Trend of {col}')
            else:
                fig = px.line(y=self.df[col], title=f'Sequence of {col}')
        else:
            x_col, y_col = columns[0], columns[1]
            # Sort by x for better line chart
            df_sorted = self.df.sort_values(x_col)
            fig = px.line(df_sorted, x=x_col, y=y_col,
                         title=f'{y_col} over {x_col}')
        
        fig.update_layout(height=500)
        return fig
    
    def _create_smart_scatter_chart(self, columns: List[str]) -> go.Figure:
        """Create intelligent scatter plot"""
        if len(columns) < 2:
            return self._create_error_chart("Scatter plot needs 2+ columns")
        
        x_col, y_col = columns[0], columns[1]
        color_col = columns[2] if len(columns) > 2 else None
        size_col = columns[3] if len(columns) > 3 else None
        
        fig = px.scatter(self.df, x=x_col, y=y_col, 
                        color=color_col, size=size_col,
                        title=f'{y_col} vs {x_col}')
        fig.update_layout(height=500)
        return fig
    
    def _create_smart_histogram(self, columns: List[str]) -> go.Figure:
        """Create intelligent histogram: counts or value-based totals"""
        x_col = columns[0]
        y_col = columns[1]

        # If there's a second column and it's numeric, use it as y
        if len(columns) > 1:
            possible_y = [c for c in columns[1:] if pd.api.types.is_numeric_dtype(self.df[c])]
            if possible_y:
                y_col = possible_y[0]

        # Create histogram
        fig = px.histogram(
            self.df,
            x=x_col,
            y=y_col,  # If y_col is None, Plotly will show counts
            histfunc='sum' if y_col else 'count',
            title=f"{'Sum of ' + y_col + ' by ' + x_col if y_col else 'Distribution of ' + x_col}",
            nbins=min(50, int(np.sqrt(len(self.df))))
        )

        fig.update_layout(height=500)
        return fig
    
    def _create_smart_box_chart(self, columns: List[str]) -> go.Figure:
        """Create intelligent box plot"""
        if len(columns) == 1:
            col = columns[0]
            fig = px.box(self.df, y=col, title=f'Distribution of {col}')
        else:
            cat_col, num_col = columns[0], columns[1]
            # Ensure proper column roles
            if self.df[cat_col].nunique() > 20:
                cat_col, num_col = num_col, cat_col
            
            fig = px.box(self.df, x=cat_col, y=num_col,
                        title=f'{num_col} by {cat_col}')
        
        fig.update_layout(height=500)
        return fig
    
    def _create_heatmap(self, columns: List[str]) -> go.Figure:
        """Create correlation heatmap"""
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(self.df[c])]
        if len(numeric_cols) < 2:
            return self._create_error_chart("Heatmap needs 2+ numeric columns")
        
        corr_matrix = self.df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Correlation Heatmap")
        fig.update_layout(height=500)
        return fig
    
    def _create_error_chart(self, message: str) -> go.Figure:
        """Create error message chart"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Chart Generation Error",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return fig
def get_column_info(df: pd.DataFrame) -> Dict:
    """Get column information"""
    info = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_type = 'datetime'
        elif unique_count < 20:
            col_type = 'categorical'
        else:
            col_type = 'text'
            
        info[col] = {
            'type': col_type,
            'unique_count': unique_count,
            'sample_values': list(df[col].dropna().head(3))
        }
    
    return info

    
def gen_plot(df, query):
    llm_provider = "openai"
    if df is not None:
        llm_processor = LLMQueryProcessor(llm_provider)
        column_info = get_column_info(df)
        analysis = llm_processor.analyze_query_with_llm(query, column_info)

        if not analysis or not isinstance(analysis, dict):
            return None, "❌ LLM analysis failed or returned invalid format."

        chart_generator = AdvancedChartGenerator(df)
        fig = chart_generator.generate_chart_from_llm_analysis(analysis)
        response = analysis.get("reasoning", "No reasoning provided.")
        return fig, str(response)

    return None, "❌ Input DataFrame is None."
