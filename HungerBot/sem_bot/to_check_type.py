import os
from dotenv import load_dotenv
load_dotenv()
from typing import Dict, List
import re
import json

class OueryClassifier:
    """Classify the type of query into [plot/general]"""
    
    def __init__(self, llm_provider: str = "openai"):
        self.provider = llm_provider
        self.client = None
        self._setup_client()
    
    def _setup_client(self):
        """Setup LLM client based on provider"""
        if self.provider == "openai":
            try:
                import openai
                self.client = openai.OpenAI()
            except ImportError:
                print("OpenAI package not installed. Run: pip install openai")
            except:
                print("OpenAI API key not configured. Add to Streamlit secrets.")
                
            
    
    def analyze_query_with_llm(self, query: str) -> Dict:
        """Use LLM to analyze query and catagorize query"""
        
        # Create context about the dataframe
        # df_context = self._create_dataframe_context(df_info)
        
        # Create the prompt
        # prompt = self._create_analysis_prompt(query, df_context)
        prompt = self._create_analysis_prompt(query)
        #print(prompt)
        
        try:
            if self.provider == "openai" and self.client:
                response = self._query_openai(prompt)
            else:
                # Fallback to rule-based
                return self._fallback_analysis(query)
            #print(response)
            return self._parse_llm_response(response)
            
        except Exception as e:
            #print(f"LLM Error: {e}")
            return self._fallback_analysis(query)
    
    # def _create_dataframe_context(self, df_info: Dict) -> str:
    #     """Create context string about dataframe structure"""
    #     context = "Dataset Information:\n"
    #     context += f"Columns and their types:\n"
        
    #     for col, info in df_info.items():
    #         context += f"- {col}: {info['type']} ({info['unique_count']} unique values)\n"
    #         if info['sample_values']:
    #             context += f"  Sample values: {info['sample_values']}\n"
        
    #     return context
    
    # def _create_analysis_prompt(self, query: str, df_context: str) -> str:
    def _create_analysis_prompt(self, query: str) -> str:
        """Create structured prompt for LLM"""
        return f"""
You are a data Enginering expert. Analyze the user's query and provide a structured response for classifying the data for further classification.

User Query: "{query}"

From analyzing the user query categorize the query into \n product_related which means user wants repsonse for a product related question or not\n
if yes provide the name of product

EXAMPLE:
1. **QUERY** : what are the sales of chai?
   thinking : chai is a product 
   **RESPONSE**: {{
        "product_name : "chai"
   }}
2. **QUERY** : what are the sales on Monday?
   thinking : chai is a product 
   **RESPONSE**: {{
        "product_name : "no"
   }}


Please provide a JSON response with the following structure:
{{
    "product_name": "name of product if product related query or just no"
}}

Response (JSON only):
"""
    
    def _query_openai(self, prompt: str) -> str:
        """Query OpenAI API"""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1
        )
        return response.choices[0].message.content
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response and extract structured data"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            print("json matched")
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                print(f"result {result}")
                return result
            else:
                raise ValueError("No JSON found in response")
        except:
            # Fallback parsing
            return {
                "product_name": "no"
            }
    
    def _fallback_analysis(self, query: str) -> Dict:
        """Fallback rule-based analysis when LLM fails"""
        query_lower = query.lower()
        
        # Chart type detection
        if any(word in query_lower for word in ['chai', 'cutting_chai', 'tea', 'cutting chai', 'chai ']):
            query_type = 'chai'
        else:
            query_type = 'no'
        
        return {
            "product_name": query_type
        }
    

def classify(query):
    llm = "openai"
    print(f"LLM set to {llm}")
    llm_processor = OueryClassifier(llm)
    analysis = llm_processor.analyze_query_with_llm(query)
    if analysis["product_name"] == "no" :
        return "no"
    else :
        return analysis["product_name"]
