import pandas as pd


class SimplePandasEngine:
    def __init__(self, df, llm, instruction_str="", verbose=False):
        self.df = df
        self.llm = llm
        self.instruction_str = instruction_str
        self.verbose = verbose

    def query(self, query_str):
        # Create prompt for LLM to generate pandas code
        prompt = f"""You are a pandas expert. Given a DataFrame with these columns:
{list(self.df.columns)}

{self.instruction_str}

User question: {query_str}

Write ONLY the pandas code to answer this question. Return only the code, no explanations.
Use 'df' as the DataFrame variable.

Example: df[df['Country'] == 'New Zealand']['Population2023'].values[0]
"""

        if self.verbose:
            print(f"Query: {query_str}")

        response = self.llm.complete(prompt)
        code = response.text.strip()

        # Clean up code
        code = code.replace('```python', '').replace('```', '').strip()

        if self.verbose:
            print(f"Generated code: {code}")

        try:
            result = eval(code, {'df': self.df, 'pd': pd})
            return str(result)
        except Exception as e:
            return f"Error: {e}\nGenerated code: {code}"