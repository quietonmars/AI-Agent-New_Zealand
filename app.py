from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
import os
import pandas as pd
from simple_pandas_engine import SimplePandasEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from pdf import nz_engine
import uuid

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Initialize the agent globally
llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base="https://api.deepseek.com/v1",
    temperature=0,
    max_tokens=4096,
    additional_kwargs={"model": "deepseek-chat"}
)

population_path = os.path.join("data", "WorldPopulation2023.csv")
population_df = pd.read_csv(population_path)

population_query_engine = SimplePandasEngine(
    df=population_df,
    verbose=False,
    instruction_str=instruction_str,
    llm=llm
)

tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="This gives information about the world population and world demographics",
        ),
    ),
    QueryEngineTool(
        query_engine=nz_engine,
        metadata=ToolMetadata(
            name="nz_data",
            description="This gives detailed information about New Zealand the country",
        ),
    )
]

agent = OpenAIAgent.from_tools(
    tools=tools,
    llm=llm,
    verbose=False,
    system_prompt=context
)

# Store chat histories per session
chat_histories = {}


@app.route('/')
def home():
    # Initialize session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        chat_histories[session['session_id']] = []
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        session_id = session.get('session_id')
        if session_id not in chat_histories:
            chat_histories[session_id] = []

        # Get response from agent
        response = agent.chat(user_message)
        bot_response = str(response)

        # Store in chat history
        chat_histories[session_id].append({
            'role': 'user',
            'content': user_message
        })
        chat_histories[session_id].append({
            'role': 'assistant',
            'content': bot_response
        })

        return jsonify({
            'response': bot_response,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/history')
def get_history():
    session_id = session.get('session_id')
    if session_id in chat_histories:
        return jsonify(chat_histories[session_id])
    return jsonify([])


@app.route('/clear', methods=['POST'])
def clear_history():
    session_id = session.get('session_id')
    if session_id in chat_histories:
        chat_histories[session_id] = []
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)