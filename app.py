from flask import Flask, render_template, request, jsonify
import requests
import replicate
import os

app = Flask(__name__)

# Replace 'your_replicate_api_token' with your actual Replicate API token
REPLICATE_API_TOKEN = 'Your Replicate Key'
os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN

# Default model configuration
LLAMA_7B_MODEL = 'a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea'
TEMPERATURE = 0.1
TOP_P = 0.9
MAX_LENGTH = 150

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate-idea', methods=['POST'])
def generate_idea():
    try:
        user_input = request.json.get('user_input')
        if not user_input:
            return jsonify({'error': 'No user input provided'}), 400

        string_dialogue = "You are an AI that generates creative ideas. Respond to the user's input accordingly.\n\n"
        string_dialogue += f"User: {user_input}\n\nAssistant:"

        output = replicate.run(
            LLAMA_7B_MODEL,
            input={"prompt": string_dialogue, "temperature": TEMPERATURE, "top_p": TOP_P, "max_length": MAX_LENGTH, "repetition_penalty": 1}
        )
        idea = "".join(output).strip()

        return jsonify({'idea': idea})
    
    except requests.exceptions.RequestException as e:
        return jsonify({'error': 'Request to Replicate API failed: ' + str(e)}), 500
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
