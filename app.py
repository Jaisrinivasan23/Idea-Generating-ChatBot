from flask import Flask, render_template, request, jsonify
import openai

app = Flask(__name__)

# Replace 'your_openai_api_key' with your actual OpenAI API key
openai.api_key = 'your_openai_api_key'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate-idea', methods=['POST'])
def generate_idea():
    user_input = request.json.get('user_input')
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI that generates creative ideas."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=150
    )
    
    idea = response.choices[0].message['content'].strip()
    
    return jsonify({'idea': idea})

if __name__ == '__main__':
    app.run(debug=True)
