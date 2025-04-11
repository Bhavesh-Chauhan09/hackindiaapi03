from flask import Flask, request, jsonify
import json
from huggingface_hub import InferenceClient

app = Flask(__name__)

# Initialize client
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    token="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"
)

def generate_answer(data):
    query = data.get("query", "")
    articles = data.get("articles", [])

    # Construct context from relevant articles
    context = ""
    for article in articles:
        title = article.get("title", "")
        content = article.get("content", "")
        if title.lower() in query.lower():
            context += f"\n### {title.capitalize()}\n{content}"

    if not context:
        return "No relevant articles found based on the query."

    # Create the prompt
    prompt = (
        f"You are a helpful assistant. Use only the following context to answer the user's question:\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    # Generate the answer
    response = client.text_generation(
        prompt=prompt,
        max_new_tokens=500,
        temperature=0.7,
        top_p=0.95,
    )

    return response

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid or missing JSON"}), 400

        answer = generate_answer(data)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    
