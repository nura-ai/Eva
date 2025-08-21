from flask import Flask, render_template, request, jsonify
import ollama
import uuid  # to generate unique session ids on the fly

app = Flask(__name__)

# --------------------------------------------------
# 1. Session store – keeps a conversation for each client
# --------------------------------------------------
chat_histories = {}          # { session_id : [ {role:..., content:...}, ... ] }

# --------------------------------------------------
# 2. Helper – start a new chat history with a system prompt
# --------------------------------------------------
def init_history():
    # Here you can set/filter the AI , tell him how to reply, how to react and so on.
    return [
        {"role": "system", "content": "You are a concise assistant. Provide short, direct, and factual answers without elaboration."}
    ]

# --------------------------------------------------
# 3. Home page – a very small demo UI
# --------------------------------------------------
@app.route("/")
def home():
    # a tiny html page that lets you chat (see templates/index.html)
    return render_template("index.html")

# --------------------------------------------------
# 4. Chat endpoint – called by the front‑end via JS
# --------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    user_msg = data.get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400

    # If the front‑end didn't send a session_id, make a new one
    session_id = data.get("session_id")
    if not session_id or session_id not in chat_histories:
        session_id = str(uuid.uuid4())
        chat_histories[session_id] = init_history()

    # Append the user’s message to the conversation history
    chat_histories[session_id].append({"role": "user", "content": user_msg})

    # Call Ollama
    try:
        response = ollama.chat(
            model="gpt-oss:20b",
            messages=chat_histories[session_id]
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    # Parameter filter over the AI, with this you can "play" as seen below:
    # However this can also bug the AI (it depends on the model)
    # try:
    #     response = ollama.chat(
    #         model="gpt-oss:20b",
    #         messages=chat_histories[session_id],
    #         options={
    #             "num_predict": 50,  # Max number of tokens to predict in the response
    #             "temperature": 0.5, # Controls randomness. Lower = more predictable, higher = more creative
    #             "top_k": 20,        # Limits the pool of tokens to sample from
    #             "top_p": 0.9,       # Further refines token selection
    #         }
    #     )
    # except Exception as exc:
    #     return jsonify({"error": str(exc)}), 500
    # The response object looks like:
    # {'model': 'gpt-oss:20b', 'created_at': '...', 'message': {'role': 'assistant', 'content': '…'}, 'done': True}

    assistant_msg = response["message"]["content"]
    # Append assistant’s reply so that context is preserved
    chat_histories[session_id].append({"role": "assistant", "content": assistant_msg})

    # Return the reply **and** the session_id so the client can keep using it
    return jsonify({
        "response": assistant_msg,
        "session_id": session_id
    })

# --------------------------------------------------
# 5. Run the server
# --------------------------------------------------
if __name__ == "__main__":
    # In production use a WSGI server (gunicorn, uWSGI, etc.)
    app.run(debug=True, host="0.0.0.0", port=5000)
