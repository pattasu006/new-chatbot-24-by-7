from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
CORS(app)

user_context = {}

# Sample training data (you can expand this)
training_sentences = [
    "hi", "hello", "hey", "good morning",
    "my name is John", "I'm Alice", "name is Bob",
    "how can I reset my password?", "forgot my password",
    "I want a refund", "how do I get a refund?",
    "where is my order?", "track delivery", "delivery status",
    "bye", "goodbye", "see you later"
]

training_labels = [
    "greet", "greet", "greet", "greet",
    "set_name", "set_name", "set_name",
    "reset_password", "reset_password",
    "refund_policy", "refund_policy",
    "order_status", "order_status", "order_status",
    "goodbye", "goodbye", "goodbye"
]

# Train ML model
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_sentences)
model = MultinomialNB()
model.fit(X_train, training_labels)

def get_intent(user_input):
    X_test = vectorizer.transform([user_input])
    return model.predict(X_test)[0]

def extract_name(text):
    import re
    match = re.search(r"(my name is|name is|i'm|i am)\s+(\w+)", text.lower())
    return match.group(2).capitalize() if match else None

def get_bot_response(user_input, user_id="default"):
    intent = get_intent(user_input)
    name = user_context.get(user_id, {}).get("name", "")

    if intent == "set_name":
        name = extract_name(user_input)
        if name:
            user_context[user_id] = {"name": name}
            return f"Nice to meet you, {name}! How can I assist you today?"
        return "Sorry, I didn't catch your name."

    elif intent == "greet":
        return f"Hello {name}!" if name else "Hello! What's your name?"

    elif intent == "reset_password":
        return "To reset your password, go to Account > Security > Reset Password."

    elif intent == "refund_policy":
        return "To request a refund, visit Help Center > Refunds and fill out the form."

    elif intent == "order_status":
        return "You can track your order under the 'My Orders' section in your profile."

    elif intent == "goodbye":
        return f"Goodbye {name}!" if name else "Goodbye! Take care."

    else:
        return "I'm not sure about that. Could you rephrase your question?"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    user_id = data.get("user_id", "default")
    response = get_bot_response(user_input, user_id)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
