import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open("intents.json", "r") as f:
    data = json.load(f)

questions = []
answers = []

for intent in data["intents"]:
    for p in intent["patterns"]:
        questions.append(p)
        answers.append(intent["response"])

vectorizer = CountVectorizer().fit(questions)
X = vectorizer.transform(questions)

print("College Chatbot (type exit to stop)")

while True:
    user = input("You: ")

    if user.lower() == "exit":
        break

    user_vec = vectorizer.transform([user])
    sim = cosine_similarity(user_vec, X)

    index = sim.argmax()
    score = sim[0][index]

    if score > 0.3:
        print("Bot:", answers[index])
    else:
        print("Bot: Sorry, I'm not sure. Please contact the college office.")