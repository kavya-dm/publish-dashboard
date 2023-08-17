
from flask import Flask, request, render_template_string
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import gensim.downloader as api

app = Flask(__name__)

def find_related_topics_word2vec(csv_file, word):
    # Load the pre-trained Word2Vec model
    word2vec_model = api.load('word2vec-google-news-300')

    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Find similar words based on the pre-trained Word2Vec model
    similar_words = []
    if word in word2vec_model.key_to_index:
        similar_words = word2vec_model.most_similar(word, topn=3000)

    # Check if similar words exist in the CSV
    related_topics = []
    for word, similarity in similar_words:
        if word in df['level3'].values and similarity > 0.25:
            related_topics.append((word, round(similarity, 2)))

    # HTML table
    if related_topics:
        table_rows = ''.join(f'<tr><td>{topic}</td><td>{similarity}</td></tr>' for topic, similarity in related_topics)
        result = f'<table><thead><tr><th>Related Topic</th><th>Similarity</th></tr></thead><tbody>{table_rows}</tbody></table>'
    else:
        result = "No related topics found in the CSV."

    return result

def find_related_topics_sentence(csv_file, phrases):
    # Load the pre-trained sentence transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Create a dictionary to store 'level3' strings and their embeddings
    embeddings_dict = {}

    # Iterate through the 'level3' strings and obtain embeddings
    for level3_string in df['level3']:
        embeddings_dict[level3_string] = model.encode(level3_string, convert_to_tensor=True)

    # Find similar phrases based on sentence embeddings
    related_topics = []
    for phrase in phrases:
        # Encode the phrase and get the sentence embedding
        query_embedding = model.encode(phrase, convert_to_tensor=True)

        # Calculate the cosine similarity between the query embedding and embeddings in the dictionary
        cos_scores = [util.pytorch_cos_sim(query_embedding, embedding) for embedding in embeddings_dict.values()]

        # Choose the top 50 matches 
        top_matches = sorted(enumerate(cos_scores), key=lambda x: x[1], reverse=True)[:50]

        for idx, score in top_matches:
            if score > 0.25:
                topic = list(embeddings_dict.keys())[idx]
                similarity = score.item()
                related_topics.append((topic, round(similarity, 2)))

    # HTML table
    if related_topics:
        table_rows = ''.join(f'<tr><td>{topic}</td><td>{similarity}</td></tr>' for topic, similarity in related_topics)
        result = f'<table><thead><tr><th>Related Topic</th><th>Similarity</th></tr></thead><tbody>{table_rows}</tbody></table>'
    else:
        result = "No related topics found in the CSV."

    return result

@app.route('/')
def index():
    # HTML interface
    html_template = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Related Topics Finder</title>
        <style>
            body {
                font-family: Verdana, sans-serif;
                max-width: 500px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                text-align: center;
                color: #3366cc;
            }
            form {
                display: flex;
                flex-direction: column;
            }
            label {
                margin-bottom: 5px;
            }
            select, input[type="text"] {
                padding: 8px;
                margin-bottom: 15px;
            }
            input[type="submit"] {
                background-color: #3366cc;
                color: #fff;
                cursor: pointer;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            th, td {
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ccc;
            }
            th {
                background-color: #f2f2f2;
            }
        </style>
    </head>
    <body>
        <h1>Topic Relatability Dashboard</h1>
        <form method="POST" action="/related_topics">
            <label for="approach">Select Approach:</label>
            <select id="approach" name="approach">
                <option value="sentence">Multiple Word Approach (Broad Topics)</option>
                <option value="word">Single Word Approach (All Topics)</option>
            </select>
            <br><br>
            <label for="phrases">Enter Input:</label><br>
            <input type="text" name="phrases" placeholder="Enter Input">
            <input type="submit" value="Submit">
        </form>
        <br>
        <a href="https://wiki.dailymotion.com/pages/viewpage.action?spaceKey=TM&title=How+to+use%3A+Topic+Relatability+Dashboard" target="_blank">Click here to access the documentation</a>
    </body>
    </html>
    '''
    return render_template_string(html_template)

@app.route('/related_topics', methods=['POST'])
def related_topics():
    csv_file_sentence = 'broad_topics.csv'
    csv_file_word2vec = 'topics_700k.csv'

    approach = request.form['approach']
    phrases = [phrase.strip() for phrase in request.form['phrases'].replace(',', ', ').split(',') if phrase.strip()]

    if approach == 'sentence':
        result = find_related_topics_sentence(csv_file_sentence, phrases)
    else:
        result = find_related_topics_word2vec(csv_file_word2vec, phrases[0])  # Using only the first input
    return result

if __name__ == '__main__':
    port = 5000
    app.run(port=port)