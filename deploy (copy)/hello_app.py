from flask import request
from flask import jsonify
from flask import Flask
from flask import send_from_directory


from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import pickle

app = Flask(__name__,
            static_folder='./static')

MAX_SEQ_LENGTH=100

def get_model():
    global model
    model = load_model('model.h5');
    print(" * Model loaded!")
get_model();

specialCharacter=['.', ',' ,'!','?',':']
def normalizeSentence(sen):
    sen=sen.split()
    rs=[]
    
    for i in sen:
        if i[-1] in specialCharacter:
            rs.append(i[0:-1]);
            rs.append(i[-1]);
        else:
            rs.append(i)
            
    return rs;

with open('./tokenizers/word_tokenizer.pickle', 'rb') as handle:
    word_tokenizer = pickle.load(handle)

with open('./tokenizers/tag_tokenizer.pickle', 'rb') as handle:
    tag_tokenizer = pickle.load(handle)

@app.route('/',methods=['GET'])
def home():
    return app.send_static_file('hello.html')

@app.route('/hello',methods=['POST'])
def hello():
    message = request.get_json(force = True)
    name = message['name']

    X_sample=normalizeSentence(name)
    X_temp=[]
    X_temp.append(X_sample)
    
    X_sample_encoded = word_tokenizer.texts_to_sequences(X_temp)  # use the tokeniser to encode input sequence
    X_sample_padded = pad_sequences(X_sample_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")
    
    t=model.predict_classes(X_sample_padded)
    
    tag=tag_tokenizer.sequences_to_texts(t)[0]

    tag=tag.split()

    n=len(X_sample)

    rs=""
    for i in range(n):
        rs=rs + X_sample[i]+ ": " + tag[i]+ "  "
    response={
        'greeting': rs
    }
    return jsonify(response)