from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer
from model import AbusiveCommentClassifier

app = Flask(__name__)

# Load model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = AbusiveCommentClassifier(n_classes=2)
model.load_state_dict(torch.load('model.pth'))
model = model.to(device)
model.eval()

# Preprocess function
def preprocess(comment, tokenizer, max_len=128):
    encoding = tokenizer.encode_plus(
        comment,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoding['input_ids'], encoding['attention_mask']

# Prediction function
def predict(comment):
    input_ids, attention_mask = preprocess(comment, tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs, dim=1)
    
    return prediction.item()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_abusive():
    data = request.get_json()
    comment = data['comment']
    prediction = predict(comment)
    result = {'result': bool(prediction)}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)

