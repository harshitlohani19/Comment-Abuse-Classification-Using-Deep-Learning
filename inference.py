import torch
from transformers import BertTokenizer
from model import AbusiveCommentClassifier

# Load the trained model
model = AbusiveCommentClassifier(n_classes=2)
model.load_state_dict(torch.load("saved_model/model.pth"))
model.eval()

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Function to predict category of a single comment
def predict_comment(comment, model, tokenizer, max_len=128):
    encoding = tokenizer.encode_plus(
        comment,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs, dim=1)

    return prediction.item()


# Example usage
comment = "This is an example comment."
prediction = predict_comment(comment, model, tokenizer)
print(f"Prediction: {prediction}")
