import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from model import AbusiveCommentClassifier
import torch.nn as nn
import torch.optim as optim


class CommentDataset(Dataset):
    def __init__(self, comments, targets, tokenizer, max_len):
        self.comments = comments
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "comment_text": comment,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(target, dtype=torch.long),
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = CommentDataset(
        comments=df.clean_comment.to_numpy(),
        targets=df.category.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len,
    )

    return DataLoader(ds, batch_size=batch_size, num_workers=4)


df = pd.read_csv("data/dataset.csv")


print(f"DataFrame Columns: {df.columns}")

# Split the data into train and test sets
df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create data loaders
train_data_loader = create_data_loader(df_train, tokenizer, max_len=128, batch_size=16)
test_data_loader = create_data_loader(df_test, tokenizer, max_len=128, batch_size=16)

# Initialize the model
model = AbusiveCommentClassifier(n_classes=2)
model = model.cuda()

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss().cuda()

# Training loop
for epoch in range(10):
    model.train()
    for data in train_data_loader:
        input_ids = data["input_ids"].cuda()
        attention_mask = data["attention_mask"].cuda()
        targets = data["targets"].cuda()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        print(f"Targets: {targets}")
        print(f"Predicted outputs: {outputs}")

        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1} completed. Loss: {loss.item()}")

# Save the model
torch.save(model.state_dict(), "saved_model/model.pth")
