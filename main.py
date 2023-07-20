import pandas as pd
import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from torch.utils.data import DataLoader, Dataset

data = pd.read_csv('data.csv')

print("File Read completed --------------------------------------------------------------------->")

# Remove html tags from Reviews
# def remove_html_tags(review):
#     soup = BeautifulSoup(review,'html.parser')
#     cleaned_text = soup.get_text()
#     return cleaned_text
#
# # Pass Dataframe reviews to function from remove html tags
# def process(data):
#     cleaned_text = remove_html_tags(data)
#     return cleaned_text

# data['clean_text'] = data['review'].apply(lambda x: process(x))
# data['review'] = data['clean_text']
# data.drop('clean_text',axis=1,inplace=True)

# print("\nRemoved Html Tags --------------------------------------------------------------------->")

# with open('stopwords.txt', 'r') as file:
#     file_content = file.read()
#

# def  remove_stop_words(reviews):
#     # stop_words = set(stopwords.words('english'))
#     stop_words = file_content
#     tokens = word_tokenize(reviews)
#     filtered_word = [word for word in tokens if word.lower() not in stop_words]
#     filtered_text = ' '.join(filtered_word)
#     return filtered_text

# def pass_data(data):
#     cleaned_words = remove_stop_words(data)
#     return cleaned_words
#
# data['clean_words'] = data['review'].apply(lambda x: pass_data(x))
# data['review'] = data['clean_words']
# data.drop('clean_words',axis=1,inplace=True)
# data.head()

# print("\nRemoved Stop-Words --------------------------------------------------------------------->")
# data.replace({'sentiment' : {'positive':1,'negative':0}},inplace=True)
# print("\nLabel Replaced --------------------------------------------------------------------->")

X = data['review']
Y = data['sentiment']

print("\n Defined Features and labels --------------------------------------------------------------------->")


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


texts = np.array(X)
labels = np.array(Y)
print("\n Features and labels Convert into array --------------------------------------------------------------------->")

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
print("\n Downloaded Distilbert tokenizer --------------------------------------------------------------------->")

dataset = CustomDataset(texts, labels, tokenizer)

val_size = int(0.33 * len(dataset))
train_size = len(dataset) - val_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)
print("build dataloader --------------------------------------------------------------------->")

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
print("Downloaded Distilbert model --------------------------------------------------------------------->")

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU : {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU avaliable")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("allocated cpu or gpu --------------------------------------------------------------------->")

num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    print("\nmodel train")
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    # Calculate average training loss for the epoch
    avg_loss = total_loss / len(train_dataloader)

    # Perform evaluation on the validation dataset
    model.eval()
    # ... code for evaluation on the validation dataset ...

    print(f"Epoch {epoch + 1}/{num_epochs} - Avg. Loss: {avg_loss:.4f}")

print("\nmodel Trained")
