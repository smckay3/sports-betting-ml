import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from tqdm import tqdm


from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

class Dataset(torch.utils.data.Dataset):
    def __init__(self, ratings, teams, outputs, masks):

        self.labels = outputs
        self.ratings = ratings
        self.teams = teams
        self.masks = masks

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_inputs(self, idx):
        # Fetch a batch of inputs
        return (self.ratings[idx], self.teams[idx], self.masks[idx])

    def __getitem__(self, idx):

        batch_inputs = self.get_batch_inputs(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_inputs, batch_y

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.0, hidden_size=48, teams=30):
        super(BertClassifier, self).__init__()

        config = BertConfig(hidden_size=hidden_size)
        print(config)
        self.team_embedding = nn.Embedding(teams, config.hidden_size//2)
        self.bert = BertEncoder(config)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU()

    def forward(self, ratings, teams, mask):
        teams = self.team_embedding(teams)
        ratings = ratings.unsqueeze(-1).expand_as(teams)
        inputs = torch.cat([ratings, teams], -1)
        output, = self.bert(inputs, attention_mask=None,return_dict=False)
        dropout_output = self.dropout(output[:, 0])
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer



def train(model, train_data, val_data, learning_rate, epochs):
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=2, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:

        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            train_ratings = train_input[0].to(device)
            train_team = train_input[1].to(device)
            train_masks = train_input[2].to(device)

            output = model(train_ratings, train_team, train_masks)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        # with torch.no_grad():

        #     for val_input, val_label in val_dataloader:

        #         val_label = val_label.to(device)
        #         mask = val_input["attention_mask"].to(device)
        #         input_id = val_input["input_ids"].squeeze(1).to(device)

        #         output = model(input_id, mask)

        #         batch_loss = criterion(output, val_label.long())
        #         total_loss_val += batch_loss.item()

        #         acc = (output.argmax(dim=1) == val_label).sum().item()
        #         total_acc_val += acc

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )


np.random.seed(112)

samples = 16
players = 32
rating = 48

output = torch.randint(0, 2, (samples,))
ratings = torch.randn((samples, players))
teams = torch.randint(0, 30, (samples, players), dtype=torch.long)
masks = torch.randint(0, 2, (samples, players))

train_data, val_data = Dataset(
    ratings[: int(samples * 0.8)], teams[: int(samples * 0.8)], output[: int(samples * 0.8)], masks[: int(samples * 0.8)]
), Dataset(ratings[int(samples * 0.8) :], teams[int(samples * 0.8) :], output[int(samples * 0.8) :], masks[int(samples * 0.8) :])

model = BertClassifier(hidden_size=rating)

EPOCHS = 50
LR = 1e-5

train(model, train_data, val_data, LR, EPOCHS)
