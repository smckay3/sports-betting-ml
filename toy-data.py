import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from tqdm import tqdm


from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs, masks):

        self.labels = outputs
        self.inputs = inputs
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
        return self.inputs[idx]

    def get_batch_masks(self, idx):
        # Fetch a batch of inputs
        return self.masks[idx]

    def __getitem__(self, idx):

        batch_inputs = self.get_batch_inputs(idx)
        batch_y = self.get_batch_labels(idx)
        batch_masks = self.get_batch_masks(idx)

        return batch_inputs, batch_y, batch_masks

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5, hidden_size=48):
        super(BertClassifier, self).__init__()

        config = BertConfig(hidden_size=hidden_size)
        print(config)
        self.bert = BertEncoder(config)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU()

    def forward(self, inputs, mask):
        print(masks.shape)
        _, pooled_output = self.bert(inputs, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
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

        for train_input, train_label, train_masks in tqdm(train_dataloader):

            train_label = train_label.to(device)
            input_id = train_input.to(device)

            output = model(input_id, train_masks)

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
ratings = torch.randn((samples, players, rating))
masks = torch.randint(0, 2, (samples, players))

train_data, val_data = Dataset(
    ratings[: int(samples * 0.8)], output[: int(samples * 0.8)], masks[: int(samples * 0.8)]
), Dataset(ratings[int(samples * 0.8) :], output[int(samples * 0.8) :], masks[int(samples * 0.8) :])

model = BertClassifier(hidden_size=rating)

# train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True)
# val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=2)

# for train_input, train_label in tqdm(train_dataloader):
#     print(train_input, train_label)
# print(len(df_train),len(df_val), len(df_test))

EPOCHS = 5
LR = 1e-6

train(model, train_data, val_data, LR, EPOCHS)
