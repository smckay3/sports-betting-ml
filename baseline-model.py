import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from torch.optim import Adam
from torch import nn

from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

datapath = "data/games.csv"
games_df = pd.read_csv(datapath)
games_df.sort_values(by="GAME_DATE_EST", inplace=True)

datapath = "data/games_details.csv"
details_df = pd.read_csv(datapath)

datapath = "data/players.csv"
players_df = pd.read_csv(datapath)

datapath = "data/teams.csv"
teams_df = pd.read_csv(datapath)

player_ratings = {}
for id in players_df.PLAYER_ID:
    player_ratings[id] = (players_df.PLAYER_NAME, 0.5)
    
class Player:
    def __init__(self, df) -> None:
        self.df = df
        self.name = df["PLAYER_NAME"]
        self.player_id = df["PLAYER_ID"]
        self.pre_rating = 0
        self.post_rating = 0
        self.plus_minus = df["PLUS_MINUS"]

        time_played = df["MIN"]
        if isinstance(time_played, float):
            self.seconds_played = 0
        else:
            time_played = time_played.split(":")

            self.seconds_played = int(time_played[0]) * 60

            if len(time_played) == 2:
                self.seconds_played += int(time_played[1])

    def __repr__(self) -> str:
        return f"({self.name}, before: {self.pre_rating}, after: {self.post_rating})"
    
class Game:
    def __init__(self, game) -> None:
        self.game_id = game["GAME_ID"]
        self.home_team_id = game["HOME_TEAM_ID"]
        self.away_team_id = game["VISITOR_TEAM_ID"]
        self.home_team = []
        self.away_team = []

        self.home_team_score = game["PTS_home"]
        self.away_team_score = game["PTS_away"]

        self.home_team_rating = 0.5
        self.away_team_rating = 0.5

    def add_player(self, player_info):
        if player_info["TEAM_ID"] == self.home_team_id:
            self.home_team.append(Player(player_info))
        if player_info["TEAM_ID"] == self.away_team_id:
            self.away_team.append(Player(player_info))

    def set_ratings(self, player_ratings):
        for player in self.home_team:
            player.pre_rating = player_ratings.get(
                player.player_id, (player.name, 0.5)
            )[1]

        for player in self.away_team:
            player.pre_rating = player_ratings.get(
                player.player_id, (player.name, 0.5)
            )[1]

        self.home_team_rating = sum(
            map((lambda player: player.pre_rating*player.seconds_played/(60*48)), self.home_team)
            ) / 5
        
        self.away_team_rating = sum(
            map((lambda player: player.pre_rating*player.seconds_played/(60*48)), self.away_team)
            ) / 5

        "Star Player Adjustment"
        home_sp_rating = 0
        away_sp_rating = 0
        for player in self.home_team:
            if player.pre_rating > home_sp_rating:
                home_sp_rating = player.pre_rating
        
        for player in self.away_team:
            if player.pre_rating > away_sp_rating:
                away_sp_rating = player.pre_rating
                
        self.home_team_rating = (self.home_team_rating * 5
                                 + home_sp_rating)/6
        self.away_team_rating = (self.away_team_rating * 5
                                 + away_sp_rating)/6
        
        "END Star Player Adjustment"
        
        "ELO Update"
        
        "Define the K-factor, which is akin to the learning rate (i.e. the"
        "maximum ELO gained or lost from one game)."
        K = .02
        
        "Define the spread between two teams s.t. a team whose ELO rating is"
        "larger by this amount is expected to win 75% of the time."
        spread = .1
        
        "Calculate expected chance for home team to win"
        E_home = 1 / (1 + 10 ** ((self.away_team_rating - self.home_team_rating)/spread))
        
        "Apply rating update based on time played"
        win = 0
        if self.home_team_score > self.away_team_score:
            win = 1
        elif self.home_team_score < self.away_team_score:
            win = 0
        else:
            win = .5
        
        for player in self.home_team:
            player.post_rating = player.pre_rating + K * (win - E_home) * player.seconds_played / (60 * 48)
            player_ratings[player.player_id] = (player.name, player.post_rating)
            
        for player in self.away_team:
            player.post_rating = player.pre_rating + K * (E_home - win) * player.seconds_played / (60 * 48)
            player_ratings[player.player_id] = (player.name, player.post_rating)
        
        "END ELO Update"

    def __str__(self) -> str:
        return f"(home: {self.home_team}, away: {self.away_team})"
    
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

    def __init__(self, dropout=0.5, hidden_size=256):
        super(BertClassifier, self).__init__()

        config = BertConfig(hidden_size=hidden_size, intermediate_size=hidden_size*4, num_hidden_layers=8, num_attention_heads=8)
        print(config)
        self.team_embedding = nn.Embedding(2, config.hidden_size//2)
        self.bert = BertEncoder(config)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 2)
        self.relu = nn.ReLU()

    def forward(self, ratings, teams, mask):
        teams = self.team_embedding(teams)
        ratings = ratings.unsqueeze(-1).expand_as(teams)
        inputs = torch.cat([ratings, teams], -1)

        # fix mask shape and transform to additive mask
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * torch.finfo(torch.float32).min

        output, = self.bert(inputs, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(output[:, 0])
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer



def train(model, train_data, val_data, learning_rate, epochs):
    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=64, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=64)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:

        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        model.train()
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

        model.eval()
        with torch.no_grad():

            for val_input, val_label in val_dataloader:

                val_label = val_label.to(device)
                val_ratings = val_input[0].to(device)
                val_team = val_input[1].to(device)
                val_masks = val_input[2].to(device)

                output = model(val_ratings, val_team, val_masks)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}"
        )


print("parsing games")
games = {}
for _, game_data in tqdm(games_df.iterrows()):
    game = Game(game_data)
    games[game.game_id] = game

print("parsing players")
for _, player_info in tqdm(details_df.iterrows()):
    games[player_info["GAME_ID"]].add_player(player_info)

samples = len(games_df)
output = torch.zeros(samples)
players = 48
ratings = torch.zeros((samples, players))
masks = torch.zeros((samples, players))

print("initializing samples and outputs")
for index, game in tqdm(enumerate(games.values())):
    if game.home_team and game.away_team:
        # "Assign true labels to data samples (win/loss)"
        if game.home_team_score == game.away_team_score:
            output[index] = .5
        else:
            output[index] = game.home_team_score > game.away_team_score

        
        # "Initialize ratings to pre rating for each game based on generic ELO system"
        # "Initialize masks to 1 if player exists in that position for input, otherwise 0"
        player_index = 0
        for player in game.home_team:
            ratings[index][player_index] = player_ratings.get(
                player.player_id, (player.name, 0.5)
                )[1]
            masks[index][player_index] = 1
            player_index += 1
        while player_index < players // 2:
            masks[index][player_index] = 0
            player_index += 1

        # Away team
        for player in game.away_team:
            ratings[index][player_index] = player_ratings.get(
                player.player_id, (player.name, 0.5)
                )[1]
            player_index += 1
        while player_index < players:
            masks[index][player_index] = 0
            player_index += 1
            
        # "Update ELO ratings after each game"
        game.set_ratings(player_ratings)

# "Teams is set so the first half of players are home (0) and the last half are away (1)"
teams = torch.zeros((samples, players), dtype=torch.long)
teams[:,:players//2] = 1

np.random.seed(112)

train_data, val_data = Dataset(
    ratings[: int(samples * 0.8)], teams[: int(samples * 0.8)], output[: int(samples * 0.8)], masks[: int(samples * 0.8)]
), Dataset(ratings[int(samples * 0.8) :], teams[int(samples * 0.8) :], output[int(samples * 0.8) :], masks[int(samples * 0.8) :])

model = BertClassifier(hidden_size=256)

EPOCHS = 500
LR = 1e-6

train(model, train_data, val_data, LR, EPOCHS)

