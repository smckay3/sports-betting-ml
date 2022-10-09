import pandas as pd
import math
from tqdm import tqdm

datapath = "data/games.csv"
games_df = pd.read_csv(datapath)
games_df.sort_values(by="GAME_DATE_EST", inplace=True)

datapath = "data/games_details.csv"
details_df = pd.read_csv(datapath)

datapath = "data/players.csv"
players_df = pd.read_csv(datapath)

datapath = "data/teams.csv"
teams_df = pd.read_csv(datapath)

print("loaded csv")

player_ratings = {}
for id in players_df.PLAYER_ID:
    player_ratings[id] = (players_df.PLAYER_NAME, 0.5)

# print(player_ratings)


class Player:
    def __init__(self, df) -> None:
        self.df = df
        self.name = df["PLAYER_NAME"]
        self.player_id = df["PLAYER_ID"]
        self.pre_rating = 0
        self.post_rating = 0

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


time_factor = 640 * 60


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
            map((lambda player: player.pre_rating), self.home_team)
        ) / len(self.home_team)

        self.away_team_rating = sum(
            map((lambda player: player.pre_rating), self.away_team)
        ) / len(self.away_team)

        total_rating = self.home_team_rating + self.away_team_rating
        expected_outcome_home = self.home_team_rating / total_rating
        expected_outcome_away = self.away_team_rating / total_rating

        total_points = self.home_team_score + self.away_team_score
        outcome_home = self.home_team_score / total_points
        outcome_away = self.away_team_score / total_points

        # print(expected_outcome_home)
        # print(outcome_home)

        diff_home = outcome_home - expected_outcome_home
        diff_away = outcome_away - expected_outcome_away

        for player in self.home_team:
            player.post_rating = (
                player.pre_rating + (player.seconds_played / time_factor) * diff_home
            )
            # print(f"player: {player.name}, prerating: {player.pre_rating}, post rating: {player.post_rating}, time played: {player.seconds_played}, diff: {diff_home}, time factor: {time_factor}")
            player_ratings[player.player_id] = (player.name, player.post_rating)

        for player in self.away_team:
            player.post_rating = (
                player.pre_rating + (player.seconds_played / time_factor) * diff_away
            )
            player_ratings[player.player_id] = (player.name, player.post_rating)

    def __str__(self) -> str:
        return f"(home: {self.home_team}, away: {self.away_team})"


print("parsing games")
games = {}
for _, game_data in tqdm(games_df.iterrows()):
    game = Game(game_data)
    games[game.game_id] = game

print("parsing players")
for _, player_info in tqdm(details_df.iterrows()):
    games[player_info["GAME_ID"]].add_player(player_info)

print("updating player ratings")
for game in tqdm(games.values()):
    if game.home_team and game.away_team:
        game.set_ratings(player_ratings)

for name, rating in player_ratings.values():
    print(name, rating)
