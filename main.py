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

        """
        RB - We shouldn't average the player's ratings for the team average.
        If the starting 5 plays most of the game and has a high rating, the
        team's rating should be high. I think we should either:
            a) Save playtime for players as they play and use this stat
            to predict their average playtime in a game for rating purposes
            OR
            b) Pick some number (say 10) and normalize all teams to have this
            number of players (choose the top N ratings on the team).
        """
        self.home_team_rating = sum(
            map((lambda player: player.pre_rating), self.home_team)
        ) / len(self.home_team)

        self.away_team_rating = sum(
            map((lambda player: player.pre_rating), self.away_team)
        ) / len(self.away_team)
        
        "Star Player Adjustment"
        home_sp_rating = 0
        away_sp_rating = 0
        for player in self.home_team:
            if player.pre_rating > home_sp_rating:
                home_sp_rating = player.pre_rating
        
        for player in self.away_team:
            if player.pre_rating > away_sp_rating:
                away_sp_rating = player.pre_rating
                
        self.home_team_rating = (self.home_team_rating * len(self.home_team)
                                 + home_sp_rating)/(len(self.home_team) + 1)
        self.away_team_rating = (self.away_team_rating * len(self.away_team)
                                 + away_sp_rating)/(len(self.away_team) + 1)
        
        "END Star Player Adjustment"
        
        "Crude Rating Update"

        """
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
        
        """
            
        "END Crude Rating Update"
        
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

"Sort players by rating" "RB - I have no idea what this does but it's not doing what I want. Found it online somewhere."
sorted_ratings = {v for k, v in sorted(player_ratings.items(), key = lambda item: item[1])}

print(sorted_ratings)

#for name, rating in sorted_ratings.values():
    #print(name, rating)

#for name, rating in player_ratings.values():
    #print(name, rating)
