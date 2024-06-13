from enum import Enum, auto
import functools
import re
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import warnings
import numpy as np


class DecayMethod(Enum):
    ZERO = auto()
    BASE_RATING = auto()
    MIN_BASE_CURRENT = auto()


# Get API token from .env file
with open(".env") as f:
    for line in f:
        if "FOOTBALL-DATA-API-KEY" in line:
            token = line.split("=")[1].strip()
            break

headers = {"X-Auth-Token": token}


def get_data_by_year(year):
    uri = f"https://api.football-data.org/v4/competitions/2021/matches?season={year}"
    response = requests.get(uri, headers=headers)
    return response.json()


def build_data_by_year(year: int, title_case: bool = True) -> pd.DataFrame:
    df = pd.DataFrame(get_data_by_year(year)["matches"])

    # Drop columns we don"t need
    df = df[["utcDate", "status", "matchday", "homeTeam", "awayTeam", "score"]]

    # Add a column for the season after the index of "utcDate"
    df.insert(df.columns.get_loc("utcDate") + 1, "season", year)

    # Parse the homeTeam and awayTeam and get the names or ids from the column
    df["home"] = df["homeTeam"].apply(lambda x: x["name"])
    df["away"] = df["awayTeam"].apply(lambda x: x["name"])
    df = df.drop(columns=["homeTeam", "awayTeam"])

    # Get home team and away team scores from the dataframe, drop score column
    df["homeScore"] = df["score"].apply(lambda x: x["fullTime"]["home"])
    df["awayScore"] = df["score"].apply(lambda x: x["fullTime"]["away"])
    df = df.drop(columns=["score"])

    # Determine the winner of the match
    df["homeOutcome"] = 1
    df["awayOutcome"] = 1
    df.loc[df["homeScore"] > df["awayScore"], "homeOutcome"] = 3
    df.loc[df["homeScore"] > df["awayScore"], "awayOutcome"] = 0
    df.loc[df["awayScore"] > df["homeScore"], "awayOutcome"] = 3
    df.loc[df["awayScore"] > df["homeScore"], "homeOutcome"] = 0

    # Convert utcDate to datetime
    df["utcDate"] = pd.to_datetime(df["utcDate"])

    # Function to convert camel case to title case
    def camel_to_title(camel_str):
        title_str = re.sub("([A-Z])", r" \1", camel_str)
        return title_str.title()

    # Function to convert camel case to snake case
    def camel_to_snake(camel_str):
        snake_str = re.sub("([A-Z])", r"_\1", camel_str)
        return snake_str.lower()

    case_func = camel_to_title if title_case else camel_to_snake

    # Apply the function to each column name
    df.columns = [case_func(col) for col in df.columns]

    return df


def value_str_to_float(club_value: str) -> float:

    return float(
        club_value.replace("€", "")
        .replace("k", "000")
        .replace("m", "0000")
        .replace("bn", "0000000")
        .replace(".", "")
    )


@functools.lru_cache
def get_club_value_at_season(club: str, season: int) -> float:
    response = requests.get(
        f"https://transfermarkt-api.fly.dev/clubs/search/{club.replace(' FC', '')}",
        headers={"accept": "application/json"},
    )
    club_id = response.json()["results"][0]["id"]

    response = requests.get(
        f"https://transfermarkt-api.fly.dev/clubs/{club_id}/players?season_id={season}",
        headers={"accept": "application/json"},
    )
    players = response.json()["players"]

    def get_market_value(player):
        try:
            return value_str_to_float(player["marketValue"])
        except KeyError:
            return 0

    return sum([get_market_value(player) for player in players])


@functools.lru_cache
def get_club_value(club: str) -> float:
    response = requests.get(
        f"https://transfermarkt-api.fly.dev/clubs/search/{club.replace(' FC', '')}",
        headers={"accept": "application/json"},
    )
    club_value: str = response.json()["results"][0]["marketValue"]
    return value_str_to_float(club_value)


def build_elo_df_from_dict(
    elo_dict: dict[str, float], adjustment_factor: float
) -> pd.DataFrame:
    # Build an elo dataframe
    elo_df = pd.DataFrame(elo_dict.items(), columns=["Team", "Elo"]).set_index("Team")

    # Get club values
    elo_df["Club Value"] = elo_df.index.map(get_club_value)

    # Normalize the club value column to [0, 1]
    elo_df["Normalized Club Value"] = (
        elo_df["Club Value"] - elo_df["Club Value"].min()
    ) / (elo_df["Club Value"].max() - elo_df["Club Value"].min())

    # Apply exponential transformation
    elo_df["Exponential Club Value"] = (
        np.exp(elo_df["Normalized Club Value"]) - 1
    )  # Shift to start from 0

    # Re-normalize to [0, 1]
    elo_df["Normalized Exponential Club Value"] = (
        elo_df["Exponential Club Value"] - elo_df["Exponential Club Value"].min()
    ) / (
        elo_df["Exponential Club Value"].max() - elo_df["Exponential Club Value"].min()
    )

    # Adjust the ELO ratings based on the normalized exponential club values
    elo_df["Adjusted Elo"] = (
        elo_df["Elo"] + adjustment_factor * elo_df["Normalized Exponential Club Value"]
    )

    # Sort by adjusted ELO
    elo_df = elo_df.sort_values(by="Adjusted Elo", ascending=False)

    return elo_df


def build_elo_between_seasons(
    df: pd.DataFrame,
    df_2023: pd.DataFrame,
    club_value_adjustment_factor: float,
) -> pd.DataFrame:
    # Get the ending ELO ratings for the teams in the 2022 season
    results = get_season_results(df)
    # Find teams that have been relegated/promoted by taking a difference of the two dataframes
    df_teams = pd.concat([df["Home"], df["Away"]]).unique()
    df_2023_teams = pd.concat([df_2023["Home"], df_2023["Away"]]).unique()

    relegated_teams = set(df_teams) - set(df_2023_teams)
    promoted_teams = set(df_2023_teams) - set(df_teams)
    teams_with_baseline = set(df_teams) & set(df_2023_teams)

    # Find the average ending ELO rating for the teams that have been relegated
    relegated_elo = results.loc[list(relegated_teams), "Total Elo"].max()

    # Set the starting ELO rating for the promoted teams to the average ending ELO rating of the relegated teams
    elo = {team: relegated_elo for team in promoted_teams}

    # Set the starting ELO rating for the teams that have been in the league for both seasons to their ending ELO rating
    elo.update(results.loc[list(teams_with_baseline), "Total Elo"].to_dict())

    # Divide Elo by 2
    elo = {team: elo[team] / 2 for team in elo}

    elo_df = build_elo_df_from_dict(elo, club_value_adjustment_factor)

    return elo_df


def get_elo_dict_from_df(df: pd.DataFrame) -> dict[str, float]:
    return df["Adjusted Elo"].to_dict()


def update_elo_win(winner_elo: float, loser_elo: float, k: int):
    expected_win = 1.0 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    change = k * (1 - expected_win)
    return winner_elo + change, loser_elo - change


def update_elo_draw(home_elo: float, away_elo: float, k: int):
    expected_home_win = 1.0 / (1 + 10 ** ((away_elo - home_elo) / 400))
    change = k * (0.5 - expected_home_win)
    return home_elo + change, away_elo - change


def process_fixture_results(
    df: pd.DataFrame,
    k: int,
    half_life: int,
    club_value_adjustment: float,
    decay_method: DecayMethod,
    elo: dict[str, float] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if elo is None:
        # Initialize elo ratings for each team
        elo = {team: 1500.0 for team in df["Home"].unique()}

        # Adjust Elo based on club value
        elo = build_elo_df_from_dict(elo, club_value_adjustment)[
            "Adjusted Elo"
        ].to_dict()

    # Process matches and update ELO ratings
    for index, row in df.iterrows():
        home_team, away_team = row["Home"], row["Away"]

        # Get current ELO ratings
        home_elo = elo[home_team]
        away_elo = elo[away_team]

        if row["Home Score"] > row["Away Score"]:  # Home team won
            home_elo, away_elo = update_elo_win(home_elo, away_elo, k)
        elif row["Away Score"] > row["Home Score"]:  # Away team won
            away_elo, home_elo = update_elo_win(away_elo, home_elo, k)
        else:  # Draw
            home_elo, away_elo = update_elo_draw(home_elo, away_elo, k)

        # Time-decay ELO ratings
        home_elo = apply_decay_factor(home_elo, half_life, decay_method)
        away_elo = apply_decay_factor(away_elo, half_life, decay_method)

        # Update ELO ratings in the dataframe and dictionary
        elo[home_team] = home_elo
        elo[away_team] = away_elo

        df.at[index, "Home Elo"] = elo[home_team]
        df.at[index, "Away Elo"] = elo[away_team]

    # Determine outcomes: 3 for win, 1 for draw, 0 for loss
    df["Home Outcome"] = 1
    df["Away Outcome"] = 1
    df.loc[df["Home Score"] > df["Away Score"], "Home Outcome"] = 3
    df.loc[df["Home Score"] > df["Away Score"], "Away Outcome"] = 0
    df.loc[df["Away Score"] > df["Home Score"], "Away Outcome"] = 3
    df.loc[df["Away Score"] > df["Home Score"], "Home Outcome"] = 0

    results = get_season_results(df)

    return df, results


def apply_decay_factor(elo: float, half_life: int, decay_method: DecayMethod) -> float:
    # Calculate the decay factor
    decay_factor = 0.5 ** (1 / half_life)

    # Return the scaled Elo rating
    match decay_method:
        case DecayMethod.ZERO:
            return elo * decay_factor
        case DecayMethod.BASE_RATING:
            return elo * decay_factor + 1500 * (1 - decay_factor)
        case DecayMethod.MIN_BASE_CURRENT:
            return elo * decay_factor + min(1500, elo) * (1 - decay_factor)


def simulate_match(
    home_elo, away_elo, model: RandomForestClassifier, scaler: StandardScaler
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = scaler.transform([[home_elo, away_elo]])
    probabilities = model.predict_proba(x)
    return np.random.choice([0, 1, 3], p=probabilities[0])


def get_season_results(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate results
    home_results = df.groupby("Home").agg({"Home Outcome": "sum", "Home Elo": "last"})
    away_results = df.groupby("Away").agg({"Away Outcome": "sum", "Away Elo": "last"})
    results = home_results.join(away_results, how="outer").fillna(0)
    results["Total Outcome"] = results["Home Outcome"] + results["Away Outcome"]
    results["Total Elo"] = results["Home Elo"] + results["Away Elo"]

    # Sort by total outcome and total ELO
    results = results.sort_values(by=["Total Outcome", "Total Elo"], ascending=False)

    # Rename Home to Team
    results.index.name = "Team"
    return results


def simulate_season(
    df: pd.DataFrame,
    elo: dict[str, float],
    model,
    scalar,
    k: int,
    half_life: int,
    decay_method: DecayMethod,
) -> pd.DataFrame:
    df_2023 = df.copy()

    # Get home and away elo
    home_elo_dict = {team: elo for team, elo in elo.items()}
    away_elo_dict = {team: elo for team, elo in elo.items()}

    # Can run matches in a match week concurrently in the future
    for index, row in df_2023.iterrows():
        home_team, away_team = row["Home"], row["Away"]

        # Get current Elo ratings
        home_elo = home_elo_dict[home_team]
        away_elo = away_elo_dict[away_team]

        # Simulate match and update ELO ratings
        outcome = simulate_match(home_elo, away_elo, model, scalar)

        match outcome:
            case 3:  # Home team won
                home_elo, away_elo = update_elo_win(home_elo, away_elo, k)
                df_2023.at[index, "Home Outcome"] = 3
                df_2023.at[index, "Away Outcome"] = 0
            case 0:  # Away team won
                away_elo, home_elo = update_elo_win(away_elo, home_elo, k)
                df_2023.at[index, "Away Outcome"] = 3
                df_2023.at[index, "Home Outcome"] = 0
            case 1:  # Draw
                home_elo, away_elo = update_elo_draw(home_elo, away_elo, k)
                df_2023.at[index, "Home Outcome"] = 1
                df_2023.at[index, "Away Outcome"] = 1

        # Time-decay Elo ratings
        home_elo = apply_decay_factor(home_elo, half_life, decay_method)
        away_elo = apply_decay_factor(away_elo, half_life, decay_method)

        # Update Elo ratings in the dictionary
        home_elo_dict[home_team] = home_elo
        away_elo_dict[away_team] = away_elo

        # Update actual results
        if row["Home Score"] > row["Away Score"]:
            df_2023.at[index, "Actual Home Outcome"] = 3
            df_2023.at[index, "Actual Away Outcome"] = 0
        elif row["Away Score"] > row["Home Score"]:
            df_2023.at[index, "Actual Home Outcome"] = 0
            df_2023.at[index, "Actual Away Outcome"] = 3
        else:
            df.at[index, "Actual Home Outcome"] = 1
            df.at[index, "Actual Away Outcome"] = 1

        df_2023.at[index, "Home Elo"] = home_elo_dict[home_team]
        df_2023.at[index, "Away Elo"] = away_elo_dict[away_team]

    return df_2023


# Define a function to simulate a season and get results
def simulate_and_get_results(i, df, elo, model, scaler, k, half_life, decay_method):
    simulated_df = simulate_season(df, elo, model, scaler, k, half_life, decay_method)
    results = get_season_results(simulated_df)
    results["Season"] = i
    return results
