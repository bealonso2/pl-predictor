import copy
import difflib
import functools
import json
import os
import re
import uuid
import warnings
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import SlowDB

S3_BUCKET = "pl-prediction"
TEST_PREFIX = "2024_test"

S3_PREFIX = os.getenv("S3_PREFIX", TEST_PREFIX)

S3_INFERENCE_DATA_DB_KEY = f"{S3_PREFIX}/data.db"
S3_INFERENCE_RESULTS_DB_KEY = f"{S3_PREFIX}/results.db"
S3_MODEL_KEY = f"{S3_PREFIX}/random_forest.joblib"
S3_SCALER_KEY = f"{S3_PREFIX}/standard_scaler.joblib"
S3_PARAMS_KEY = f"{S3_PREFIX}/best_params.json"

# Number of matches to consider for form
FORM_MATCHES = 5


class DecayMethod(Enum):
    ZERO = 0
    BASE_RATING = 1
    MIN_BASE_CURRENT = 2


@dataclass
class BestParams:
    k: int
    half_life: int
    club_value_adjustment: float
    decay_method: DecayMethod


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


@functools.lru_cache
def get_team_crests_from_season(season: int) -> pd.DataFrame:
    uri = f"https://api.football-data.org/v4/competitions/PL/teams?season={season}"
    response = requests.get(uri, headers=headers)
    teams = response.json()["teams"]

    # Create a dataframe from the teams
    teams_to_crest = {team["name"]: team["crest"] for team in teams}
    return pd.DataFrame(teams_to_crest.items(), columns=["team", "crest"])


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

    # All matches that are not "FINISHED" should have NA for the scores and outcomes
    df.loc[
        df["status"] != "FINISHED",
        ["homeScore", "awayScore", "homeOutcome", "awayOutcome"],
    ] = np.nan

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


def find_manager(managers_df: pd.DataFrame, team: str, date: pd.Timestamp) -> str:
    # Try to find the closest team name match
    clubs = managers_df["club"].unique()
    team = difflib.get_close_matches(team, clubs)[0]
    managers_df = managers_df[managers_df["club"] == team]

    # Get the managers for the team at the given date
    try:
        return managers_df[
            (managers_df["start"] <= date.to_datetime64())
            & (managers_df["end"] >= date.to_datetime64())
        ]["manager"].iloc[0]
    except IndexError:
        # Print the team and date for which the manager was not found
        print(f"Manager not found for {team} at {date}")
        return ""


def db_add_managers_to_df(df: pd.DataFrame) -> pd.DataFrame:
    with SlowDB.connect(S3_BUCKET, S3_INFERENCE_DATA_DB_KEY, readonly=True) as conn:
        managers_df = pd.read_sql("SELECT * FROM premier_league_managers", conn)

    # Convert utc_date to datetime
    df["utc_date"] = pd.to_datetime(
        df["utc_date"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    ).dt.tz_convert(None)

    # Convert start and end dates to datetime
    managers_df["start"] = pd.to_datetime(
        managers_df["start"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    ).fillna(pd.Timestamp.now())

    # Fill missing end dates with the latest date
    managers_df["end"] = pd.to_datetime(
        managers_df["end"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    ).fillna(df["utc_date"].max())

    # Add one day to the end date
    managers_df["end"] = managers_df["end"] + pd.Timedelta(days=1)

    # For each fixture, find the manager of the home team
    df["home_manager"] = df.apply(
        lambda x: find_manager(managers_df, x["home"], x["utc_date"]), axis=1
    )

    # For each fixture, find the manager of the away team
    df["away_manager"] = df.apply(
        lambda x: find_manager(managers_df, x["away"], x["utc_date"]), axis=1
    )

    return df


def get_league_table_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns to the dataframe to indicate the league table position of
    the home and away teams at the time of the fixture, for the given portion of
    the dataframe.
    """
    df = df.copy()

    for i, row in df.iterrows():
        # Get the portion of the dataframe up to the date of the fixture
        portion: pd.DataFrame = df[df["utc_date"] < row["utc_date"]]

        home_results = portion.groupby("home").agg({"home_outcome": "sum"})
        away_results = portion.groupby("away").agg({"away_outcome": "sum"})
        results = home_results.join(away_results, how="outer").fillna(0)
        results["total_outcome"] = results["home_outcome"] + results["away_outcome"]

        # Rank the teams by total outcome
        results = results.sort_values(by="total_outcome", ascending=False)

        if row["home"] not in results.index or row["away"] not in results.index:
            # Get the number of unique values in the total outcome column
            unique_outcomes = results[results["total_outcome"] > 0][
                "total_outcome"
            ].nunique()

            # Next free position is the number of unique outcomes + 1
            next_free_position = unique_outcomes + 1
            df.loc[i, "home_position"] = next_free_position
            df.loc[i, "away_position"] = next_free_position
            continue

        # Get the league table position of the home team at the time of the fixture including ties
        home_total_outcome = results.loc[row["home"], "total_outcome"]
        home_index = results.index.get_loc(
            results[results["total_outcome"] == home_total_outcome].iloc[0].name
        )
        df.loc[i, "home_position"] = home_index + 1

        # Get the league table position of the away team at the time of the fixture including ties
        away_total_outcome = results.loc[row["away"], "total_outcome"]
        away_index = results.index.get_loc(
            results[results["total_outcome"] == away_total_outcome].iloc[0].name
        )
        df.loc[i, "away_position"] = away_index + 1

    # Convert the columns to integers
    df["home_position"] = df["home_position"].astype(int)
    df["away_position"] = df["away_position"].astype(int)
    return df


def add_manager_tenure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Initialize the manager tenure columns
    df["home_manager_tenure"] = 0
    df["away_manager_tenure"] = 0

    manager_df = df[["home_manager", "away_manager"]]
    for i, row in manager_df.iterrows():
        # Get the portion of the dataframe up to the date of the fixture
        df_to_date = manager_df.loc[:i, :].copy()

        # Count the number of times the home manager has appeared in the dataframe
        home_manager_values = df_to_date["home_manager"].value_counts().to_dict()
        away_manager_values = df_to_date["away_manager"].value_counts().to_dict()

        # Get the tenure of the home manager
        home_manager = row["home_manager"]
        df.loc[i, "home_manager_tenure"] = home_manager_values.get(
            home_manager, 0
        ) + away_manager_values.get(home_manager, 0)

        # Get the tenure of the away manager
        away_manager = row["away_manager"]
        df.loc[i, "away_manager_tenure"] = home_manager_values.get(
            away_manager, 0
        ) + away_manager_values.get(away_manager, 0)

    # Set tenure columns to integers
    df["home_manager_tenure"] = df["home_manager_tenure"].astype(int)
    df["away_manager_tenure"] = df["away_manager_tenure"].astype(int)

    return df


def add_form(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Initialize the form columns
    df["home_form"] = 0
    df["away_form"] = 0

    for i, row in df.iterrows():
        # Get the portion of the dataframe up to the date of the fixture
        portion = df.loc[:i, :]

        # Get the total outcome of the home and away teams
        all_home_results: pd.DataFrame = portion[
            (portion["home"] == row["home"]) | (portion["away"] == row["home"])
        ]
        all_home_results = all_home_results[all_home_results.index != i]
        home_results = all_home_results.tail(FORM_MATCHES)

        if home_results.empty:
            home_form = 0

        # Keep only the results of the home team
        home_form = (
            home_results[home_results["home"] == row["home"]]["home_outcome"].sum()
            + home_results[home_results["away"] == row["home"]]["away_outcome"].sum()
        )

        all_away_results: pd.DataFrame = portion[
            (portion["home"] == row["away"]) | (portion["away"] == row["away"])
        ]
        all_away_results = all_away_results[all_away_results.index != i]
        away_results = all_away_results.tail(FORM_MATCHES)

        if away_results.empty:
            away_form = 0

        # Keep only the results of the away team
        away_form = (
            away_results[away_results["home"] == row["away"]]["home_outcome"].sum()
            + away_results[away_results["away"] == row["away"]]["away_outcome"].sum()
        )

        # Add the form to the dataframe
        df.loc[i, "home_form"] = home_form
        df.loc[i, "away_form"] = away_form

    return df


def db_get_data_for_latest_season() -> pd.DataFrame:
    with SlowDB.connect(S3_BUCKET, S3_INFERENCE_DATA_DB_KEY, readonly=True) as conn:
        df = pd.read_sql(
            "SELECT * FROM football_data_season_results WHERE season = (SELECT MAX(season) FROM football_data_season_results)",
            conn,
        )

    # Add other columns to the dataframe
    return process_data_by_df(df)


def db_get_data_by_year(year: int) -> pd.DataFrame:
    with SlowDB.connect(S3_BUCKET, S3_INFERENCE_DATA_DB_KEY, readonly=True) as conn:
        df = pd.read_sql(
            f"SELECT * FROM football_data_season_results WHERE season = {year}", conn
        )

    # Add other columns to the dataframe
    return process_data_by_df(df)


def process_data_by_df(df: pd.DataFrame) -> pd.DataFrame:
    # Add managers to the dataframe
    df = db_add_managers_to_df(df)

    # Make sure the dataframe is sorted by date
    df = df.sort_values(by="utc_date").reset_index(drop=True)

    # Add manager tenure to the dataframe
    df = add_manager_tenure(df)

    # Add form to the dataframe
    df = add_form(df)

    # Add league table position to the dataframe
    df = get_league_table_position(df)

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
def db_get_club_value_at_season(club: str, season: int) -> float:
    with SlowDB.connect(S3_BUCKET, S3_INFERENCE_DATA_DB_KEY, readonly=True) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT value FROM transfermarkt_club_values WHERE club = '{club}' AND season = {season}"
        )
        club_value = cursor.fetchone()[0]
    return club_value


@functools.lru_cache
def db_get_club_value(club: str) -> float:
    with SlowDB.connect(S3_BUCKET, S3_INFERENCE_DATA_DB_KEY, readonly=True) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT value FROM transfermarkt_club_values WHERE club = '{club}' ORDER BY season DESC"
        )
        club_value = cursor.fetchone()[0]
    return club_value


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
    elo_dict: dict[str, float], adjustment_factor: float, season: int = None
) -> pd.DataFrame:
    # Build an elo dataframe
    elo_df = pd.DataFrame(elo_dict.items(), columns=["team", "elo"]).set_index("team")

    # Define a club value function based on whether a season is provided
    club_value_function = (
        functools.partial(db_get_club_value_at_season, season=season)
        if season
        else db_get_club_value
    )

    # Get club values
    elo_df["club_value"] = elo_df.index.map(club_value_function)

    # Normalize the club value column to [0, 1]
    elo_df["normalized_club_value"] = (
        elo_df["club_value"] - elo_df["club_value"].min()
    ) / (elo_df["club_value"].max() - elo_df["club_value"].min())

    # Apply exponential transformation
    elo_df["exponential_club_value"] = (
        np.exp(elo_df["normalized_club_value"]) - 1
    )  # Shift to start from 0

    # Re-normalize to [0, 1]
    elo_df["normalized_exponential_club_value"] = (
        elo_df["exponential_club_value"] - elo_df["exponential_club_value"].min()
    ) / (
        elo_df["exponential_club_value"].max() - elo_df["exponential_club_value"].min()
    )

    # Adjust the ELO ratings based on the normalized exponential club values
    elo_df["adjusted_elo"] = (
        elo_df["elo"] + adjustment_factor * elo_df["normalized_exponential_club_value"]
    )

    # Sort by adjusted ELO
    elo_df = elo_df.sort_values(by="adjusted_elo", ascending=False)

    return elo_df


def process_team_to_points(df: pd.DataFrame, teams: list) -> pd.DataFrame:
    # If total outcome is not present, set it to 0
    if "total_outcome" not in df.columns:
        # Set the total outcome to 0 for all teams
        df = pd.DataFrame(index=teams, data={"total_outcome": 0})

        # Rename the index to team
        df.index.name = "team"

    # Isolate the total outcome column
    team_to_points = df[["total_outcome"]]

    # Rename the column to points
    team_to_points = team_to_points.rename(columns={"total_outcome": "points"})

    # Change the data type of the points column to int
    team_to_points["points"] = team_to_points["points"].astype(int)

    return team_to_points


def build_elo_before_season(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Download the best parameters from S3
    best_params = download_best_params_from_s3()

    # Get the season of the dataframe
    season_ids = df["season"].unique()
    assert len(season_ids) == 1, "There should be only one season in the dataframe"
    season_id = season_ids[0]

    # Get the ending ELO ratings for the teams in the season

    previous_season_df = db_get_data_by_year(season_id - 1)

    # Process the previous season results
    df, results = process_fixture_results(
        df,
        best_params.k,
        best_params.half_life,
        best_params.club_value_adjustment,
        best_params.decay_method,
        None,
    )
    previous_season_df, _ = process_fixture_results(
        previous_season_df,
        best_params.k,
        best_params.half_life,
        best_params.club_value_adjustment,
        best_params.decay_method,
        None,
    )

    # Get a list of unique teams in the current season
    teams = pd.concat([df["home"], df["away"]]).unique()

    # Get the ending ELO ratings for the teams in the season
    return (
        build_elo_between_seasons(
            previous_season_df, df, best_params.club_value_adjustment
        ),
        process_team_to_points(results, teams),
    )


def build_elo_between_seasons(
    df: pd.DataFrame,
    df_2023: pd.DataFrame,
    club_value_adjustment_factor: float,
) -> pd.DataFrame:
    # Get the ending ELO ratings for the teams in the 2022 season
    results = get_season_results(df)
    # Find teams that have been relegated/promoted by taking a difference of the two dataframes
    df_teams = pd.concat([df["home"], df["away"]]).unique()
    df_2023_teams = pd.concat([df_2023["home"], df_2023["away"]]).unique()

    relegated_teams = set(df_teams) - set(df_2023_teams)
    promoted_teams = set(df_2023_teams) - set(df_teams)
    teams_with_baseline = set(df_teams) & set(df_2023_teams)

    # Find the average ending ELO rating for the teams that have been relegated
    relegated_elo = results.loc[list(relegated_teams), "total_elo"].max()

    # Set the starting ELO rating for the promoted teams to the average ending ELO rating of the relegated teams
    elo = {team: relegated_elo for team in promoted_teams}

    # Set the starting ELO rating for the teams that have been in the league for both seasons to their ending ELO rating
    elo.update(results.loc[list(teams_with_baseline), "total_elo"].to_dict())

    # Divide Elo by 2
    elo = {team: elo[team] / 2 for team in elo}

    # Scale the Elo back towards 1500 by 33%
    elo = {team: 0.5 * elo[team] + 0.5 * 1500 for team in elo}

    # Gets Elo based on current club value
    elo_df = build_elo_df_from_dict(elo, club_value_adjustment_factor)

    return elo_df


def get_elo_dict_from_df(df: pd.DataFrame) -> dict[str, float]:
    return df["adjusted_elo"].to_dict()


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
        elo = {team: 1500.0 for team in df["home"].unique()}

        # Get the season from the first fixture
        season = df["season"].iloc[0]

        # Adjust Elo based on club value
        elo = build_elo_df_from_dict(elo, club_value_adjustment, season)[
            "adjusted_elo"
        ].to_dict()

    # If no matches have been played, return the dataframe with the initial ELO ratings
    if not df[df["status"] == "FINISHED"].shape[0]:
        df["home_elo"] = df["home"].map(elo)
        df["away_elo"] = df["away"].map(elo)

        return df, pd.DataFrame()

    # Process matches and update ELO ratings
    for index, row in df[df["status"] == "FINISHED"].iterrows():
        home_team, away_team = row["home"], row["away"]

        # Get current ELO ratings
        home_elo = elo[home_team]
        away_elo = elo[away_team]

        if row["home_score"] > row["away_score"]:  # Home team won
            home_elo, away_elo = update_elo_win(home_elo, away_elo, k)
        elif row["away_score"] > row["home_score"]:  # Away team won
            away_elo, home_elo = update_elo_win(away_elo, home_elo, k)
        else:  # Draw
            home_elo, away_elo = update_elo_draw(home_elo, away_elo, k)

        # Time-decay ELO ratings
        home_elo = apply_decay_factor(home_elo, half_life, decay_method)
        away_elo = apply_decay_factor(away_elo, half_life, decay_method)

        # Update ELO ratings in the dataframe and dictionary
        elo[home_team] = home_elo
        elo[away_team] = away_elo

        df.at[index, "home_elo"] = elo[home_team]
        df.at[index, "away_elo"] = elo[away_team]

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


def process_mathematical_table_positions(
    df: pd.DataFrame, team_to_points: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate matches played, max points, min position, and max position for each team
    in the team_to_points dataframe using the finished matches in the dataframe.
    """
    df = df.copy()

    number_of_teams = len(team_to_points)

    # Initialize the mathematical table position columns
    team_to_points["matches_played"] = 0
    team_to_points["max_points"] = 0
    team_to_points["min_possible_position"] = number_of_teams
    team_to_points["max_possible_position"] = 1

    # Calculate the total matches played over the season
    total_matches = (number_of_teams - 1) * 2

    # Calculate the matches played for each team
    for _, match in df[df["status"] == "FINISHED"].iterrows():
        team_to_points.loc[match["home"], "matches_played"] += 1
        team_to_points.loc[match["away"], "matches_played"] += 1

    # Calculate the max points for each team
    team_to_points["max_points"] = (
        total_matches - team_to_points["matches_played"]
    ) * 3 + team_to_points["points"]

    # Calculate the min and max possible positions for each team
    for team in team_to_points.index:
        team_points = team_to_points.loc[team, "points"]
        team_max_points = team_to_points.loc[team, "max_points"]

        # Calculate the min possible position
        team_to_min_points = team_to_points["max_points"].to_dict()
        # Pop the current team from the dictionary
        team_to_min_points.pop(team, None)
        team_to_points.loc[team, "min_possible_position"] = (
            # Sum the number of teams with max points ahead of this team's current points
            # Assume the other team will overcome this team's goal difference
            sum(1 for points in team_to_min_points.values() if points >= team_points)
            + 1
        )

        # Calculate the max possible position
        team_to_max_points = team_to_points["points"].to_dict()
        # Pop the current team from the dictionary
        team_to_max_points.pop(team, None)
        team_to_points.loc[team, "max_possible_position"] = (
            # Sum the number of teams ahead of this team's max points
            # Ties are assumed to be overcome by goal difference
            sum(1 for points in team_to_max_points.values() if points > team_max_points)
            # Add 1 to make this a 1-indexed position (a league table position)
            + 1
        )

    return team_to_points


def get_match_probabilities(
    home_elo: float,
    away_elo: float,
    home_position: int,
    away_position: int,
    home_manager_tenure: int,
    away_manager_tenure: int,
    home_form: int,
    away_form: int,
    model: RandomForestClassifier,
    scaler: StandardScaler,
) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = scaler.transform(
            [
                [
                    home_elo,
                    away_elo,
                    home_position,
                    away_position,
                    home_manager_tenure,
                    away_manager_tenure,
                    home_form,
                    away_form,
                ]
            ]
        )
    probabilities = model.predict_proba(x)
    return probabilities[0]


def simulate_match(
    home_elo: float,
    away_elo: float,
    home_position: int,
    away_position: int,
    home_manager_tenure: int,
    away_manager_tenure: int,
    home_form: int,
    away_form: int,
    model: RandomForestClassifier,
    scaler: StandardScaler,
) -> int:
    probabilities = get_match_probabilities(
        home_elo,
        away_elo,
        home_position,
        away_position,
        home_manager_tenure,
        away_manager_tenure,
        home_form,
        away_form,
        model,
        scaler,
    )
    return np.random.choice([0, 1, 3], p=probabilities)


def get_season_results(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate results
    home_results = df.groupby("home").agg({"home_outcome": "sum", "home_elo": "last"})
    away_results = df.groupby("away").agg({"away_outcome": "sum", "away_elo": "last"})
    results = home_results.join(away_results, how="outer").fillna(0)
    results["total_outcome"] = results["home_outcome"] + results["away_outcome"]
    results["total_elo"] = results["home_elo"] + results["away_elo"]

    # Sort by total outcome and total ELO
    results = results.sort_values(by=["total_outcome", "total_elo"], ascending=False)

    # Rename Home to Team
    results.index.name = "team"
    return results


def get_starts_of_next_matchweeks(df: pd.DataFrame) -> list[pd.Timestamp]:
    """Get the next two matchweeks from the dataframe."""
    # Make a copy of the dataframe
    df_next_sim = df.copy()

    # Filter out all utc_date that are in the past and matches that do not have a status of 'TIMED'
    df_next_sim = df_next_sim[
        (df_next_sim["utc_date"] > pd.Timestamp.now())
        & (df_next_sim["status"] == "TIMED")
    ]

    # Convert all utc_date to the local date, drop the time component
    df_next_sim["local_date"] = df_next_sim["utc_date"].dt.floor("D")

    # Get the unique local dates
    unique_local_dates = df_next_sim["local_date"].unique()

    # Starting from the back of the list, remove future dates that don't have a match
    # one day in front of it. But if the date is standalone, keep it.
    start_of_match_week_dates = []

    for i in range(len(unique_local_dates) - 1, -1, -1):
        if unique_local_dates[i] in start_of_match_week_dates:
            continue

        # Check to see if the date has a match one day in front of it
        if unique_local_dates[i] - pd.DateOffset(1) not in unique_local_dates:
            start_of_match_week_dates.append(unique_local_dates[i])

    # Make sure tomorrow is not in the list, because we are running the simulation for tomorrow
    start_of_match_week_dates = [
        date
        for date in start_of_match_week_dates
        if date != pd.Timestamp.now().floor("D") + pd.DateOffset(1)
    ]

    # Send an alert if there are no matches to simulate
    if not start_of_match_week_dates:
        print("No matches to simulate")

    # Sort the dates in ascending order (earliest date first)
    start_of_match_week_dates = sorted(start_of_match_week_dates)

    # Return the entire list of dates
    return start_of_match_week_dates


@dataclass
class CurrentSeasonState:
    df: pd.DataFrame
    home_elo: dict[str, float]
    away_elo: dict[str, float]
    team_to_points: dict[str, int]
    team_to_form: dict[str, deque[int]]
    model: RandomForestClassifier
    scaler: StandardScaler
    k: int
    half_life: int
    decay_method: DecayMethod

    def copy(self) -> "CurrentSeasonState":
        return CurrentSeasonState(
            df=self.df.copy(),
            home_elo=self.home_elo.copy(),
            away_elo=self.away_elo.copy(),
            team_to_points=self.team_to_points.copy(),
            team_to_form=copy.deepcopy(self.team_to_form),
            model=copy.deepcopy(self.model),
            scaler=copy.deepcopy(self.scaler),
            k=self.k,
            half_life=self.half_life,
            decay_method=self.decay_method,
        )


def process_finished_matches(
    df: pd.DataFrame,
    elo: dict[str, float],
    model: RandomForestClassifier,
    scaler: StandardScaler,
    k: int,
    half_life: int,
    decay_method: DecayMethod,
) -> CurrentSeasonState:
    df = df.copy()

    # Get home and away elo
    home_elo_dict = {team: elo for team, elo in elo.items()}
    away_elo_dict = {team: elo for team, elo in elo.items()}

    # Initialize a dictionary to store points for each team
    team_to_points = {team: 0 for team in df["home"].unique()}

    # Initialize a dictionary to store the form of each team
    # No need to update the dictionary as deque is mutable and will update in place
    team_to_form = {team: deque(maxlen=FORM_MATCHES) for team in df["home"].unique()}

    # Process matches and update ELO ratings that have been played
    for index, row in df[df["status"] == "FINISHED"].iterrows():
        # Get the unique values of team_to_points and sort them in descending order
        team_to_points_ranks = list(set(team_to_points.values()))
        team_to_points_values = sorted(team_to_points_ranks, reverse=True)

        # Get the points for the home and away teams
        home_points = team_to_points[row["home"]]
        away_points = team_to_points[row["away"]]

        # Update home and away league table positions based on index of team in team_to_points
        home_position = team_to_points_values.index(home_points) + 1
        away_position = team_to_points_values.index(away_points) + 1

        # Add home and away positions to the dataframe
        df.loc[index, "home_position"] = home_position
        df.loc[index, "away_position"] = away_position

        # Get current form of each team
        home_form = team_to_form[row["home"]]
        away_form = team_to_form[row["away"]]

        # Add form to the dataframe
        df.loc[index, "home_form"] = sum(home_form)
        df.loc[index, "away_form"] = sum(away_form)

        # Get the home and away teams
        home_team, away_team = row["home"], row["away"]

        # Get current Elo ratings
        home_elo = home_elo_dict[home_team]
        away_elo = away_elo_dict[away_team]

        # Process finished results
        if row["home_score"] > row["away_score"]:
            home_elo, away_elo = update_elo_win(home_elo, away_elo, k)
            home_outcome = 3
            away_outcome = 0
        elif row["away_score"] > row["home_score"]:
            away_elo, home_elo = update_elo_win(away_elo, home_elo, k)
            home_outcome = 0
            away_outcome = 3
        else:
            home_elo, away_elo = update_elo_draw(home_elo, away_elo, k)
            home_outcome = 1
            away_outcome = 1

        # Update the dataframe with the real or simulated results
        df.at[index, "home_elo"] = home_elo
        df.at[index, "away_elo"] = away_elo
        df.at[index, "home_outcome"] = home_outcome
        df.at[index, "away_outcome"] = away_outcome

        # Time-decay Elo ratings
        home_elo = apply_decay_factor(home_elo, half_life, decay_method)
        away_elo = apply_decay_factor(away_elo, half_life, decay_method)

        # Update Elo ratings in the dictionary
        home_elo_dict[home_team] = home_elo
        away_elo_dict[away_team] = away_elo

        # Update points for each team
        team_to_points[home_team] += home_outcome
        team_to_points[away_team] += away_outcome

        # Add points to the form of each team
        home_form.append(home_outcome)
        away_form.append(away_outcome)

    return CurrentSeasonState(
        df=df,
        home_elo=home_elo_dict,
        away_elo=away_elo_dict,
        team_to_points=team_to_points,
        team_to_form=team_to_form,
        model=model,
        scaler=scaler,
        k=k,
        half_life=half_life,
        decay_method=decay_method,
    )


def get_upcoming_probabilities(
    current_season_state: CurrentSeasonState,
    list_of_matchweeks: list[pd.Timestamp],
) -> pd.DataFrame:
    current_season_state = current_season_state.copy()

    # Remove finished matches from the dataframe
    current_season_state_df = current_season_state.df
    df_upcoming = current_season_state_df[
        current_season_state_df["status"] != "FINISHED"
    ].copy()

    # Skip evaluation of the first matchweek, no matches should be between now and then
    # TODO verify this works
    if len(list_of_matchweeks) <= 1:
        return df_upcoming

    # Filter once for utc_date >= now
    df_upcoming = df_upcoming[df_upcoming["utc_date"] >= pd.Timestamp.now()]

    for matchweek in list_of_matchweeks:
        df_temp = df_upcoming[(df_upcoming["utc_date"] < matchweek)]

        if not df_temp.empty:
            df_upcoming = df_temp.copy()
            break
    else:
        return df_upcoming

    # Get home and away elo
    home_elo_dict = current_season_state.home_elo
    away_elo_dict = current_season_state.away_elo

    # Get the dictionary to store points for each team
    team_to_points = current_season_state.team_to_points

    # Get the dictionary to store the form of each team
    team_to_form = current_season_state.team_to_form

    # Get simulation parameters
    model = current_season_state.model
    scaler = current_season_state.scaler

    # Initialize the columns to store the probabilities
    df_upcoming["home_win_probability"] = pd.NA
    df_upcoming["draw_probability"] = pd.NA
    df_upcoming["away_win_probability"] = pd.NA

    # For each fixture
    for index, row in df_upcoming.iterrows():
        # Get the unique values of team_to_points and sort them in descending order
        team_to_points_ranks = list(set(team_to_points.values()))
        team_to_points_values = sorted(team_to_points_ranks, reverse=True)

        # Get the points for the home and away teams
        home_points = team_to_points[row["home"]]
        away_points = team_to_points[row["away"]]

        # Update home and away league table positions based on index of team in team_to_points
        home_position = team_to_points_values.index(home_points) + 1
        away_position = team_to_points_values.index(away_points) + 1

        # Add home and away positions to the dataframe
        df_upcoming.loc[index, "home_position"] = home_position
        df_upcoming.loc[index, "away_position"] = away_position

        # Get current form of each team
        home_form = team_to_form[row["home"]]
        away_form = team_to_form[row["away"]]

        # Add form to the dataframe
        df_upcoming.loc[index, "home_form"] = sum(home_form)
        df_upcoming.loc[index, "away_form"] = sum(away_form)

        # Get the home and away teams
        home_team, away_team = row["home"], row["away"]

        # Get current Elo ratings
        home_elo = home_elo_dict[home_team]
        away_elo = away_elo_dict[away_team]

        # Simulate match and update ELO ratings
        probabilities = get_match_probabilities(
            home_elo,
            away_elo,
            home_position,
            away_position,
            row["home_manager_tenure"],
            row["away_manager_tenure"],
            row["home_form"],
            row["away_form"],
            model,
            scaler,
        )

        # Update the dataframe with the probabilities
        p_away_win, p_draw, p_home_win = probabilities
        df_upcoming.at[index, "home_win_probability"] = p_home_win
        df_upcoming.at[index, "draw_probability"] = p_draw
        df_upcoming.at[index, "away_win_probability"] = p_away_win

    return df_upcoming


def simulate_season(
    current_season_state: CurrentSeasonState,
) -> pd.DataFrame:
    # Copy the current season state to ensure immutability
    current_season_state = current_season_state.copy()

    # Get the dataframe
    df = current_season_state.df

    # Get home and away elo
    home_elo_dict = current_season_state.home_elo
    away_elo_dict = current_season_state.away_elo

    # Get the dictionary to store points for each team
    team_to_points = current_season_state.team_to_points

    # Get the dictionary to store the form of each team
    team_to_form = current_season_state.team_to_form

    # Get simulation parameters
    model = current_season_state.model
    scaler = current_season_state.scaler
    k = current_season_state.k
    half_life = current_season_state.half_life
    decay_method = current_season_state.decay_method

    # Process matches and update ELO ratings that have NOT been played
    for index, row in df[df["status"] != "FINISHED"].iterrows():
        # Get the unique values of team_to_points and sort them in descending order
        team_to_points_ranks = list(set(team_to_points.values()))
        team_to_points_values = sorted(team_to_points_ranks, reverse=True)

        # Get the points for the home and away teams
        home_points = team_to_points[row["home"]]
        away_points = team_to_points[row["away"]]

        # Update home and away league table positions based on index of team in team_to_points
        home_position = team_to_points_values.index(home_points) + 1
        away_position = team_to_points_values.index(away_points) + 1

        # Add home and away positions to the dataframe
        df.loc[index, "home_position"] = home_position
        df.loc[index, "away_position"] = away_position

        # Get current form of each team
        home_form = team_to_form[row["home"]]
        away_form = team_to_form[row["away"]]

        # Add form to the dataframe
        df.loc[index, "home_form"] = sum(home_form)
        df.loc[index, "away_form"] = sum(away_form)

        # Get the home and away teams
        home_team, away_team = row["home"], row["away"]

        # Get current Elo ratings
        home_elo = home_elo_dict[home_team]
        away_elo = away_elo_dict[away_team]

        # Simulate match and update ELO ratings
        outcome = simulate_match(
            home_elo,
            away_elo,
            home_position,
            away_position,
            row["home_manager_tenure"],
            row["away_manager_tenure"],
            row["home_form"],
            row["away_form"],
            model,
            scaler,
        )

        match outcome:
            case 3:  # Home team won
                home_elo, away_elo = update_elo_win(home_elo, away_elo, k)
                home_outcome = 3
                away_outcome = 0
            case 0:  # Away team won
                away_elo, home_elo = update_elo_win(away_elo, home_elo, k)
                home_outcome = 0
                away_outcome = 3
            case 1:  # Draw
                home_elo, away_elo = update_elo_draw(home_elo, away_elo, k)
                home_outcome = 1
                away_outcome = 1

        # Update the dataframe with the real or simulated results
        df.at[index, "home_elo"] = home_elo
        df.at[index, "away_elo"] = away_elo
        df.at[index, "home_outcome"] = home_outcome
        df.at[index, "away_outcome"] = away_outcome

        # Time-decay Elo ratings
        home_elo = apply_decay_factor(home_elo, half_life, decay_method)
        away_elo = apply_decay_factor(away_elo, half_life, decay_method)

        # Update Elo ratings in the dictionary
        home_elo_dict[home_team] = home_elo
        away_elo_dict[away_team] = away_elo

        # Update points for each team
        team_to_points[home_team] += home_outcome
        team_to_points[away_team] += away_outcome

        # Add points to the form of each team
        home_form.append(home_outcome)
        away_form.append(away_outcome)

    return df


# Define a function to simulate a season and get results
def simulate_and_get_results(
    i: int,
    current_season_state: CurrentSeasonState,
) -> pd.DataFrame:
    simulated_df = simulate_season(current_season_state)
    results = get_season_results(simulated_df)
    results["season"] = i
    return results


def upload_best_params_to_s3(params: dict, should_delete: bool = False) -> None:
    params_file = Path("best_params.json")

    # Convert enum values to integers
    for key, value in params.items():
        if isinstance(value, Enum):
            params[key] = value.value

    with open(params_file, "w") as f:
        json.dump(params, f)

    upload_to_s3(
        S3_BUCKET,
        S3_PARAMS_KEY,
        params_file,
        params_file.read_bytes(),
        "Best Params",
        should_delete,
    )


@functools.lru_cache
def download_best_params_from_s3() -> BestParams:
    params_file = Path("best_params.json").resolve()
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, S3_PARAMS_KEY, str(params_file))

    with open(params_file, "r") as f:
        params = json.load(f)

    # Delete the file
    params_file.unlink()

    return BestParams(
        k=params["k"],
        half_life=params["decay_half_life"],
        club_value_adjustment=params["club_value_adjustment_factor"],
        decay_method=DecayMethod(params["decay_method"]),
    )


@dataclass
class SimulationConfig:
    deployment_hook: str
    commands: list[str]


def download_simulation_config_from_s3() -> SimulationConfig:
    config_file = Path("inference_config.json").resolve()
    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, f"{S3_PREFIX}/inference_config.json", str(config_file))

    with open(config_file, "r") as f:
        config: dict = json.load(f)

    # Delete the file
    config_file.unlink()

    return SimulationConfig(
        deployment_hook=config.get("deployment_hook", ""),
        commands=config.get("commands", ["python", "inference.py"]),
    )


def schedule_next_simulation(
    time: pd.Timestamp,
    commands: list[str],
) -> None:
    # Schedule the next simulation using a custom lambda function
    lambda_client = boto3.client("lambda")

    # Define the payload to pass to the lambda function
    payload = {
        "timestamp": time.strftime("cron(%M %H %d %m ? %Y)"),
        "commands": commands,
        "s3_prefix": S3_PREFIX,
    }

    # Invoke the Lambda function
    response = lambda_client.invoke(
        FunctionName="schedule_pl_sim",
        InvocationType="RequestResponse",
        Payload=json.dumps(payload),
    )

    # Print response
    print("Next task schedule response:", response["Payload"].read().decode("utf-8"))


def upload_model_and_scaler_to_s3(
    model_file: str, scaler_file: str, should_delete: bool = False
) -> None:
    with open(model_file, "rb") as f:
        data = f.read()
    upload_to_s3(S3_BUCKET, S3_MODEL_KEY, model_file, data, "Model", should_delete)

    with open(scaler_file, "rb") as f:
        data = f.read()
    upload_to_s3(S3_BUCKET, S3_SCALER_KEY, scaler_file, data, "Scaler", should_delete)


def upload_to_s3(
    bucket: str, key: str, file: Path, data: bytes, name: str, should_delete: bool
) -> None:
    s3 = boto3.client("s3")
    response = s3.put_object(Bucket=bucket, Key=key, Body=data)

    if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
        print(f"{name.title()} uploaded to s3://{S3_BUCKET}/{key}")
        if should_delete:
            Path(file).unlink()
    else:
        print(f"Failed to upload {name.lower()}")


def download_model_and_scaler_from_s3(model_file: Path, scaler_file: Path) -> None:
    s3 = boto3.client("s3")

    # Download the model
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, str(model_file))
    print(f"Model downloaded to {model_file}")

    # Download the scaler
    s3.download_file(S3_BUCKET, S3_SCALER_KEY, str(scaler_file))
    print(f"Scaler downloaded to {scaler_file}")


def db_store_results(
    season: str,
    average_results_df: pd.DataFrame,
    upcoming_results_df: pd.DataFrame,
    team_positions_df: pd.DataFrame,
    team_to_points_df: pd.DataFrame,
) -> None:
    # Create a simulation uuid
    simulation_uuid = str(uuid.uuid4())

    # Save the team positions and average results to the database
    with SlowDB.connect(S3_BUCKET, S3_INFERENCE_RESULTS_DB_KEY) as conn:
        # Ensure the simulations table exists
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS simulations (
                id INTEGER PRIMARY KEY,
                uuid TEXT NOT NULL,
                date TIMESTAMP NOT NULL,
                season TEXT NOT NULL
            )
        """
        )

        # Store the simulation uuid and date in the database
        conn.execute(
            f"INSERT INTO simulations (uuid, date, season) VALUES ('{simulation_uuid}', CURRENT_TIMESTAMP, '{season}')"
        )

        # Add the uuid to the dataframe
        team_positions_df["simulation_uuid"] = simulation_uuid

        # Save the dataframe to the database
        team_positions_df.to_sql("team_positions", con=conn, if_exists="append")

        # Add the uuid to the dataframe
        average_results_df["simulation_uuid"] = simulation_uuid

        # Save the dataframe to the database
        average_results_df.to_sql("average_results", con=conn, if_exists="append")

        # Keep only the columns needed for the upcoming results
        if not upcoming_results_df.empty:
            upcoming_results_df = upcoming_results_df[
                [
                    "home",
                    "away",
                    "utc_date",
                    "home_win_probability",
                    "draw_probability",
                    "away_win_probability",
                ]
            ].copy()

            # Add the uuid to the dataframe
            upcoming_results_df["simulation_uuid"] = simulation_uuid

            # Save the dataframe to the database
            upcoming_results_df.to_sql("upcoming_results", con=conn, if_exists="append")

        # Add the uuid to the dataframe and save it to the database
        team_to_points_df["simulation_uuid"] = simulation_uuid
        team_to_points_df.to_sql("team_to_points", con=conn, if_exists="append")
