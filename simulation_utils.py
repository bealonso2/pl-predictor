import difflib
from enum import Enum, auto
import functools
import re
import sqlite3
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


def find_manager(managers_df: pd.DataFrame, team: str, date: pd.Timestamp) -> str:
    # Try to find the closest team name match
    clubs = managers_df["club"].unique()
    team = difflib.get_close_matches(team, clubs)[0]
    managers_df = managers_df[managers_df["club"] == team]

    # Get the managers for the team at the given date
    return managers_df[
        (managers_df["start"] <= date.to_datetime64())
        & (managers_df["end"] >= date.to_datetime64())
    ]["manager"].iloc[0]


def db_add_managers_to_df(df: pd.DataFrame) -> pd.DataFrame:
    with sqlite3.connect("data.db") as conn:
        managers_df = pd.read_sql("SELECT * FROM premier_league_managers", conn)

    # Convert start and end dates to datetime
    managers_df["start"] = pd.to_datetime(
        managers_df["start"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    ).fillna(pd.Timestamp.now())

    managers_df["end"] = pd.to_datetime(
        managers_df["end"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    ).fillna(pd.Timestamp.now())

    # Add one day to the end date
    managers_df["end"] = managers_df["end"] + pd.Timedelta(days=1)

    # Convert utc_date to datetime
    df["utc_date"] = pd.to_datetime(
        df["utc_date"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    ).fillna(pd.Timestamp.now())

    # For each fixture, find the manager of the home team
    df["home_manager"] = df.apply(
        lambda x: find_manager(managers_df, x["home"], x["utc_date"]), axis=1
    )

    # For each fixture, find the manager of the away team
    df["away_manager"] = df.apply(
        lambda x: find_manager(managers_df, x["away"], x["utc_date"]), axis=1
    )

    # Add a column to indicate the index of manager change
    df["home_manager_count"] = 0
    df["away_manager_count"] = 0

    # Get the unique managers for each team and find the index of the manager change
    home_managers = (
        df[["home", "home_manager", "utc_date"]]
        .drop_duplicates()
        .rename(columns={"home": "team", "home_manager": "manager"})
    )
    away_managers = (
        df[["away", "away_manager", "utc_date"]]
        .drop_duplicates()
        .rename(columns={"away": "team", "away_manager": "manager"})
    )
    teams_to_managers = pd.concat([home_managers, away_managers]).drop_duplicates()

    for team in teams_to_managers["team"].unique():
        managers: pd.DataFrame = teams_to_managers[teams_to_managers["team"] == team][
            ["manager", "utc_date"]
        ]

        # Sort by date
        managers = managers.sort_values(by="utc_date")
        managers = managers["manager"].unique()

        # Skip if there is only one manager by dropping the first manager
        managers = managers[1:]

        for i, manager in enumerate(managers, 1):
            df.loc[
                (df["home"] == team) & (df["home_manager"] == manager),
                "home_manager_count",
            ] = i
            df.loc[
                (df["away"] == team) & (df["away_manager"] == manager),
                "away_manager_count",
            ] = i

    return df


def get_league_table_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns to the dataframe to indicate the league table position of
    the home and away teams at the time of the fixture, for the given portion of
    the dataframe.
    """
    df = df.copy()

    for i, row in df.iterrows():
        # Get the portion of the dataframe up to the date of the fixture
        portion: pd.DataFrame = df[df["utc_date"] <= row["utc_date"]]

        home_results = portion.groupby("home").agg({"home_outcome": "sum"})
        away_results = portion.groupby("away").agg({"away_outcome": "sum"})
        results = home_results.join(away_results, how="outer").fillna(0)
        results["total_outcome"] = results["home_outcome"] + results["away_outcome"]

        # Sort by total outcome
        results = results.sort_values(by="total_outcome", ascending=False)

        # Get the league table position of the home team
        df.loc[i, "home_position"] = int(results.index.get_loc(row["home"])) + 1

        # Get the league table position of the away team
        df.loc[i, "away_position"] = int(results.index.get_loc(row["away"])) + 1

    # Convert the columns to integers
    df["home_position"] = df["home_position"].astype(int)
    df["away_position"] = df["away_position"].astype(int)
    return df


def db_get_data_by_year(year: int) -> pd.DataFrame:
    with sqlite3.connect("data.db") as conn:
        df = pd.read_sql(
            f"SELECT * FROM football_data_season_results WHERE season = {year}", conn
        )

    # Add managers to the dataframe
    df = db_add_managers_to_df(df)

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
    with sqlite3.connect("data.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT value FROM transfermarkt_club_values WHERE club = '{club}' AND season = {season}"
        )
        club_value = cursor.fetchone()[0]
    return club_value


@functools.lru_cache
def db_get_club_value(club: str) -> float:
    with sqlite3.connect("data.db") as conn:
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

    # Process matches and update ELO ratings
    for index, row in df.iterrows():
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


def simulate_match(
    home_elo: float,
    away_elo: float,
    home_position: int,
    away_position: int,
    model: RandomForestClassifier,
    scaler: StandardScaler,
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = scaler.transform([[home_elo, away_elo, home_position, away_position]])
    probabilities = model.predict_proba(x)
    return np.random.choice([0, 1, 3], p=probabilities[0])


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


def simulate_season(
    df: pd.DataFrame,
    elo: dict[str, float],
    model: RandomForestClassifier,
    scalar: StandardScaler,
    k: int,
    half_life: int,
    decay_method: DecayMethod,
) -> pd.DataFrame:
    df_2023 = df.copy()

    # Get home and away elo
    home_elo_dict = {team: elo for team, elo in elo.items()}
    away_elo_dict = {team: elo for team, elo in elo.items()}

    # Initialize a dictionary to store points for each team
    team_to_points = {team: 0 for team in df_2023["home"].unique()}

    # Can run matches in a match week concurrently in the future
    for index, row in df_2023.iterrows():
        # Sort team_to_points by value descending
        team_to_points = dict(
            sorted(team_to_points.items(), key=lambda item: item[1], reverse=True)
        )

        # Update home and away league table positions
        home_position = team_to_points[row["home"]]
        away_position = team_to_points[row["away"]]

        # Add home and away positions to the dataframe
        df_2023.loc[index, "home_position"] = home_position
        df_2023.loc[index, "away_position"] = away_position

        # Get the home and away teams
        home_team, away_team = row["home"], row["away"]

        # Get current Elo ratings
        home_elo = home_elo_dict[home_team]
        away_elo = away_elo_dict[away_team]

        # Simulate match and update ELO ratings
        outcome = simulate_match(
            home_elo, away_elo, home_position, away_position, model, scalar
        )

        match outcome:
            case 3:  # Home team won
                home_elo, away_elo = update_elo_win(home_elo, away_elo, k)
                df_2023.at[index, "home_outcome"] = 3
                df_2023.at[index, "away_outcome"] = 0
            case 0:  # Away team won
                away_elo, home_elo = update_elo_win(away_elo, home_elo, k)
                df_2023.at[index, "home_outcome"] = 0
                df_2023.at[index, "away_outcome"] = 3
            case 1:  # Draw
                home_elo, away_elo = update_elo_draw(home_elo, away_elo, k)
                df_2023.at[index, "home_outcome"] = 1
                df_2023.at[index, "away_outcome"] = 1

        # Time-decay Elo ratings
        home_elo = apply_decay_factor(home_elo, half_life, decay_method)
        away_elo = apply_decay_factor(away_elo, half_life, decay_method)

        # Update Elo ratings in the dictionary
        home_elo_dict[home_team] = home_elo
        away_elo_dict[away_team] = away_elo

        # Update actual results
        if row["home_score"] > row["away_score"]:
            home_outcome = 3
            away_outcome = 0
        elif row["away_score"] > row["home_score"]:
            home_outcome = 0
            away_outcome = 3
        else:
            home_outcome = 1
            away_outcome = 1

        # Update points for each team
        team_to_points[home_team] += home_outcome
        team_to_points[away_team] += away_outcome

        # Update the dataframe with the actual results
        df_2023.at[index, "actual_home_outcome"] = home_outcome
        df_2023.at[index, "actual_away_outcome"] = away_outcome
        df_2023.at[index, "home_elo"] = home_elo_dict[home_team]
        df_2023.at[index, "away_elo"] = away_elo_dict[away_team]

    return df_2023


# Define a function to simulate a season and get results
def simulate_and_get_results(i, df, elo, model, scaler, k, half_life, decay_method):
    simulated_df = simulate_season(df, elo, model, scaler, k, half_life, decay_method)
    results = get_season_results(simulated_df)
    results["season"] = i
    return results
