import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import warnings
import numpy as np


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


def get_club_value(club: str) -> float:
    response = requests.get(
        f"https://transfermarkt-api.fly.dev/clubs/search/{club.replace(' FC', '')}",
        headers={"accept": "application/json"},
    )
    club_value: str = response.json()["results"][0]["marketValue"]
    return float(
        club_value.replace("€", "")
        .replace("m", "0000")
        .replace("bn", "0000000")
        .replace(".", "")
    )


def build_elo_df_from_dict(
    elo_dict: dict[str, float], adjustment_factor: float = 100
) -> pd.DataFrame:
    # Build an elo dataframe
    elo_df = pd.DataFrame(elo_dict.items(), columns=["Team", "Elo"]).set_index("Team")

    # Get club values
    elo_df["Club Value"] = elo_df.index.map(get_club_value)

    # Normalize the club value column
    elo_df["Normalized Club Value"] = (
        elo_df["Club Value"] - elo_df["Club Value"].min()
    ) / (elo_df["Club Value"].max() - elo_df["Club Value"].min())

    # Adjust the ELO ratings based on the normalized club values
    elo_df["Adjusted Elo"] = (
        elo_df["Elo"] + adjustment_factor * elo_df["Normalized Club Value"]
    )

    return elo_df


def update_elo_win(winner_elo, loser_elo, k=40):
    expected_win = 1.0 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    change = k * (1 - expected_win)
    return winner_elo + change, loser_elo - change


def update_elo_draw(home_elo, away_elo, k=40):
    expected_home_win = 1.0 / (1 + 10 ** ((away_elo - home_elo) / 400))
    change = k * (0.5 - expected_home_win)
    return home_elo + change, away_elo - change


def process_fixture_results(
    df: pd.DataFrame, k: int = 40
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Initialize elo ratings for each team
    elo = {team: 1500 for team in df["Home"].unique()}

    # Adjust Elo based on club value
    elo = build_elo_df_from_dict(elo, 300)["Adjusted Elo"].to_dict()

    # Process matches and update ELO ratings
    for index, row in df.iterrows():
        home_team, away_team = row["Home"], row["Away"]

        if row["Home Score"] > row["Away Score"]:  # Home team won
            elo[home_team], elo[away_team] = update_elo_win(
                elo[home_team], elo[away_team], k
            )
        elif row["Away Score"] > row["Home Score"]:  # Away team won
            elo[away_team], elo[home_team] = update_elo_win(
                elo[away_team], elo[home_team], k
            )
        else:  # Draw
            elo[home_team], elo[away_team] = update_elo_draw(
                elo[home_team], elo[away_team], k
            )

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
    df: pd.DataFrame, elo: dict[str, float], model, scalar
) -> pd.DataFrame:
    df_2023 = df.copy()

    # Divide elo by 2 to get home and away elo
    home_elo = {team: elo for team, elo in elo.items()}
    away_elo = {team: elo for team, elo in elo.items()}

    # Can run matches in a match week concurrently in the future
    for index, row in df_2023.iterrows():
        home_team, away_team = row["Home"], row["Away"]

        # Simulate match and update ELO ratings
        outcome = simulate_match(
            home_elo[home_team], away_elo[away_team], model, scalar
        )

        match outcome:
            case 3:  # Home team won
                home_elo[home_team], away_elo[away_team] = update_elo_win(
                    home_elo[home_team], away_elo[away_team]
                )
                df_2023.at[index, "Home Outcome"] = 3
                df_2023.at[index, "Away Outcome"] = 0
            case 0:  # Away team won
                away_elo[away_team], home_elo[home_team] = update_elo_win(
                    away_elo[away_team], home_elo[home_team]
                )
                df_2023.at[index, "Away Outcome"] = 3
                df_2023.at[index, "Home Outcome"] = 0
            case 1:  # Draw
                home_elo[home_team], away_elo[away_team] = update_elo_draw(
                    home_elo[home_team], away_elo[away_team]
                )
                df_2023.at[index, "Home Outcome"] = 1
                df_2023.at[index, "Away Outcome"] = 1

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

        df_2023.at[index, "Home Elo"] = home_elo[home_team]
        df_2023.at[index, "Away Elo"] = away_elo[away_team]

    return df_2023


# Define a function to simulate a season and get results
def simulate_and_get_results(i, df_2023, elo, model, scaler):
    simulated_df = simulate_season(df_2023, elo, model, scaler)
    results = get_season_results(simulated_df)
    results["Season"] = i
    return results
