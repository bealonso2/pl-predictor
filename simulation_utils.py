import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import warnings
import numpy as np


def update_elo_win(winner_elo, loser_elo, k=40):
    expected_win = 1.0 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    change = k * (1 - expected_win)
    return winner_elo + change, loser_elo - change


def update_elo_draw(home_elo, away_elo, k=40):
    expected_home_win = 1.0 / (1 + 10 ** ((away_elo - home_elo) / 400))
    change = k * (0.5 - expected_home_win)
    return home_elo + change, away_elo - change


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


def simulate_season(df: pd.DataFrame, elo, model, scalar) -> pd.DataFrame:
    df_2023 = df.copy()

    # Divide elo by 2 to get home and away elo
    home_elo = {team: elo / 2 for team, elo in elo.items()}
    away_elo = {team: elo / 2 for team, elo in elo.items()}

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
