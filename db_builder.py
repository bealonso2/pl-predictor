from typing import Callable

import bs4
import pandas as pd
import requests

import SlowDB
from simulation_utils import (
    S3_BUCKET,
    S3_INFERENCE_DATA_DB_KEY,
    build_data_by_year,
    get_club_value_at_season,
    get_team_crests_from_season,
)


# Define update or insert function
def update_or_insert_data(
    table: str,
    columns: list[str],
    new_data_df: pd.DataFrame,
    database_function: Callable[[pd.DataFrame], pd.DataFrame] = lambda x: x,
) -> pd.DataFrame:
    # Call the database function on the new data
    new_data_df = database_function(new_data_df)

    with SlowDB.connect(S3_BUCKET, S3_INFERENCE_DATA_DB_KEY) as conn:
        # Read the data from the database
        try:
            existing_df = database_function(pd.read_sql(f"SELECT * FROM {table}", conn))
        except:
            # If the table does not exist, write the new data to the database
            df = new_data_df
        else:
            # Set the index for both DataFrames
            new_data_df = new_data_df.set_index(columns)
            existing_df = existing_df.set_index(columns)

            # Find overlapping rows (by index)
            overlapping_rows = new_data_df.index.intersection(existing_df.index)

            # Drop the overlapping rows from the existing DataFrame
            existing_df.drop(overlapping_rows, axis="index", inplace=True)

            # Reset the index for both DataFrames
            new_data_df = new_data_df.reset_index()
            existing_df = existing_df.reset_index()

            # Append the new rows to the existing DataFrame
            if "season" in existing_df.columns or "season" in new_data_df.columns:
                sort_columns = ["season"]
            else:
                sort_columns = columns
            df = (
                pd.concat([existing_df, new_data_df], ignore_index=True)
                .sort_values(sort_columns)
                .reset_index(drop=True)
            )

        # Replace the data in the database
        rows_updated = df.to_sql(table, conn, if_exists="replace", index=False)

        # Print the number of rows updated
        print(f"Updated {rows_updated} rows in the {table} table")

        conn.commit()

        return df


# Define a function to only replace the data
def replace_data(table: str, new_data_df: pd.DataFrame) -> pd.DataFrame:
    with SlowDB.connect(S3_BUCKET, S3_INFERENCE_DATA_DB_KEY) as conn:
        # Replace the data in the database
        rows_updated = new_data_df.to_sql(table, conn, if_exists="replace", index=False)

        # Print the number of rows updated
        print(f"Updated {rows_updated} rows in the {table} table")

        conn.commit()
    return new_data_df


def update_results(seasons: list[int]) -> pd.DataFrame:
    """
    Update the results information in the database
    """
    results_df = pd.concat([build_data_by_year(year, False) for year in seasons])

    def convert_utc_date_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
        df["utc_date"] = pd.to_datetime(df["utc_date"])
        return df

    # Update the database
    results_df = update_or_insert_data(
        "football_data_season_results",
        ["season", "home", "away"],
        results_df,
        convert_utc_date_to_datetime,
    )

    return results_df


def update_results_and_club_values(
    seasons: list[int], update_club_values: bool
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Update the club values in the database after updating the results
    """
    # Update the managers information
    results_df = update_results(seasons)

    if not update_club_values:
        print("Club values not updated")
        return results_df, None

    # Get unique club names minus FC
    clubs = results_df["home"].unique()

    # Get club value for each season
    club_values = {}

    for club in clubs:
        club_values[club] = [
            {"season": season, "value": get_club_value_at_season(club, season)}
            for season in seasons
        ]

    # Flatten the dictionary
    flattened_data = []

    for club, values in club_values.items():
        for value in values:
            flattened_data.append(
                {"club": club, "season": value["season"], "value": value["value"]}
            )

    # Convert the list of dictionaries to a DataFrame
    transfers_df = pd.DataFrame(flattened_data)

    transfers_df = update_or_insert_data(
        "transfermarkt_club_values", ["club", "season"], transfers_df
    )

    return results_df, transfers_df


def update_manager_tenure():
    # Get the data from the website
    url = "https://en.wikipedia.org/wiki/List_of_Premier_League_managers"
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text, "html.parser")

    # Get the table with classes wikitable sortable plainrowheaders jquery-tablesorter
    table = soup.find("table", {"class": "wikitable sortable plainrowheaders"})

    # Get the rows from the table
    rows = table.find_all("tr")
    rows = rows[1:]

    # Get the data from the rows
    data = []

    for row in rows:
        cells = row.find_all(["th", "td"])

        # Get rid of all child sup elements
        for cell in cells:
            for sup in cell.find_all("sup"):
                sup.decompose()

        cells = [cell.text.strip() for cell in cells if cell.text.strip()]
        data.append(cells)

    # Convert the data to a DataFrame
    managers_df = pd.DataFrame(
        data, columns=["manager", "club", "start", "end", "duration_days", "years"]
    )

    # Add columns for keys
    managers_df["incumbent"] = managers_df["manager"].apply(
        lambda x: True if str(x).endswith(" †") else False
    )
    managers_df["caretaker"] = managers_df["manager"].apply(
        lambda x: True if str(x).endswith(" ‡") else False
    )
    managers_df["incumbent_not_in_league"] = managers_df["manager"].apply(
        lambda x: True if str(x).endswith(" §") else False
    )

    # Remove the symbols from the manager names
    managers_df["manager"] = managers_df["manager"].str.replace(" †", "")
    managers_df["manager"] = managers_df["manager"].str.replace(" ‡", "")
    managers_df["manager"] = managers_df["manager"].str.replace(" §", "")

    # Function to convert all start and end dates to datetime
    managers_df["start"] = pd.to_datetime(managers_df["start"], errors="coerce")
    managers_df["end"] = pd.to_datetime(managers_df["end"], errors="coerce")

    # Replace the table in the database
    managers_df = replace_data("premier_league_managers", managers_df)


def update_team_crests():
    # Connect to the database
    with SlowDB.connect(S3_BUCKET, S3_INFERENCE_DATA_DB_KEY) as conn:
        # Read the unique clubs from the database
        clubs_df = pd.read_sql(
            "SELECT season, home FROM football_data_season_results", conn
        )

        # If the team_to_crests table does not exist, create it
        try:
            crests_df = pd.read_sql("SELECT * FROM team_to_crests", conn)
        except:
            conn.execute(
                "CREATE TABLE team_to_crests (team TEXT PRIMARY KEY, crest TEXT)"
            )
            crests_df = pd.DataFrame(columns=["team", "crest"])

        # Get the unique clubs
        clubs = clubs_df["home"].unique()

        # Get the unique clubs that are not in the team_to_crests table
        crests_to_update = [
            club for club in clubs if club not in crests_df["team"].values
        ]

        if not crests_to_update:
            print("All crests are up to date")
            conn.commit()
            crests_df
        else:
            # Figure out the latest season for the clubs in crests to update
            clubs_agg = (
                clubs_df[clubs_df["home"].isin(crests_to_update)]
                .groupby("home")
                .agg({"season": "max"})
                .reset_index()
            )

            # Get unique seasons from the clubs_agg DataFrame
            seasons = clubs_agg["season"].unique()

            # Get the latest data for the clubs in crests to update
            df = pd.concat([get_team_crests_from_season(season) for season in seasons])

            # Set the index to the team and drop duplicates
            df = df.set_index("team").drop_duplicates()

            # Update the crests_df DataFrame
            crests_df = pd.concat(
                [crests_df.set_index("team", drop=True), df]
            ).drop_duplicates()

            # Replace the data in the database
            crests_df.to_sql("team_to_crests", conn, if_exists="replace")

            # Commit the changes
            conn.commit()
