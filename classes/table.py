import math
from classes.team import Team


class Table:
    """Class to encapsulate the Premier League Table.
    Fixtures need to be table-aware to be able to measure importance of fixtures."""

    # the teams in the table
    _teams: list[Team]

    def __init__(self, teams: list[Team]) -> None:
        self._teams = teams

    def _sort_teams(self):
        # Sort the teams by points, then by wins
        # TODO - update with goal difference when available
        self._teams.sort(key=lambda x: (x.points, x.wins), reverse=True)

    def _update_table(self) -> None:
        """Updates the table with the current position of the teams"""

        # Sort the teams
        self._sort_teams()

        for team in self._teams:
            team.position = self.get_position(team)

    def get_position(self, team: Team) -> int:
        """Returns the position of a team in the table"""
        # return the position of the team in the table
        return self._teams.index(team) + 1

    @property
    def total_teams(self) -> int:
        """Returns the total number of teams in the table"""
        return len(self._teams)

    def finalize_table(self) -> None:
        for team in self._teams:
            # Set the team's final position
            team.season_result()

    @staticmethod
    def calculate_modifiers(
        home: Team, away: Team, table: "Table"
    ) -> tuple[float, float]:
        """Importance of a match calculation. This should be based on how much
        the result of the match will have an impact on the final table.

        Right now it will just be a difference of position. The closer the teams
        are in the table, the more important the match is."""
        # TODO - update this to take into account potential for position in the final table
        # Initialize the variables
        home_modifier = 1.0
        away_modifier = 1.0

        home_position = home.position
        away_position = away.position
        position_difference = home_position - away_position

        # The maximum contribution to the modifier is 0.03 which should be given when the
        # position difference is -1, otherwise it is a square root function.
        position_difference_modifier = (
            -math.sqrt(abs(position_difference) / (19 / 0.09)) + 0.03
        )
        if position_difference < 0:
            # Home team should get the maximum boost (with fans)
            home_modifier += position_difference_modifier
            # The away team mey under perform
            away_modifier -= position_difference_modifier / 2
        elif position_difference > 0:
            # Away team should get a slight boost
            away_modifier += position_difference_modifier / 2
            # This is supposed to be the case where the home team is better than the away team but may not rise to the occasion
            home_modifier -= position_difference_modifier / 2

        # The later in the season, the more important the match.
        home_mp_modifier = math.sqrt(home.matches_played / (38 / 0.9)) + 0.1
        away_mp_modifier = math.sqrt(away.matches_played / (38 / 0.9)) + 0.1

        # Final modifiers
        home_modifier *= home_mp_modifier
        away_modifier *= away_mp_modifier

        return home_modifier, away_modifier
