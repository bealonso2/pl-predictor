from classes.team import Team


class Table:
    """Class to encapsulate the Premier League Table.
    Fixtures need to be table-aware to be able to measure importance of fixtures."""

    # the teams in the table
    _teams: list[Team]

    def __init__(self, teams: list[Team]) -> None:
        self._teams = teams

    def __str__(self) -> str:
        """Returns a string representation of the table"""
        # the header

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
