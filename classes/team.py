from typing import Union

# adding functionality for 'form' by using fixtures. Right now fixtures won't change
FORM = 0.005


class Team:
    """Class to encapsulate a premier league team"""

    # name field consistent with fixtures
    name = ""

    # optional display name to be consistent with PL website
    _display_name = ""

    # the probability of a win at home and away from home
    home_prob = 0.0
    away_prob = 0.0
    _starting_home_prob = 0.0
    _starting_away_prob = 0.0

    # the teams stats
    points: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    matches_played: int = 0

    # end of season stats
    finish: int = 0
    total_points: int = 0
    total_wins: int = 0
    total_draws: int = 0
    total_losses: int = 0

    # the team can be constructed using these three parameters
    def __init__(
        self,
        name: str,
        display_name: Union[str, None],
        home_prob: float,
        away_prob: float,
    ) -> None:
        self.name = name
        self._display_name = display_name
        self.home_prob = home_prob
        self.away_prob = away_prob
        self._starting_home_prob = home_prob
        self._starting_away_prob = away_prob

    # returns the display name or name if no display name
    @property
    def display_name(self):
        return self._display_name if self._display_name else self.name

    def match_result(func: callable) -> None:
        """Function to handle the result of a match

        Args:
            func (callable): The function to call to handle the result
        """

        def func_wrapper(self: "Team"):
            func(self)
            self.matches_played += 1

        return func_wrapper

    @match_result
    def win(self):
        """Function to handle wins"""
        self.points += 3
        self.wins += 1
        if self.home_prob < 1:
            self.home_prob += FORM
        if self.away_prob < 1:
            self.away_prob += FORM

    @match_result
    def lose(self):
        """Function to handle losses"""
        self.losses += 1
        self.matches_played += 1
        if self.home_prob > 5 * FORM:
            self.home_prob -= FORM
        if self.away_prob > 5 * FORM:
            self.away_prob -= FORM

    @match_result
    def draw(self):
        """Function to handle draws"""
        self.points += 1
        self.draws += 1
        self.matches_played += 1

    # variable to keep track of table position
    def result(self, finish: int) -> None:
        """Function to reset the teams stats and add previous season stats to
        cumulative stats

        Args:
            finish (int): The position the team finished in the table
        """
        # Set the fields that are cumulative
        self.finish += finish
        self.total_points += self.points
        self.total_wins += self.wins
        self.total_draws += self.draws
        self.total_losses += self.losses

        # Reset the team stats
        self.points = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0

        # Reset the team with the starting probabilities
        self.home_prob = self._starting_home_prob
        self.away_prob = self._starting_away_prob
