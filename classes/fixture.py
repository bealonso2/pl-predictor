import random

from classes.team import Team


class Fixture:
    """Class to encapsulate a premier league fixture"""

    # the home and away teams
    home: Team
    away: Team

    # the match can be constructed using these three parameters
    def __init__(self, home: Team, away: Team) -> None:
        self.home = home
        self.away = away

    def play_match(self) -> None:
        """Function to play the match"""
        # Get the home and away probabilities
        home = self.home.home_prob
        away = self.away.away_prob

        # computing the draw constant
        # it should be less likely to draw if one team is really good and
        # one team is really bad
        # draw_const = 0.27 + home - away #/ (1 + home - away)
        total = home + away
        draw_const = 0.27 * abs(1 - home - away) / total

        draw = 100 * (draw_const + home + away)
        winner = random.randint(0, int(draw)) / 100.0

        # logic to assign points
        if winner < draw_const:
            self.home.draw()
            self.away.draw()
        elif winner > draw_const and winner < draw_const + home:
            self.home.win()
            self.away.lose()
        else:
            self.home.lose()
            self.away.win()
