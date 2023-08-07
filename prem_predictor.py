from classes.simulation import PremierLeaguePredictor


def main():
    # create the simulation
    simulation = PremierLeaguePredictor()

    # simulate the seasons
    simulation.simulate_all_seasons()

    # output the results
    print(simulation)
    simulation.to_csv()
    simulation.to_json()


if __name__ == "__main__":
    main()
