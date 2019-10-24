
import argparse


parser = argparse.ArgumentParser(description="Play with the scaled down config to change network bandwidth distribution")

parser.add_argument("--relchoice", help="csv file produced by the generate.py command, gives the list of relay, and their ConsensusWeight", default="conf/relays.choice.csv")
parser.add_argument("--config", help="path to the shadow config", default="shadow.config.xml")


if __name__ == "__main__":
    args = parser.parse_args()
    
    relays = parse_
