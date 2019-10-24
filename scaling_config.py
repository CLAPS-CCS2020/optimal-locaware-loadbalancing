
import argparse


parser = argparse.ArgumentParser(description="Play with the scaled down config to change network bandwidth distribution")

parser.add_argument("--relchoice", help="csv file produced by the generate.py command, gives the list of relay, and their ConsensusWeight", default="conf/relays.choice.csv")
parser.add_argument("--config", help="path to the shadow config", default="shadow.config.xml")
parser.add_argument("--case", help="ok parameters: even or constrained guard bandwidth")


def parse_relaychoice(relchoice):
    return pandas.read_csv(relchoice)
if __name__ == "__main__":
    args = parser.parse_args()
    
    relays = parse_relaychoice(args.relchoice)
    
    
    G,M,E,D = 0, 0, 0, 0
    for Index, relay in relays.iterrows():
        if "relayexitguard" in relay['Name']:
            D += relay['ConsensusWeight']
        elif "relayguard" in relay['Name']:
            G += relay['ConsensusWeight']
        elif "relayexit" in relay['Name']:
            E += relay['ConsensusWeight']
        else:
            M += relay['ConsensusWeight']

    ## Compute bandwidth weights
    bw_weights =  BandwidthWeights()
    casename, Wgg, Wgd, Wee, Wed, Wmg, Wme, Wmd=bw_weights.recompute_bwweights(G, M, E, D, G+M+E+D)
    print("Wgg: {}".format(Wgg))

    if args.case == "even":
        # E+D == G_new*Wgg => G_new = E+D/Wgg
        G_new = (E+D/Wgg)*10000.0
        print("G_new/G: {0}".format(G_new/G))
        casename, Wgg, Wgd, Wee, Wed, Wmg, Wme,Wmd=bw_weights.recompute_bwweights(G_new, M, E, D, G+M+E+D)
        print("{},Wgg={},Wgd={},Wee={},Wed={},Wmg={},Wme={},Wmd={}".format(casename,
                                                                           Wgg,
                                                                           Wgd,
                                                                           Wee,
                                                                           Wmg,
                                                                           Wme,
                                                                           Wmd))
    
