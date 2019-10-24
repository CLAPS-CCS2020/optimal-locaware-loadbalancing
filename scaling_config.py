import pandas
import argparse
from bandwidth_weights import *

parser = argparse.ArgumentParser(description="Play with the scaled down config to change network bandwidth distribution")

parser.add_argument("--relchoice", help="csv file produced by the generate.py command, gives the list of relay, and their ConsensusWeight", default="conf/relays.choice.csv")
parser.add_argument("--config", help="path to the shadow config", default="shadow.config.xml")
parser.add_argument("--case", help="ok parameters: even or constrained guard bandwidth")


def parse_relaychoice(relchoice):
    return pandas.read_csv(relchoice)

def edit_shadow_config(relays, config_path):
    with open(config_path) as f:
        with open("shadow.config.2.xml", "w") as altshadow:
            for line in f:
                if "relayguard" in line:
                    elems = line.split(" ")[-1]
                    name = elems[1].split('"')[0]
                    elems[5] = "bandwidthdown={}".format(relays[name])
                    elems[6] = "bandwidthdup={}".format(relays[name])
                    newline = " ".join(elems)
                    print(newline, file=altshadow)
                else:
                    print(line[:-1], file=altshadow)


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
    print("{},Wgg={},Wgd={},Wee={},Wed={},Wmg={},Wme={},Wmd={}".format(casename,
        Wgg, Wgd, Wee, Wed, Wmg, Wme, Wmd))

    if args.case == "even":
        # E+D == G_new*Wgg => G_new = E+D/Wgg
        G_new = 2*E+2*D-M
        print("G_new/G: {0}".format(G_new/G))
        casename, Wgg, Wgd, Wee, Wed, Wmg, Wme,Wmd=bw_weights.recompute_bwweights(G_new, M, E, D, G_new+M+E+D)
        print("{},Wgg={},Wgd={},Wee={},Wed={},Wmg={},Wme={},Wmd={}".format(casename,
            Wgg, Wgd, Wee, Wed, Wmg, Wme, Wmd))

        print("this Network should be even G = M = E+D: G:{}, "
              "M:{}, E+D:{}".format(G_new*Wgg/10000.0,
                  M+G_new*Wmg/10000.0, E+D))
        reducing_bw = G_new/G
        print("Removing {}% of guard relays".format(reducing_bw))
        nrelays = {}
        for Index, relay in relays.iterrows():
            if "relayguard" in relay['Name']:
                nrelays[relay['Name']] = int(round(relay['ConsensusWeight'] * reducing_bw))
        print("Editing the shadow config, and saving to shadow.config.2.xml")
        edit_shadow_config(nrelays, args.config)
        

