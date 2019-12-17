import pandas
import argparse
from bandwidth_weights import *
import json

parser = argparse.ArgumentParser(description="Play with the scaled down config to change network bandwidth distribution")

parser.add_argument("--relchoice", help="csv file produced by the generate.py command, gives the list of relay, and their ConsensusWeight", default="conf/relays.choice.csv")
parser.add_argument("--v3bw", help="path to the v3bw file to edit with the new bandwidths")
parser.add_argument("--config", help="path to the shadow config")
parser.add_argument("--case", help="ok parameters: even or constrained guard bandwidth")
parser.add_argument("--shadow_relay_dump", help="If we do a CLAPS simulation, we need to edit the bandwidths in that file as well")

def parse_relaychoice(relchoice):
    return pandas.read_csv(relchoice)

def edit_shadow_config(relays, config_path):
    with open(config_path) as f:
        with open("shadow.config.2.xml", "w") as altshadow:
            for line in f:
                if "relayguard" in line:
                    elems = line.split()
                    name = elems[1].split('"')[1]
                    elems[5] = 'bandwidthdown="{}"'.format(relays[name])
                    elems[6] = 'bandwidthup="{}"'.format(relays[name])
                    newline = " ".join(elems)
                    print(newline, file=altshadow)
                else:
                    print(line[:-1], file=altshadow)


def edit_v3bw_config(relays, v3bw_path):
    with open(v3bw_path) as f:
        with open("v3bw.2", "w") as altv3bw:
            for line in f:
                if "relayguard" in line:
                    elems = line.split()
                    name = elems[2].split("=")[1]
                    elems[1] = "bw={}".format(relays[name])
                    newline = " ".join(elems)
                    print(newline, file=altv3bw)
                else:
                    print(line[:-1], file=altv3bw)

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
    elif args.case == "scarce":
        G_new = M
    print("G_new/G: {0}".format(G_new/G))
    casename, Wgg, Wgd, Wee, Wed, Wmg, Wme,Wmd=bw_weights.recompute_bwweights(G_new, M, E, D, G_new+M+E+D)
    print("{},Wgg={},Wgd={},Wee={},Wed={},Wmg={},Wme={},Wmd={}".format(casename,
        Wgg, Wgd, Wee, Wed, Wmg, Wme, Wmd))
    if args.case == "even":
        print("this Network should be even G = M = E+D: G:{}, "
              "M:{}, E+D:{}".format(G_new*Wgg/10000.0,
                  M+G_new*Wmg/10000.0, E+D))
    elif args.case == "scarce":
        print("This network has scarce G bandwidth, and scarce M bandwidth -- We use the guard+exit relays to load-balance to: entry:{}, middle:{}, exit:{}".format(
            (G_new*Wgg+D*Wgd)/10000.0, M+D*Wmd/10000.0, E+D*Wed/10000.0))
    reducing_bw = G_new/G
    print("Removing {}% of guard relays".format(100-reducing_bw*100))
    nrelays = {}
    for Index, relay in relays.iterrows():
        if "relayguard" in relay['Name']:
            nrelays[relay['Name']] = int(round(relay['ConsensusWeight'] * reducing_bw))
            relays.at[Index, 'ConsensusWeight'] = nrelays[relay['Name']]
    relays.to_csv("relay.choices.2.csv")
    if args.config:
        print("Editing the shadow config, and saving to shadow.config.2.xml")
        edit_shadow_config(nrelays, args.config)
    if args.v3bw:
        print("Editing the v3bw file and saving to v3bw.2")
        edit_v3bw_config(nrelays, args.v3bw)
    if args.shadow_relay_dump:
        with open(args.shadow_relay_dump) as f:
            relay_dump = json.load(f)
            for relay in nrelays:
                relay_dump[relay][6] = nrelays[relay]
        with open("shadow_relay_dump.json", "w") as f:
            json.dump(relay_dump, f)
    

