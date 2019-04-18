
import os, sys
import argparse
import pickle
from slim_desc import ClusterRouter
from tor_users_per_country import get_network_state
import requests
import time as dtime
from stem import Flag
"""

This file takes an network state as an input, and cluster guard relays
by the n announced prefix from a same AS.0

"""

parser = argparse.ArgumentParser(description="get a Tor network state information and cluster guard relays")
parser.add_argument("--ns_file", help="slim_desc.NetworkState path to pickle file", required=True)
parser.add_argument("--cluster_type", type=str, help="Support either 'AS' or 'prefix'; i.e., --cluster_type AS")
parser.add_argument("--out", help="out dir to store the clustered guards", required=True)

RIPE_BGP_INFO = "https://stat.ripe.net/data/bgp-state/data.json?resource="
PREFIX_OVERVIEW = "https://stat.ripe.net/data/prefix-overview/data.json?resource="

def main(args):
    clusters = {} # key: string of prefix IP, val: slim_desc.ClusterRouter
    network_state = get_network_state(args.ns_file)
    descriptors = network_state.descriptors
    guardsfp = [relay for relay in network_state.cons_rel_stats if Flag.GUARD in network_state.cons_rel_stats[relay].flags and
            not Flag.EXIT in network_state.cons_rel_stats[relay].flags and
            Flag.VALID in network_state.cons_rel_stats[relay].flags and
            Flag.RUNNING in network_state.cons_rel_stats[relay].flags]
    #building clusters
    if args.cluster_type == "identity":
        for guard in guardsfp:
            clusters[guard] = ClusterRouter(None, descriptors[guard].address)
            clusters[guard].addRouter(network_state.cons_rel_stats[guard])

    elif args.cluster_type == "prefix":
        for guard in guardsfp:
            ipv4_address = descriptors[guard].address
            r = requests.get(RIPE_BGP_INFO+ipv4_address+"&timestamp="+str(network_state.cons_valid_after))
            if r.status_code > 200:
                for _ in range(0,2):
                    print("Status code of previous request: {} - Waiting 5 seconds and retry".format(r.status_code))
                    dtime.sleep(5)
                    if r.status_code < 300:
                        break
                if r.status_code > 200:
                    print("Something went wrong with this req: {}".format(r.json()))
                    continue
            bgpinfo = r.json()
            prefix = bgpinfo['data']['resource']
            if prefix in clusters:
                clusters[prefix].addRouter(network_state.cons_rel_stats[guard])
            else:
                if len(bgpinfo['data']['bgp_state']) == 0:
                    asn = "-1"
                else:
                    asn = int(bgpinfo['data']['bgp_state'][0]['path'][-1])
                clusters[prefix] = ClusterRouter(asn, prefix)
                clusters[prefix].addRouter(network_state.cons_rel_stats[guard])

    elif args.cluster_type == "AS":
        for guard in guardsfp:
            ipv4_address = descriptors[guard].address
            r = requests.get(PREFIX_OVERVIEW+ipv4_address+"&timestamp="+str(network_state.cons_valid_after))
            if r.status_code > 200:
                for _ in range(0,2):
                    print("Status code of previous request: {} - Waiting 5 seconds and retry".format(r.status_code))
                    dtime.sleep(5)
                    if r.status_code < 300:
                        break
                if r.status_code > 200:
                    print("Something went wrong with this req: {}".format(r.json()))
                    continue
            prefixinfo = r.json()
            asns = prefixinfo['data']['asns']
            if len(asns) > 1:
                print("shit, it seems that this IP originates from several ASes {}, what do we do?".format(asns))
                asn = asns[0]['asn']
            elif len(asns) == 0:
                print("Shit, it seems that we don't have information regarding this IP {}.. What do we do?".format(ipv4_address))
                asn = -1
            else:
                asn = asns[0]['asn']
            if asn in clusters:
                clusters[asn].addRouter(network_state.cons_rel_stats[guard])
            else:
                clusters[asn] = ClusterRouter(asn, None)
                clusters[asn].addRouter(network_state.cons_rel_stats[guard])


    print("We had {} guards before and now {} clusters".format(len(guardsfp), len(list(clusters.keys()))))
    outpath = os.path.join(args.out, "clusters"+args.cluster_type+"_"+str(network_state.cons_valid_after))
    with open(outpath, 'wb') as f:
        pickle.dump(clusters, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    args = parser.parse_args()
    if args.cluster_type == "AS" or args.cluster_type == "prefix" or args.cluster_type == "identity":
        sys.exit(main(args))
    else:
        print("Unsupported cluster type: {}, see help".format(args.cluster_type))
