
import os, sys
import argparse
import pickle
from slim_desc import ClusterRouter
from tor_users_per_country import get_network_state
import requests
import time as dtime
from process_consensuses import timestamp

"""

This file takes an network state as an input, and cluster guard relays
by the n announced prefix from a same AS.0

"""

parser = argparse.ArgumentParser(description="get a Tor network state information and cluster guard relays")
parser.add_argument("--ns_file", help="slim_desc.NetworkState path to pickle file", required=True)
parser.add_argument("--out", help="out dir to store the clustered guards", required=True)

RIPE_BGP_INFO = "https://stat.ripe.net/data/bgp-state/data.json?resource="

def main(args):
    
    clusters = {} # key: string of prefix IP, val: slim_desc.ClusterRouter
    network_state = get_network_state(args.ns_file)
    descriptors = network_state.descriptors
    guardsfp = [relay for relay in network_state.cons_rel_stats if Flag.GUARD in network_state.cons_rel_stats[relay].flags and
            not Flag.EXIT in network_state.cons_rel_stats[relay].flags]
    #building clusters
    for guard in guardsfp:
        ipv4_address = descriptors[guard].address
        r = requests.get(RIPE_BGP_INFO+ipv4_address+"&timestamp="+timestamp(network_state.cons_valid_after))
        if r.status_code > 200:
            for _ in range(0,2):
                print("Status code of previous request: {} - Waiting 5 seconds and retry".format(r.status_code))
                dtime.sleep(5)
                if r.status_code < 300:
                    break
            if r.status_code > 200:
                raise ValueError("Something went wrong with the server")
        bgpinfo = r.json()
        prefix = bgpinfo['data']['resource']
        if prefix in clusters:
            clusters[prefix].addRouter(network_state.cons_rel_stats[guard])
        else:
            asn = int(bgpinfo['data']['bgp_state'][0]['path'][-1])
            clusters[prefix] = clusterRouter(asn, prefix)
            clusters[prefix].addRouter(network_state.cons_rel_states[guard])
    
    print("We had {} guards before and now {} clusters".format(len(guardsfp), len(list(clusters.keys()))))
    outpath = os.path.join(args.out, "clusters_"+network_state.cons_valid_after.strftime('%Y-%m-%d-%H-%M-%S'))
    with open(outpath, 'wb') as f:
        pickle.dump(clusters, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    args = parser.parse_args()

    sys.exit(main(args))
