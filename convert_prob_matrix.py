from slim_desc import CluserRouter
import json
import os, sys
from tor_users_per_country import get_network_state
import argparse
"""

Current format of the probmatrix, e.g., for lastor is:

Nested dictionaries, maps city code -> relay fp -> penalty score.  The
penalty is the location's geographic distance to the relay.  A relay has
infinite penalty if the relay's location could not be determined.

This script expects to convert the matrix to:

Nested dictionaries; maps city code -> slim_desc.ClusterRouter -> penalty score.

The main reason for this change is that the modelization script used to define
the objective and constraints manipulates ClusterRouter objects for guards
clustered if the penalty is expected to be the same value. ClusterRouter
objects can hold only 1 guard.

"""

parser = argparse.ArgumentParser()
parser.add_argument("--pmatrix", type=str)
parser.add_argument("--network_state", type=str)
parser.add_argument("--out_dir", type=str)

if __name__ == "__main__":

    args = parser.parse_args()
    with open(pmatrix_file, "r") as f:
        pmatrix = json.load(f)

    new_pmatrix = {k, {} for (k, v) in pmatrix.items()}
    network_state = get_network_state(args.network_state)

    ## all guards fp from lastor_penalty should be inside the network_state. If
    ## not, we have a sync issue :)
    
    for loc in pmatrix:
        ignore_guard = {}
        for guardfp in pmatrix[loc]:
            if guardfp not in ignore_guard:
                rel_stat = network_state.cons_rel_stat[guardfp]
                cluster = ClusterRouter()
                cluster.addRouter(rel_stat)
                for nguardfp in pmatrix[loc]:
                    if pmatrix[loc][guardfp] == pmatrix[loc][nguardfp]:
                        cluser.addRouter(rel_stat)
                        ignore_guard[nguardfp] = True
                new_pmatrix[loc][cluster] = pmatrix[loc][guardfp]
    
    with open(os.path.join(args.out_dir, "penalty_matrix_clustered.json", "w") as f:
        json.dump(new_pmatrix, f)
    




    
