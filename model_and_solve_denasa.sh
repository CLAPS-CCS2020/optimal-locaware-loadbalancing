LOCATION_TO_USERS=denasa_complete/asn_to_users.json
OBJ_FUNC=$1
PMATRIX=denasa_complete/denasa_penalty.json
#path to vanilla expected penalties for suspicious ASes
VANILLA_EPL=denasa_complete/denasa_vanilla_epl.json
#Directory which holds only all clustering_n_badness.txt files
CLUSTER_CLIENT_INFO=denasa_complete/clustering_files/
#contains all guard info
CLUSTER_GUARD_FILE=denasa_complete/clustersidentity_1552608000
#contains slim network state info:
NETWORK_STATE=out/network-state-2019-03/2019-03-01-00-00-00-network_state
#in which directory .mps and .sol are written
OUT=$2
#GP attack parameter
THETA=8.0
PWD=$(pwd)

for filepath in $CLUSTER_CLIENT_INFO*; do
  FILE=$(basename -- "$filepath")
  echo "file:$FILE"
  python3 weights_optimization.py --tor_users_to_location $LOCATION_TO_USERS --json --obj_function $OBJ_FUNC --theta $THETA --pmatrix $PMATRIX --penalty_vanilla $VANILLA_EPL --client_clust_representative $CLUSTER_CLIENT_INFO$FILE --cluster_file $CLUSTER_GUARD_FILE --out_dir $OUT --network_state $NETWORK_STATE 2>&1

  ./build/bin/clp $PWD/$OUT/location_aware_with_obj_$OBJ_FUNC\_theta_$THETA.mps -minimize -dualsimplex -solution $PWD/$OUT/func_obj$OBJ_FUNC\_theta_$THETA\_$FILE.sol 2>&1
done

