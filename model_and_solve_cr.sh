LOCATION_TO_USERS=cr_penalty_matrix/asn_to_users-1k.json
OBJ_FUNC=$1
PMATRIX=cr_penalty_matrix/cr_penalty.json
VANILLA_EPL=cr_penalty_matrix/cr_vanilla_epl.json
CLUSTER_CLIENT_INFO=cr_penalty_matrix/raptor_l1/
CLUSTER_GUARD_FILE=cr_penalty_matrix/clustersidentity_1552608000
OUT=$2
THETA=$3
PWD=$(pwd)

for filepath in $CLUSTER_CLIENT_INFO*; do
  FILE=$(basename -- "$filepath")
  echo "file:$FILE"
  python3 weights_optimization.py --tor_users_to_location $LOCATION_TO_USERS --json --obj_function $OBJ_FUNC --theta $THETA --pmatrix $PMATRIX --penalty_vanilla $VANILLA_EPL --client_clust_representative $CLUSTER_CLIENT_INFO$FILE --cluster_file $CLUSTER_GUARD_FILE --out_dir $OUT --network_state out/network-state-2019-03/2019-03-01-00-00-00-network_state 2>&1

  ./build/bin/clp $PWD/$OUT/location_aware_with_obj_$OBJ_FUNC\_theta_$THETA.mps -minimize -dualsimplex -solution $PWD/$OUT/func_obj$OBJ_FUNC\_theta_$THETA\_$FILE.sol 2>&1
done

