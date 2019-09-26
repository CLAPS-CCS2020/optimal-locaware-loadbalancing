LOCATION_TO_USERS=shadow_data/client-distribution.json
OBJ_FUNC=3
PMATRIX=shadow_data/counter_raptor/penalty/shadow_cr_penalties.json
VANILLA_EPL=shadow_data/counter_raptor/penalty/shadow_cr_epls.json
CLUSTER_CLIENT_INFO=shadow_data/counter_raptor/clusters/
NETWORK_STATE=shadow_data/shadow_relay_dump.json
CLP_PATH=./build/bin/clp
OUT=$1
THETA=5.0
PWD=$(pwd)

for filepath in $CLUSTER_CLIENT_INFO*; do
  FILE=$(basename -- "$filepath")
  echo "file:$FILE"
  python3 weights_optimization.py CR_SHADOW --tor_users_to_location $LOCATION_TO_USERS --obj_function $OBJ_FUNC --theta $THETA --pmatrix $PMATRIX --penalty_vanilla $VANILLA_EPL --client_clust_representative $CLUSTER_CLIENT_INFO$FILE --out_dir $OUT --shadow_relay_dump $NETWORK_STATE 2>&1

  $CLP_PATH $PWD/$OUT/location_aware_with_obj_$OBJ_FUNC\_theta_$THETA.mps -minimize -dualsimplex -solution $PWD/$OUT/func_obj$OBJ_FUNC\_theta_$THETA\_$FILE.sol 2>&1
done

