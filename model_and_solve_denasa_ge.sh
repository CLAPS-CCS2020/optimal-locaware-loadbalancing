LOCATION_TO_USERS=denasa_complete/asn_to_users.json
OBJ_FUNC=3
PMATRIX_DENASA_G=denasa_complete/denasa_penalty.json
PMATRIX_DENASA_GE=
#path to vanilla expected penalties for suspicious ASes
VANILLA_EPL_DENASA_G=denasa_complete/denasa_vanilla_epl.json
VANILLA_EPL_DENASA_GE=
#Directory which holds only all clustering_n_badness.txt files
CLUSTER_CLIENT_INFO=denasa_complete/clustering_files/
#contains all guard info
#Should maybe cluster guards per AS later?
CLUSTER_GUARD_FILE=denasa_complete/clustersidentity_1552608000
#contains slim network state info:
NETWORK_STATE=out/network-state-2019-03/2019-03-15-00-00-00-network_state
# fingerprints to asn file
FP_TO_ASN=test
#in which directory .mps and .sol are written
OUT=$1
#GP attack parameter
THETA=5.0
CLP_PATH=./build/bin/clp

for filepath in $CLUSTER_CLIENT_INFO*; do
  FILE=$(basename -- "$filepath")
  python3 weights_optimization.py DeNASA --tor_users_to_location $LOCATION_TO_USERS --obj_function $OBJ_FUNC --theta $THETA --pmatrix $PMATRIX_DENASA_G --penalty_vanilla $VANILLA_EPL_DENASA_G --client_clust_representative $CLUSTER_CLIENT_INFO$FILE --cluster_file $CLUSTER_GUARD_FILE --out_dir $OUT --network_state $NETWORK_STATE 2>&1
  # run the solver
  $CLP_PATH $PWD/$OUT/location_aware_with_obj_$OBJ_FUNC\_theta_$THETA.mps -minimize -dualsimplex -solution $PWD/$OUT/denasa_g__obj$OBJ_FUNC\_theta_$THETA\_$FILE.sol 2>&1
  # run deNasa g&e with the solver's output
  python3 weights_optimization.py DeNASA_EXIT --tor_users_to_location $LOCATION_TO_USERS --obj_function $OBJ_FUNC --theta $THETA --pmatrix $PMATRIX_DENASA_GE --penalty_vanilla $VANILLA_EPL_DENASA_GE --client_clust_representative $CLUSTER_CLIENT_INFO$FILE --cluster_file $CLUSTER_GUARD_FILE --out_dir $OUT --network_state $NETWORK_STATE --deNasa_so_guards $PWD/$OUT/denasa_g_obj$OBJ_FUNC\_theta_$THETA\_$FILE.sol --fp_to_asn $FP_TO_ASN 2>&1
  $CLP_PATH $PWD/out/location_ware_with_obj_$OBJ_FUNC\_theta_$THETA.mps -minimize -dualsimplex -solution $PWD/$OUT/denasa_ge_obj$OBJ_FUNC\_theta_$THETA\_$FILE.sol 2>&1
done
