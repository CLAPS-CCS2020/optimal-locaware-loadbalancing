
LOCATION_TO_USERS=deNasa/lapsit_denasa/asn_to_users.json
OBJ_FUNC=$1
PMATRIX=deNasa/lapsit_denasa/denasa_penalty.json
CLUSTER_FILE=out/clustersidentity_1551398400
OUT=$2
THETA=$3
NAME=lapsit

python3 weights_optimization.py --tor_users_to_location $LOCATION_TO_USERS --json --obj_function $OBJ_FUNC --theta $THETA --pmatrix $PMATRIX --cluster_file $CLUSTER_FILE --out_dir $OUT --network_state out/network-state-2019-03/2019-03-01-00-00-00-network_state 2>&1

./build/bin/clp /home/frochet/Tor/optimal-locaware-loadbalancing/$OUT/location_aware_$NAME\_with_obj_$OBJ_FUNC\_theta_$THETA.mps -minimize -dualsimplex -solution /home/frochet/Tor/optimal-locaware-loadbalancing/$OUT/$NAME\_obj$OBJ_FUNC\_theta_$THETA.sol 2>&1


