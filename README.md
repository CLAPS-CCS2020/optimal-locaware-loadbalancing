# optimal-locaware-loadbalancing

Set of scripts aimed at deriving scores to select relays in a
location-aware fashion while still load-balancing the
Tor network.

## Goals

  - Easily deployable / implementable system; Does not change the
    fundamental logic of Tor's weight selection, but only *how*
    those weights are computed to offer location-aware protection

  - We offer a destination-naive framework to target any objective function. it
    could be one of the previous works. We specify the problem as a Linear
    Optimization and improve on multi dimensions specifying security constraints

  - All relays have the *same* load factor as Vanilla Tor. That is, the network
    is load-balanced with the same assumptions as Tor today: iff the external
    systems providing information are rights. (constraint)
  
  - \theta-GP-secure (constraint).

  - No single user location behaves worse than with Vanilla Tor (constraint)

Our system requires one more information compared to Vanilla Tor to work. In the
same logic that Vanilla Tor could not achieve to balance users among relay
without knowing how much bandwidth relay have, we cannot achieve load-balancing
with a location-aware path selection without knowing both how much bandwidth
relays have and how much clients are connecting from a given location. That is,
we require to establish a method to evaluate client density among all possible
locations. In the following Procedures, we first cover what could potentially be
an option to evaluate client density, and then we explain how to specify the
Linear Programming problem.

## Client locations - Per AS

    Offers an heuristic to evaluate client distribution per AS without
    relying on new measurements (we only use the data which Tor already
    provides - i.e., an estimation of Tor users per country). This heuristic
    is a fundamental block of the load-balancing system, like Torflow
    measurements are.
    To map towards a distribution of clients per AS, we may either:  

    - Compute the joint distribution of Tor clients per country with
        the number of available IP addresses per custumer ASes (using
        BGP databases).  

    - Compute the joint distribution of Tor clients per country with
        the distribution of Ping-responsive IP addresses per custumer
        ASes (using RIPE atlas probes). 

    - Compute the joint distribution of Tor clients per country with
        any genius idea about measuring the proportion of people per AS.
        What about using information on Netflix strategic deployment
        (i.e., they try to put servers close to people)?
        (paper: Open Connect Everywhere: A Glimpse at the Internet
        Ecosystem through the Lens of the Netflix CDN)
  
    Estimated work for procedure A: more than 1 week full time.  

  
## Weight calculation - Counter-Raptor like problem

   Modeling to derive guard and middle weights with penalty matrix for
   guard usage only.
   Performance constraints for load-balancing consider that the exit
   position total bandwdith is scarce

   Apply the following minmax optimization problem to derive weights for each
     location  

   Let W_l the density of Clients per AS computed from A), such that
   \sum W_l = 1  
   Let R_l a discrete distribution of scores to compute for guard selection
   common to all clients in location l.  
   Let G the total consensus weight of the guard-flagged relays  
   Let Wgg the fraction of bandwidth of each guard-flagged relay
   dedicated to the entry position, computed as described in
   dir-spec.txt Section 3.8.4.  
   Let L = \sum W_l\*R_l, a mixed distribution accounting for client density  
   Let P a matrix called Penalty Matrix, for all locations and for all guards.
   P\_{i,j} gives a penalty score associated with the path between location i
   and guard j (higher is worse).  
   Let V_i the expected penalty under vanilla Tor for location i computed as:
   V_i = \sum{G_j} (Pr_i * P_{i,j}

   We want to find an allocation of weights for each R_l such that:

      min max_i ( [\sum_{i} L(j)*P_{i,j}  for j in all guards])
  
   alternatively, the following optimization functions are considered:
      
      min max_i ([\sum_{j} W(i)*R(i,j)*P_{i, j}  for i in all locations])
      
      minimize the total vulnerability experienced by clients as a
      whole:
      min_R (\sum_i \sum_j W(j)*R(i,j)*P_{i,j})
      
      minimize the largest (over all client locations i) *expected*
      vulnerability:
  
      min max_i (\sum_j R(i,j)*P_{i,j}) 
   
   under constraints:
      
      1) for l in allLocation:
           for i in AllGuards:
             R_l(i) >= 0
      
      2) for l in AllLocations:
          \sum R_l(i) = G*Wgg
      
      3) for i in allGuards:
          L(i) <= BW_i

      4) for l in AllLocations:
          max_l max_i R_l(i)/(\sum_{j}R_l(j)*relCost(i)) <= \theta
      
      5) \sum_{G_k}(R_l(k) * P_{j,k}) <= V_j * \sum_{G_i}(BW_i) 
 
  Constraints 1), 2) and 3) guarantee to preserve current Tor's
load-balancing system. Constraint 4) trade-off location-aware benefit with
defense against guard-placement attacks, as Aaron pointed out in his
07.03.19 email. Constraint 5) enforces no worse than vanilla for any location

  Assuming a solver works on this (should be); then we may know compute
the set of weights for the middle position (global for all clients):

  Wmg_i = 1 - (L(i)/BW_i)

  Each Tor clients located in location l selects guard with Pr(G=i) = R_l(i)/\sum(R(j)) and
middle relay Pr(M=i) = Wmg_i\*BW_i/\sum(Wmg_j\*BW_j)

  This procedure guarantees a load factor of 1 on each relay, with a
minimization of the expected path penalty (for whatever definition of penalty).

  Estimated work for procedure B: more than one week full time -
Difficult bits: computing P, and modelizing the above problem such
that it is efficiently solved.

## Weight calculation - DeNASA-like problems

TODO



## Install 

# Requirements 

- stem, pandas, numpy, pycountry
- download descriptors from collector.torproject.org
- install Clp using coinbrew install script
- download userstats info from metrics
- Rob's shadow map built from Ripe Atlas Probes

## Steps

- Produce network_state files with process_consensuses.py 
- Produce Tor client distribution

### Counter-Raptor General analysis 

- Get the penalty matrix and the expected vanilla penalties
- Get the cluster info
- Produce the clusterd guard identity file (useless step currently
  though. Has been created for, apparently, uncessary optimization)
- Edit model_and_solve_cr.sh's variable path and run it
- Edit run_cr_analyze_and_plot.sh and run it to get some nicy figure

### Counter-Raptor Shadow analysis

- Get tor-claps (https://github.com/frochet/tor-claps-0.3.5.8)
- Get Shadow and the new shadow tor plugin (frochet's version)
  - https://github.com/frochet/shadow-tor-private
- Generate simulation files (the output relay.choices.csv must be used
  to generate penalies)
  - The stage command takes 2 more inputs (the city_trait file and the
    UN WUP2018 file). Both are inside the resource directory of my
    shadow-tor-private repo (see output of shadowtortools stage --help)
  - The generate step takes one more argument (city_probs) which is the
    path to the city distribution probability computed at the stage
    command
- Get the penalty matrix and the expected vanilla penalties
- Get the cluster info
- Edit model_and_solve_cr_shadow.sh's variable path and run it (if
  you're expect to run a CLAPS simulation)
- Convert the sol file to its shadow format using
  convert_solution_to_shadow_format.py (Not yet available)
- Convert Counter-Raptor weights to its shadow format using
  convert_solution_to_shadow_format.py (partially available ~ Resilience
  not account the last part of the Counter-Raptor paper, where
  Resiliences are re-computed with some smoothing)
- use the shadowtortools post tools to update the topology with the
  cluster (see shadowtortools postprodclust --help)

