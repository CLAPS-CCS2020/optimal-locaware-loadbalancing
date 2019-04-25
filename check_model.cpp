#include <coin/ClpSimplex.hpp>
#include <coin/ClpPresolve.hpp>


/**
 *
 * Step 1: Download and build clp locally; or install coinor-libclp-dev from
 * your packages if you're lucky to have it
 *
 * Step 2: g++ check_model.cpp -o check_model -lClp 
 *
 * (optionally -L/to/your/build/lib directory)
 *
 * This simple program aims at veryfing if the model is feasible x)
 */


int main (int argc, const char *argv[])
{
    
  ClpSimplex model;
  model.readMps(argv[1]); // initialized by readMps or whatever
  ClpPresolve presolveInfo;
  ClpSimplex * presolvedModel = presolveInfo.presolvedModel(model); 
  // at this point we have original model and a new model.  The  information
    if (!presolvedModel) {
    // if presolvedModel is NULL, then it was primal infeasible
      return 1;
  }
  // the model is feasible
  return 0;
}
