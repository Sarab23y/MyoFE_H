// Opt-in Linux ELF interposition. Original DOLFIN algorithms remain in libdolfin.
// Build against the exact container headers/library. Never install over DOLFIN.
#include <dolfin/nls/NewtonSolver.h>
#include <dolfin/nls/NonlinearProblem.h>
#include <dolfin/la/GenericVector.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <string>
#include "newton_symbols.h"

namespace {
using Callback = void (*)(int, std::size_t, double, double, double, int, std::size_t);
Callback callback = nullptr;
thread_local bool observing = false;
template <typename T> T original(const char* symbol) {
  void* address = dlsym(RTLD_NEXT, symbol);
  if (!address) { std::fprintf(stderr, "MyoFE observer cannot resolve %s: %s\n", symbol, dlerror()); std::abort(); }
  return reinterpret_cast<T>(address);
}
void emit(int event, std::size_t iteration, double residual, double relative,
          double relaxation, int converged, std::size_t linear) {
  if (callback && !observing) {
    observing = true;
    callback(event, iteration, residual, relative, relaxation, converged, linear);
    observing = false;
  }
}
}
extern "C" void myofe_newton_set_callback(Callback value) { callback = value; }
extern "C" int myofe_newton_observer_abi() { return 1; }

bool dolfin::NewtonSolver::converged(const dolfin::GenericVector& r,
    const dolfin::NonlinearProblem& problem, std::size_t iteration) {
  using Fn = bool (*)(dolfin::NewtonSolver*, const dolfin::GenericVector&,
                     const dolfin::NonlinearProblem&, std::size_t);
  static Fn forward = original<Fn>(MYOFE_CONVERGED_SYMBOL);
  const bool result = forward(this, r, problem, iteration);
  const std::string criterion = parameters["convergence_criterion"];
  if (criterion != "residual") {
    emit(9, iteration, 0., 0., 0., result, krylov_iterations());
    return result; // preserve original behavior, refuse misleading telemetry
  }
  emit(0, iteration, residual(), relative_residual(),
       std::numeric_limits<double>::quiet_NaN(), result, krylov_iterations());
  return result;
}
void dolfin::NewtonSolver::update_solution(dolfin::GenericVector& x,
    const dolfin::GenericVector& dx, double relaxation,
    const dolfin::NonlinearProblem& problem, std::size_t iteration) {
  using Fn = void (*)(dolfin::NewtonSolver*, dolfin::GenericVector&,
      const dolfin::GenericVector&, double, const dolfin::NonlinearProblem&, std::size_t);
  static Fn forward = original<Fn>(MYOFE_UPDATE_SYMBOL);
  emit(1, iteration, 0., 0., relaxation, 0, krylov_iterations());
  forward(this, x, dx, relaxation, problem, iteration);
  emit(2, iteration + 1, 0., 0., relaxation, 0, krylov_iterations());
}
