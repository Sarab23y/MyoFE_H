"""Compile isolated ELF hook using actual image symbols; no library replacement."""
from __future__ import print_function
import argparse
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys
import shutil
import platform


# Profiles are explicit, not a permissive major-version test. Both use the
# protected virtual hook contract, verified against actual headers AND symbols.
SUPPORTED = {'2017.1.0': 'dolfin-2017.1-native-v1',
             '2019.1.0': 'dolfin-2019.1-native-v1'}
EXPECTED = {
 'CONVERGED': 'dolfin::NewtonSolver::converged(dolfin::GenericVector const&, dolfin::NonlinearProblem const&, unsigned long)',
 'UPDATE': 'dolfin::NewtonSolver::update_solution(dolfin::GenericVector&, dolfin::GenericVector const&, double, dolfin::NonlinearProblem const&, unsigned long)'}


def select_symbols(version, symbols, demangled):
    if version not in SUPPORTED:
        raise RuntimeError('unsupported DOLFIN observer profile: '+version)
    if len(symbols) != len(demangled):
        raise RuntimeError('symbol/demangler count mismatch')
    selected = {}
    for key, expected in EXPECTED.items():
        prefix = expected.split('(')[0]+'('
        found = [(symbol, name) for symbol,name in zip(symbols,demangled)
                 if name.startswith(prefix)]
        if len(found) != 1 or found[0][1] != expected:
            raise RuntimeError('unreviewed '+version+' native signature for '+key+': '+str(found))
        selected[key] = found[0][0]
    return selected


def demangle(symbols):
    # check_output(input=...) is NOT supported in the production Python 2.7.
    proc = subprocess.Popen(['c++filt'], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = proc.communicate(('\n'.join(symbols)+'\n').encode('ascii'))
    if proc.returncode:
        raise RuntimeError('c++filt failed: '+error.decode('utf-8','replace'))
    return output.decode('utf-8').splitlines()


CONTRACT = r"""
#include <dolfin/nls/NewtonSolver.h>
#include <dolfin/nls/NonlinearProblem.h>
#include <dolfin/la/GenericVector.h>
#include <type_traits>
// override rejects nonvirtual/changed hooks at compile time, without instantiating
// or replacing the actual production solver. Compile only; never linked/run.
struct MyoFEContract : dolfin::NewtonSolver {
 bool converged(const dolfin::GenericVector&, const dolfin::NonlinearProblem&,
                std::size_t) override;
 void update_solution(dolfin::GenericVector&, const dolfin::GenericVector&, double,
                      const dolfin::NonlinearProblem&, std::size_t) override;
};
static_assert(std::is_same<decltype(((const dolfin::NewtonSolver*)0)->residual()), double>::value, "residual API changed");
static_assert(std::is_same<decltype(((const dolfin::NewtonSolver*)0)->relative_residual()), double>::value, "relative residual API changed");
static_assert(std::is_same<decltype(((const dolfin::NewtonSolver*)0)->krylov_iterations()), std::size_t>::value, "linear iteration API changed");
"""

def build(output):
    if not sys.platform.startswith('linux'):
        raise RuntimeError('ELF LD_PRELOAD observer requires the actual Linux FEniCS container')
    if platform.machine() not in ('x86_64','amd64'):
        raise RuntimeError('review explicit-this forwarding ABI for this architecture')
    import dolfin
    if dolfin.__version__ not in SUPPORTED:
        raise RuntimeError('unsupported DOLFIN observer profile: '+dolfin.__version__)
    directory=subprocess.check_output(['pkg-config','--variable=libdir','dolfin']).decode().strip()
    candidates=glob.glob(os.path.join(directory,'libdolfin.so'))
    if len(candidates)!=1:
        raise RuntimeError('cannot uniquely locate libdolfin.so in '+directory)
    library=os.path.realpath(candidates[0])
    raw=subprocess.check_output(['nm','-D','--defined-only',library]).decode().splitlines()
    symbols=[line.split()[-1] for line in raw if line.split()]
    demangled=demangle(symbols)
    selected=select_symbols(dolfin.__version__, symbols, demangled)
    os.makedirs(output)  # no reuse, including failed builds
    with open(os.path.join(output,'newton_symbols.h'),'w') as handle:
        for key,value in sorted(selected.items()):
            handle.write('#define MYOFE_%s_SYMBOL "%s"\n'%(key,value))
    source=os.path.join(os.path.dirname(os.path.abspath(__file__)),'native_newton_observer.cpp')
    flags=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','dolfin']).decode())
    target=os.path.join(os.path.realpath(output),'libmyofe_newton_observer.so')
    compiler=next((os.path.join(directory,'mpicxx') for directory in os.environ.get('PATH','').split(os.pathsep)
                   if os.access(os.path.join(directory,'mpicxx'),os.X_OK)), 'c++')
    contract=os.path.join(os.path.realpath(output),'newton_contract.cpp')
    with open(contract,'w') as handle:handle.write(CONTRACT)
    header_check=[compiler,'-std=c++11','-fsyntax-only',contract]+flags
    subprocess.check_call(header_check)
    command=[compiler,'-std=c++11','-fPIC','-shared',source,'-I'+os.path.realpath(output),'-o',target]+flags+['-fvisibility=default','-Wl,-z,defs','-ldl']
    subprocess.check_call(command)
    def digest(path):
        h=hashlib.sha256()
        with open(path,'rb') as handle:
            for data in iter(lambda:handle.read(1048576),b''):h.update(data)
        return h.hexdigest()
    with open(os.path.join(output,'build.json'),'w') as handle:
        json.dump(dict(dolfin=dolfin.__version__,library=library,library_sha256=digest(library),
            source_sha256=digest(source),hook_sha256=digest(target),symbols=selected,
            adapter_profile=SUPPORTED[dolfin.__version__],header_contract_compiled=True,
            header_contract_sha256=digest(contract),header_contract_command=header_check,
            runtime_telemetry_available='unverified until native_1 and native_32 probes pass',
            command=command,compiler=subprocess.check_output([compiler,'--version']).decode(),
            callback_reachability_validated=False,physical_invariance_validated=False),handle,indent=2)
    print(target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    build(p.parse_args().output)
