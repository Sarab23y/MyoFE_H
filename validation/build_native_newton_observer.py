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


def build(output):
    if not sys.platform.startswith('linux'):
        raise RuntimeError('ELF LD_PRELOAD observer requires the actual Linux FEniCS container')
    if platform.machine() not in ('x86_64','amd64'):
        raise RuntimeError('review explicit-this forwarding ABI for this architecture')
    import dolfin
    if dolfin.__version__ != '2019.1.0':
        raise RuntimeError('review ABI/source before using a different DOLFIN: '+dolfin.__version__)
    directory=subprocess.check_output(['pkg-config','--variable=libdir','dolfin']).decode().strip()
    candidates=glob.glob(os.path.join(directory,'libdolfin.so'))
    if len(candidates)!=1:
        raise RuntimeError('cannot uniquely locate libdolfin.so in '+directory)
    library=os.path.realpath(candidates[0])
    raw=subprocess.check_output(['nm','-D','--defined-only',library]).decode().splitlines()
    symbols=[line.split()[-1] for line in raw if line.split()]
    demangled=subprocess.check_output(['c++filt'],input=('\n'.join(symbols)+'\n').encode()).decode().splitlines()
    selected={}
    for key,prefix in [('CONVERGED','dolfin::NewtonSolver::converged('),
                       ('UPDATE','dolfin::NewtonSolver::update_solution(')]:
        found=[symbol for symbol,name in zip(symbols,demangled) if name.startswith(prefix)]
        if len(found)!=1:
            raise RuntimeError('expected one exported '+prefix+', got '+str(found))
        selected[key]=found[0]
    os.makedirs(output)  # no reuse, including failed builds
    with open(os.path.join(output,'newton_symbols.h'),'w') as handle:
        for key,value in sorted(selected.items()):
            handle.write('#define MYOFE_%s_SYMBOL "%s"\n'%(key,value))
    source=os.path.join(os.path.dirname(os.path.abspath(__file__)),'native_newton_observer.cpp')
    flags=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','dolfin']).decode())
    target=os.path.join(os.path.realpath(output),'libmyofe_newton_observer.so')
    compiler=next((os.path.join(directory,'mpicxx') for directory in os.environ.get('PATH','').split(os.pathsep)
                   if os.access(os.path.join(directory,'mpicxx'),os.X_OK)), 'c++')
    command=[compiler,'-std=c++11','-fPIC','-shared',source,'-I'+os.path.realpath(output),'-o',target]+flags+['-fvisibility=default','-ldl']
    subprocess.check_call(command)
    def digest(path):
        h=hashlib.sha256()
        with open(path,'rb') as handle:
            for data in iter(lambda:handle.read(1048576),b''):h.update(data)
        return h.hexdigest()
    with open(os.path.join(output,'build.json'),'w') as handle:
        json.dump(dict(dolfin=dolfin.__version__,library=library,library_sha256=digest(library),
            source_sha256=digest(source),hook_sha256=digest(target),symbols=selected,
            command=command,compiler=subprocess.check_output([compiler,'--version']).decode(),
            callback_reachability_validated=False,physical_invariance_validated=False),handle,indent=2)
    print(target)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    build(p.parse_args().output)
