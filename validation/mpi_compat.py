"""Communicator compatibility for diagnostic scripts (Python 2.7/3).

Keep DOLFIN's original communicator for DOLFIN calls. Use the mpi4py view
of that same communicator for Python-object collectives.
"""
from __future__ import print_function

if __package__:
    from .diagnostic_io import write_report, root_call
else:
    from diagnostic_io import write_report, root_call


def world_comm(dolfin):
    """Support both the 2019 MPI.comm_world and legacy factory API."""
    if hasattr(dolfin.MPI, 'comm_world'):
        return dolfin.MPI.comm_world
    return dolfin.mpi_comm_world()


def as_mpi4py(comm):
    """Return the same MPI group/context, rejecting unsupported wrappers.

    Never substitute MPI.COMM_WORLD: callers may supply a subcommunicator.
    The returned communicator is borrowed and must not be freed here.
    """
    methods = ('Get_rank', 'Get_size', 'gather', 'allgather', 'bcast')
    if all(callable(getattr(comm, name, None)) for name in methods):
        if comm.Get_size() < 1 or not 0 <= comm.Get_rank() < comm.Get_size():
            raise ValueError('Null or invalid diagnostic communicator')
        return comm
    convert = getattr(comm, 'tompi4py', None)
    if not callable(convert):
        raise TypeError('Unsupported diagnostic communicator: %s' % type(comm).__name__)
    converted = convert()
    if not all(callable(getattr(converted, name, None)) for name in methods):
        raise TypeError('tompi4py() did not return an object-collective communicator')
    if converted.Get_size() < 1 or not 0 <= converted.Get_rank() < converted.Get_size():
        raise ValueError('Null or invalid converted communicator')
    for original, accessor in (('getRank', 'Get_rank'), ('getSize', 'Get_size')):
        query = getattr(comm, original, None)
        if callable(query) and query() != getattr(converted, accessor)():
            raise ValueError('Communicator conversion changed %s' % original)
    return converted


def probe(output, expected_size):
    """Exercise real mesh/PETSc conversion and a reversed subcommunicator."""
    import json
    import dolfin
    from mpi4py import MPI
    from petsc4py import PETSc

    mesh = dolfin.UnitCubeMesh(2, 2, 2)
    raw = mesh.mpi_comm()
    comm = as_mpi4py(raw)
    wrapped = PETSc.Comm(comm)
    converted = as_mpi4py(wrapped)
    same = MPI.Comm.Compare(comm, converted) == MPI.IDENT
    subgroup = comm.Split(color=comm.Get_rank() % 2, key=-comm.Get_rank())
    subwrapped = PETSc.Comm(subgroup)
    subview = as_mpi4py(subwrapped)
    sub_same = MPI.Comm.Compare(subgroup, subview) == MPI.IDENT
    local = {'rank': comm.Get_rank(), 'size': comm.Get_size(),
             'world_rank': as_mpi4py(world_comm(dolfin)).Get_rank(),
             'raw_type': type(raw).__module__ + '.' + type(raw).__name__,
             'petsc_conversion_same_context': bool(same),
             'subgroup_conversion_same_context': bool(sub_same),
             'subgroup_rank': subview.Get_rank(), 'subgroup_size': subview.Get_size()}
    sub_members = subview.allgather(comm.Get_rank())
    local['subgroup_reversed_rank_order'] = sub_members == sorted(sub_members, reverse=True)
    ranks = converted.gather(local, root=0)
    def finish():
        passed = (comm.Get_size() == expected_size and
                  [r['rank'] for r in ranks] == list(range(expected_size)) and
                  all(r['size'] == expected_size and r['petsc_conversion_same_context'] and
                      r['subgroup_conversion_same_context'] and r['subgroup_reversed_rank_order']
                      for r in ranks))
        write_report(output, {'passed': bool(passed), 'mpi_size': comm.Get_size(), 'ranks': ranks})
        return passed
    passed = root_call(comm, finish)
    # Drop the PETSc view before releasing the explicitly created subgroup.
    del subview
    del subwrapped
    subgroup.Free()
    return 0 if passed else 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--expected-size', required=True, type=int)
    args = parser.parse_args()
    raise SystemExit(probe(args.output, args.expected_size))
