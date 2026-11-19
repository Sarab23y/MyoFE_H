"""Exercise communicator adapters without requiring a local MPI runtime."""
import types
import unittest

from validation.mpi_compat import as_mpi4py, world_comm


class ObjectComm:
    def __init__(self, rank, size):
        self.rank, self.size = rank, size
        self.calls = []

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def gather(self, value, root=0):
        self.calls.append(('gather', value, root))
        return [value] if self.rank == root else None

    def allgather(self, value):
        return [value] * self.size

    def bcast(self, value, root=0):
        self.calls.append(('bcast', value, root))
        return value


class PETScComm:
    """Expose PETSc-style methods, deliberately omitting gather."""
    def __init__(self, comm):
        self.comm = comm
        self.conversions = 0

    def getRank(self):
        return self.comm.rank

    def getSize(self):
        return self.comm.size

    def tompi4py(self):
        self.conversions += 1
        return self.comm


class CommCompatibilityTests(unittest.TestCase):
    def test_null_or_out_of_range_rank_is_rejected(self):
        for rank, size in ((-1, 0), (-1, 32), (32, 32)):
            with self.assertRaises(ValueError):
                as_mpi4py(ObjectComm(rank, size))
            with self.assertRaises(ValueError):
                as_mpi4py(PETScComm(ObjectComm(rank, size)))

    def test_mpi4py_is_preserved_without_duplication(self):
        comm = ObjectComm(7, 32)
        self.assertIs(as_mpi4py(comm), comm)

    def test_petsc_wrapper_has_no_gather_and_converts(self):
        original = ObjectComm(0, 32)
        wrapper = PETScComm(original)
        self.assertFalse(hasattr(wrapper, 'gather'))
        converted = as_mpi4py(wrapper)
        self.assertIs(converted, original)
        self.assertEqual(converted.gather({'rank': 0}, root=0), [{'rank': 0}])
        self.assertEqual(wrapper.conversions, 1)

    def test_all_32_local_ranks_and_root_behavior_preserved(self):
        for rank in range(32):
            original = ObjectComm(rank, 32)
            converted = as_mpi4py(PETScComm(original))
            self.assertEqual((converted.Get_rank(), converted.Get_size()), (rank, 32))
            self.assertEqual(converted.gather(rank, root=0), [rank] if rank == 0 else None)

    def test_serial_collectives_and_boolean_consensus(self):
        converted = as_mpi4py(PETScComm(ObjectComm(0, 1)))
        self.assertEqual(converted.gather('point'), ['point'])
        self.assertTrue(all(converted.allgather(True)))
        self.assertFalse(all(converted.allgather(False)))
        self.assertEqual(converted.bcast('result'), 'result')

    def test_subcommunicator_is_preserved(self):
        subgroup = ObjectComm(2, 5)
        self.assertIs(as_mpi4py(PETScComm(subgroup)), subgroup)

    def test_conversion_rank_and_size_changes_are_rejected(self):
        for bad in (ObjectComm(1, 32), ObjectComm(0, 16)):
            wrapper = PETScComm(ObjectComm(0, 32))
            wrapper.tompi4py = lambda: bad
            with self.assertRaises(ValueError):
                as_mpi4py(wrapper)

    def test_unsupported_and_invalid_conversion_fail_explicitly(self):
        with self.assertRaises(TypeError):
            as_mpi4py(object())
        wrapper = PETScComm(ObjectComm(0, 1))
        wrapper.tompi4py = lambda: object()
        with self.assertRaises(TypeError):
            as_mpi4py(wrapper)

    def test_modern_and_legacy_world_api(self):
        comm = ObjectComm(0, 32)
        modern = types.SimpleNamespace(MPI=types.SimpleNamespace(comm_world=comm))
        self.assertIs(world_comm(modern), comm)
        legacy = types.SimpleNamespace(MPI=types.SimpleNamespace(), mpi_comm_world=lambda: comm)
        self.assertIs(world_comm(legacy), comm)


if __name__ == '__main__':
    unittest.main()
