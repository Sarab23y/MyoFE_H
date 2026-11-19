import copy
import unittest
from validation.rank_scatter_audit import audit_rank_scatter


class RankScatterTests(unittest.TestCase):
    def payloads(self):
        return [dict(production_ids=ids, authoritative_ids=ids[:],
                     ownership_range=[min(ids), max(ids)+1],
                     owned_indices=list(range(len(ids))), values=[17*i+.25 for i in ids])
                for ids in ([1, 0], [3, 2])]

    def test_reordered_ranks_scatter_by_explicit_ids(self):
        self.assertTrue(audit_rank_scatter(self.payloads(), 4)['passed'])
        self.assertTrue(audit_rank_scatter(self.payloads()[::-1], 4)['passed'])

    def test_wrong_owned_or_global_or_marker_is_rejected(self):
        for key, value in [('production_ids', [0, 1]), ('owned_indices', [1, 0]),
                           ('values', [.25, 34.25]), ('production_ids', [2, 2]),
                           ('ownership_range', [0, 3]),
                           ('production_ids', [2]), ('production_ids', [2, 99])]:
            payloads = copy.deepcopy(self.payloads())
            payloads[0][key] = value
            self.assertFalse(audit_rank_scatter(payloads, 4)['passed'], key)


if __name__ == '__main__':
    unittest.main()
