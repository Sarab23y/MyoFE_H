"""Python-2-compatible diagnostic of production-style rank/global assignment."""
from __future__ import division


def audit_rank_scatter(payloads, dimension):
    """Compare dofmap.dofs() scatter with authoritative local/global markers.

    The caller obtains all arrays from actual scalar FE owned vectors. This
    helper neither imports MPI nor substitutes a fabricated FE mapping.
    """
    output = [None]*dimension
    errors = []
    seen = set()
    for rank, item in enumerate(payloads):
        ids, authoritative, local, values = [item[k] for k in
            ('production_ids', 'authoritative_ids', 'owned_indices', 'values')]
        n = len(values)
        if not (len(ids) == len(authoritative) == len(local) == n):
            errors.append('rank %d owned/vector/ID length mismatch' % rank)
            continue
        if list(local) != list(range(n)):
            errors.append('rank %d owned indices are not the vector prefix' % rank)
        bounds = item.get('ownership_range')
        if (bounds is None or len(bounds) != 2 or
                sorted(authoritative) != list(range(int(bounds[0]), int(bounds[1])))):
            errors.append('rank %d authoritative IDs differ from FE ownership range' % rank)
        if list(ids) != list(authoritative):
            errors.append('rank %d production dofs differ from local_to_global_index' % rank)
        for gid, actual_gid, value in zip(ids, authoritative, values):
            if gid != int(gid) or not 0 <= gid < dimension or gid in seen:
                errors.append('duplicate/invalid global ID on rank %d' % rank)
                continue
            seen.add(gid)
            output[int(gid)] = value
            if value != 17*actual_gid + .25:
                errors.append('rank %d marker/value mismatch' % rank)
    if seen != set(range(dimension)):
        errors.append('global coverage mismatch')
    scatter_equal = all(v == 17*i+.25 for i, v in enumerate(output))
    return {'passed': not errors and scatter_equal,
            'errors': errors[:100], 'error_count': len(errors),
            'owned_point_count': len(seen), 'global_dimension': dimension,
            'rank_to_global_scatter_verified': scatter_equal,
            'evidence_class': 'source-confirmed FE mapping check when called with real FE vectors'}
