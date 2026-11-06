#!/usr/bin/env python3
"""Read-only full-run MyoFE plateau-data extractor. Python 3.6+, no packages.

Run: python3 extract_calcium_xi_baroreflex_no_numpy.py
Produces calcium_xi_baroreflex_full_run.zip beside sim_output.
Includes all scalar data.csv columns and selected spatial traces, not plots.
Missing fields are reported, never replaced by other stresses or by zeros.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

BASE = '/mnt/gpfs2_4m/scratch/sba431/demos/ECM/ECM_40s_healthy_quad_validation_optimized/sim_output'
IDS = [1477, 577, 4629, 4633, 2589, 3041, 1941, 1473]
CONTEXT = {'active_stress', 'hs_length', 'fiber_strain', 'fiber_stretch_raw', 'ecm_instantaneous_fiber_stress'}


def category(name):
    s = name.lower()
    if s.endswith('_data.csv'):
        s = s[:-9]
    elif s.endswith('.csv'):
        s = s[:-4]
    if s == 'sff_mesh':
        return 'Xi passive stress (Sff_mesh)'
    if s in {'myofiber_passive', 'xi_stress', 'xi_passive_stress', 'passive_myofiber_stress'}:
        return 'possible Xi field; definition needs verification'
    if ('calcium' in s or 'serca' in s or re.search(r'(^|_)ca($|_)', s)
            or s in {'cai', 'casr', 'cacytos', 'cacyto', 'ca_cyt', 'j_release', 'j_uptake', 'j_leak'}):
        return 'calcium handling'
    if (re.search(r'(^|_)k_(act|serca|leak|1|on)($|_)', s) or 'baro' in s
            or re.search(r'(^|_)b_(a|b|c|control|setpoint|drive|recov)($|_)', s)
            or s in {'heart_rate','t_quiescent_period','arterioles_resistance','veins_compliance'}):
        return 'control parameter or baroreflex signal'
    if s in CONTEXT:
        return 'mechanical context'
    return None


def inspect_header(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return [s.strip() for s in next(csv.reader(f), [])]


def extract(path, dest, full=False):
    """Retain original rows/values; explicitly select global DOF headers."""
    with path.open(newline='', encoding='utf-8-sig') as src, dest.open('w', newline='') as dst:
        reader = csv.reader(src)
        original = next(reader, None)
        if not original:
            raise ValueError('empty file')
        header = [s.strip() for s in original]
        if header.count('time') != 1:
            raise ValueError('expected one explicit time column; no invented timestamps')
        ti = header.index('time')
        lookup = {}
        for col, name in enumerate(header):
            if col == ti:
                continue
            try:
                gid = int(name)
            except ValueError:
                continue
            if gid in lookup:
                raise ValueError('duplicate numeric DOF label {}'.format(gid))
            lookup[gid] = col
        if full:
            columns = list(range(len(header)))
            mode = 'all original global columns'
        elif lookup:
            missing = [i for i in IDS if i not in lookup]
            if missing:
                raise ValueError('numeric columns present but selected DOF labels missing: {}'.format(missing))
            columns = [ti]+[lookup[i] for i in IDS]
            mode = 'selected global DOFs'
        else:
            columns = list(range(len(header)))
            mode = 'named scalar/aggregate columns; not treated as local point values'
        writer = csv.writer(dst)
        writer.writerow([original[i] for i in columns])
        n = 0; first = previous = last = None
        dtmin = dtmax = None
        nonfinite = {}; nonnumeric = {}
        for line, row in enumerate(reader, 2):
            if not row:
                continue
            if len(row) != len(header):
                raise ValueError('CSV width mismatch at line {}'.format(line))
            t = float(row[ti])
            if not math.isfinite(t) or (previous is not None and t <= previous):
                raise ValueError('nonfinite, duplicate or decreasing time at line {}'.format(line))
            if first is None:
                first = t
            if previous is not None:
                step = t-previous
                dtmin = step if dtmin is None else min(dtmin,step)
                dtmax = step if dtmax is None else max(dtmax,step)
            previous = last = t
            for i in columns:
                if i == ti:
                    continue
                try:
                    v = float(row[i])
                    if not math.isfinite(v):nonfinite[header[i]] = nonfinite.get(header[i],0)+1
                except ValueError:
                    nonnumeric[header[i]] = nonnumeric.get(header[i],0)+1
            # Preserve NaNs/strings and report them, rather than quietly deleting data.
            writer.writerow([row[i] for i in columns])
            n += 1
        if n < 2:
            raise ValueError('fewer than two time samples')
        return {'file':path.name, 'mode':mode, 'rows':n, 'first_time':first, 'last_time':last,
                'min_dt':dtmin, 'max_dt':dtmax, 'columns':[header[i] for i in columns],
                'nonfinite_counts':nonfinite, 'nonnumeric_counts':nonnumeric}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sim-output', default=BASE)
    ap.add_argument('--extra-file', action='append', default=[], help='Additional exact CSV filename inside sim_output')
    args = ap.parse_args()
    base = Path(args.sim_output)
    if not (base/'data.csv').is_file():
        raise ValueError('Cannot find {}'.format(base/'data.csv'))
    for filename in args.extra_file:
        if Path(filename).name != filename or not filename.endswith('.csv'):
            raise ValueError('--extra-file must be a CSV filename within sim_output')
    report = {'source_folder':str(base),'global_dof_ids':IDS,'exported':[], 'unavailable_or_failed':[],
              'discovery_note':'Filename/column matching finds candidates, not proof of definitions or units.',
              'processing':'All available timestamps; no smoothing, interpolation, averaging or unit conversion.',
              'plateau_assessment':'Not performed during extraction. Compare complete-beat transients and full-run control drift after upload.'}
    inventory=[]; chosen=[]
    for p in sorted(base.iterdir()):
        if not p.is_file():continue
        item={'filename':p.name,'bytes':p.stat().st_size}
        if p.suffix.lower()=='.csv':
            try:
                header=inspect_header(p)
                item.update({'columns':len(header),'header_first_15':header[:15],
                             'all_selected_dofs_present':all(str(i) in header for i in IDS)})
            except (OSError,UnicodeError,csv.Error) as exc:
                item['header_error']=str(exc)
            reason=category(p.name)
            if p.name!='data.csv' and (reason or p.name in args.extra_file):
                chosen.append((p,reason or 'explicit extra file'))
        inventory.append(item)
    for filename in args.extra_file:
        if not (base/filename).is_file():report['unavailable_or_failed'].append({'file':filename,'reason':'requested extra file absent'})
    with tempfile.TemporaryDirectory(prefix='ca_xi_baro_') as temp:
        work=Path(temp)
        print('Reading full data.csv (all times and scalar columns) ...',flush=True)
        info=extract(base/'data.csv',work/'data.csv',full=True)
        report['exported'].append(info)
        report['candidate_global_columns']={name:category(name) for name in info['columns'] if category(name)}
        print('  {} rows; {:.6f} to {:.6f} s; {} columns'.format(info['rows'],info['first_time'],info['last_time'],len(info['columns'])),flush=True)
        for p,reason in chosen:
            print('Reading {} ...'.format(p.name),flush=True)
            try:
                entry=extract(p,work/p.name)
                entry['candidate_category']=reason
                report['exported'].append(entry)
                print('  {} rows; {}'.format(entry['rows'],entry['mode']),flush=True)
            except (ValueError,OSError,csv.Error,UnicodeError) as exc:
                if (work/p.name).exists():(work/p.name).unlink()
                report['unavailable_or_failed'].append({'file':p.name,'reason':str(exc)})
                print('  NOT EXPORTED:',exc,flush=True)
        # Exact known Xi file is never replaced by total_passive or ECM stress.
        if not any(e['file'].lower()=='sff_mesh_data.csv' for e in report['exported']):
            report['unavailable_or_failed'].append({'file':'Sff_mesh_data.csv','reason':'known Xi field not exported; inspect inventory/global columns and source definition'})
        signals=' '.join(report['candidate_global_columns'])+' '+' '.join(e['file'] for e in report['exported'])
        report['calcium_concentration_candidate_detected']=bool(re.search(r'calcium|(^|[_\s])ca([_\s.]|$)|\bcai\b',signals.lower()))
        if not report['calcium_concentration_candidate_detected']:
            report['unavailable_or_failed'].append({'file':'calcium concentration','reason':'no recognized saved concentration candidate; rates alone cannot establish Ca transient repeatability'})
        for filename in ['output_metadata.json','ecm_baseline_stability.json']:
            source=base/filename
            if source.is_file():shutil.copyfile(str(source),str(work/filename))
        geometry=base.parent/'midwall_midventricle_selection_no_numpy'/'selected_midwall_midventricle_points.csv'
        if geometry.is_file():
            with geometry.open(newline='') as f:
                selected=list(csv.DictReader(f))
            if set(int(r['global_dof']) for r in selected)==set(IDS):
                shutil.copyfile(str(geometry),str(work/geometry.name))
            else:report['unavailable_or_failed'].append({'file':geometry.name,'reason':'IDs differ; geometry omitted'})
        (work/'extraction_report.json').write_text(json.dumps(report,indent=2))
        (work/'output_file_inventory.json').write_text(json.dumps(inventory,indent=2))
        (work/'READ_ME.txt').write_text(
            'Full-run calcium, Xi and baroreflex diagnostic data.\n'
            'All original scalar data.csv columns are included. Selected spatial fields retain the eight global DOF labels.\n'
            'Named scalar/aggregate files are explicitly marked; do not interpret them as individual material points.\n'
            'No source simulation files were modified. No missing values were fabricated.\n'
            'Read extraction_report.json for unavailable fields, nonfinite values and possible name matches.\n'
            'Sff_mesh is the Xi field identified by the prior output audit; verify exact stress definition and units before interpretation.\n'
            'No plateau decision is made here. Ca and passive stress require repeatable cycles, not flat instantaneous values.\n'
            'Controller signals may remain pulsatile: examine both cycle statistics and slow drift.\n'
            'The original 5% ECM-field criterion is not automatically a tolerance for every variable.\n')
        archive=work/'bundle.zip'
        with zipfile.ZipFile(str(archive),'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(work.iterdir()):
                if p!=archive:z.write(str(p),p.name)
        result=base.parent/'calcium_xi_baroreflex_full_run.zip'
        shutil.copyfile(str(archive),str(result))
    print('\nEXPORTED {} time-series files.'.format(len(report['exported'])))
    for item in report['unavailable_or_failed']:
        print('NOTE: {}: {}'.format(item['file'],item['reason']))
    print('\nDONE. Upload this ZIP even if some optional fields are absent:')
    print(result)
    print('Size: {:.2f} MB'.format(result.stat().st_size/1024.0**2))


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,csv.Error,UnicodeError) as exc:raise SystemExit('ERROR: {}'.format(exc))
