#!/usr/bin/env python3
"""Read-only phase-matched displacement/active-stress analysis for LCC.

Requires only NumPy and h5py; Matplotlib is optional. XDMF fields are selected
by Attribute Name, so different fields may legitimately share timestamps.
"""
import argparse, csv, json, os, sys
import xml.etree.ElementTree as ET
import numpy as np

CYCLES=((37.484,38.316),(38.316,39.147),(39.147,39.978))
PHASES=(.10,.30,.50,.70)
HOTSPOTS=(2497,4844,2833,1580)

def coordinate_lookup(coordinate_path, id_path=None):
    """Read an explicit saved ID-to-reference-coordinate map."""
    if id_path is None:
        id_path = os.path.join(os.path.dirname(coordinate_path), 'quadrature_global_dof.npy')
    ids = np.asarray(np.load(id_path)).reshape(-1)
    coordinates = np.asarray(np.load(coordinate_path), float)
    if coordinates.shape != (len(ids), 3) or not np.isfinite(coordinates).all():
        raise ValueError('Invalid quadrature coordinate shape/values')
    if len(set(ids)) != len(ids) or not np.isfinite(ids).all() or np.any(ids != ids.astype(np.int64)):
        raise ValueError('Invalid or duplicated quadrature IDs')
    return dict((int(gid), coordinates[i]) for i, gid in enumerate(ids))

def descendants(e,name): return [x for x in e.iter() if x.tag.split('}')[-1]==name]
def children(e,name): return [x for x in list(e) if x.tag.split('}')[-1]==name]

def read_item(item,xdmf,handles):
    import h5py
    nested=descendants(item,'DataItem')
    hdf=[x for x in nested if x.get('Format','').upper()=='HDF']
    if hdf: item=hdf[-1]
    text=(item.text or '').strip()
    if ':' not in text: raise ValueError('unsupported non-HDF DataItem %r'%text)
    filename,dataset=text.split(':',1)
    if not os.path.isabs(filename): filename=os.path.join(os.path.dirname(xdmf),filename)
    filename=os.path.abspath(filename)
    if filename not in handles: handles[filename]=h5py.File(filename,'r')
    return np.asarray(handles[filename][dataset])

def read_xdmf(path,field_names,time_window=None):
    """Read the shared mesh and only requested Attribute records."""
    root=ET.parse(path).getroot(); handles={}; found={name:[] for name in field_names}
    metadata={}; geometry=topology=None; topology_type=''; skipped={}
    try:
        grids=[g for g in descendants(root,'Grid') if g.get('GridType','Uniform')=='Uniform']
        for grid in grids:
            gs=children(grid,'Geometry'); ts=children(grid,'Topology')
            if gs and descendants(gs[0],'DataItem'): geometry=descendants(gs[0],'DataItem')[-1]
            if ts and descendants(ts[0],'DataItem'):
                topology=descendants(ts[0],'DataItem')[-1]; topology_type=ts[0].get('TopologyType','')
            times=children(grid,'Time')
            if not times: continue
            time=float(times[0].get('Value'))
            if time_window is not None and not time_window[0] <= time <= time_window[1]:
                continue
            for attr in children(grid,'Attribute'):
                name=attr.get('Name','')
                if name not in found:
                    skipped[name]=skipped.get(name,0)+1; continue
                items=descendants(attr,'DataItem')
                if not items: raise ValueError("Attribute %s has no DataItem"%name)
                found[name].append((time,np.asarray(read_item(items[0],path,handles),float)))
                current={'center':attr.get('Center','Node'),'attribute_type':attr.get('AttributeType','')}
                if name in metadata and metadata[name]!=current:
                    raise ValueError('XDMF metadata changes for field %s'%name)
                metadata[name]=current
        if geometry is None or topology is None:
            gs=descendants(root,'Geometry'); ts=descendants(root,'Topology')
            if not gs or not ts: raise ValueError('XDMF contains no mesh')
            geometry=descendants(gs[0],'DataItem')[-1]; topology=descendants(ts[0],'DataItem')[-1]
            topology_type=ts[0].get('TopologyType','')
        points=np.asarray(read_item(geometry,path,handles),float)
        cells=np.asarray(read_item(topology,path,handles),np.int64)
    finally:
        for h in handles.values(): h.close()
    duplicate_counts={}
    for name,records in found.items():
        if not records: raise ValueError("no XDMF Attribute Name='%s'; other fields: %s"%(name,sorted(skipped)))
        unique={}; duplicates=0
        for time,values in records:
            if time in unique:
                duplicates+=1
                if unique[time].shape!=values.shape or not np.array_equal(unique[time],values):
                    raise ValueError('conflicting %s records at time %s'%(name,time))
            else: unique[time]=values
        found[name]=sorted(unique.items()); duplicate_counts[name]=duplicates
    return points,cells,topology_type,found,metadata,skipped,duplicate_counts

def normalize_displacement(a,n):
    a=np.asarray(a,float)
    if a.ndim==1 and a.size==3*n: a=a.reshape((n,3))
    if a.ndim==2 and a.shape==(3,n): a=a.T
    if a.shape!=(n,3): raise ValueError('displacement shape %s != (%d,3)'%(a.shape,n))
    return a

def normalize_field(a,center,npoints,ncells):
    a=np.asarray(a,float)
    expected=npoints if center.lower()=='node' else ncells
    if a.ndim==1: a=a.reshape((-1,1))
    if a.ndim==2 and a.shape[0]!=expected and a.shape[1]==expected: a=a.T
    if a.ndim!=2 or a.shape[0]!=expected:
        raise ValueError('field shape %s incompatible with %s count %d'%(a.shape,center,expected))
    representation='scalar' if a.shape[1]==1 else 'vector_%d'%a.shape[1]
    return a,representation

def nearest(records,target,cycle):
    allowed=[(i,r) for i,r in enumerate(records) if cycle[0]<=r[0]<=cycle[1]]
    if not allowed: raise ValueError('no snapshots inside cycle')
    i,record=min(allowed,key=lambda x:abs(x[1][0]-target))
    return {'value':record[1].copy(),'time':float(record[0]),'requested_time':target,
            'phase_error':float(record[0]-target),'source_times':[float(record[0])],
            'weights':[1.0],'indices':[i]}

def interpolate(records,target,cycle):
    inside=[(i,r) for i,r in enumerate(records) if cycle[0]<=r[0]<=cycle[1]]
    exact=[(i,r) for i,r in inside if abs(r[0]-target)<=1e-12]
    if exact:
        i,r=exact[0]; return {'value':r[1].copy(),'time':target,'requested_time':target,
                              'phase_error':0.0,'source_times':[float(r[0])],
                              'weights':[1.0],'indices':[i]}
    lower=[(i,r) for i,r in inside if r[0]<target]; upper=[(i,r) for i,r in inside if r[0]>target]
    if not lower or not upper:
        raise ValueError('cannot bracket %.9f inside cycle %.9f:%.9f without extrapolation'%
                         (target,cycle[0],cycle[1]))
    i0,r0=max(lower,key=lambda x:x[1][0]); i1,r1=min(upper,key=lambda x:x[1][0])
    if not (cycle[0]<=r0[0]<target<r1[0]<=cycle[1]):
        raise ValueError('interpolation would cross a cycle boundary')
    w1=(target-r0[0])/(r1[0]-r0[0]); w0=1.0-w1
    return {'value':w0*r0[1]+w1*r1[1],'time':target,'requested_time':target,
            'phase_error':0.0,'source_times':[float(r0[0]),float(r1[0])],
            'weights':[float(w0),float(w1)],'indices':[i0,i1]}

def rigid(a,b):
    ac=a.mean(0); bc=b.mean(0); left,unused,right=np.linalg.svd(np.dot((a-ac).T,b-bc))
    rotation=np.dot(left,right)
    if np.linalg.det(rotation)<0: left[:,-1]*=-1; rotation=np.dot(left,right)
    return rotation,bc-np.dot(ac,rotation)

def containing_tetra(point,points,cells,tol):
    for i,cell in enumerate(cells):
        x=points[np.asarray(cell[:4],int)]; m=np.column_stack((x[1]-x[0],x[2]-x[0],x[3]-x[0]))
        try: tail=np.linalg.solve(m,point-x[0])
        except np.linalg.LinAlgError: continue
        bary=np.concatenate(([1-tail.sum()],tail))
        if np.all(bary>=-tol) and np.all(bary<=1+tol): return int(i),bary
    return None,None

def quadratic_derivatives(b):
    dl=np.asarray(((-1.,-1.,-1.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)))
    result=[(4*b[i]-1)*dl[i] for i in range(4)]
    for i,j in ((0,1),(1,2),(2,0),(0,3),(1,3),(2,3)):
        result.append(4*(dl[i]*b[j]+dl[j]*b[i]))
    return np.asarray(result)
def shape_weights(cell,b):
    if len(cell)>=10:
        l0,l1,l2,l3=b
        return np.asarray((l0*(2*l0-1),l1*(2*l1-1),l2*(2*l2-1),l3*(2*l3-1),
                           4*l0*l1,4*l1*l2,4*l2*l0,4*l0*l3,4*l1*l3,4*l2*l3))
    return b
def sample_node_field(values,cell,b): return np.dot(shape_weights(cell,b),values[np.asarray(cell[:len(shape_weights(cell,b))],int)])
def deformation_gradient(points,u,cell,b):
    ids=np.asarray(cell,int)
    if len(ids)>=10:
        ids=ids[:10]; d=quadratic_derivatives(b); dx=np.dot(points[ids].T,d); du=np.dot(u[ids].T,d)
        return np.eye(3)+np.dot(du,np.linalg.inv(dx)),'quadratic_tetra10'
    ids=ids[:4]; x=points[ids]; v=u[ids]
    dx=np.column_stack((x[1]-x[0],x[2]-x[0],x[3]-x[0])); du=np.column_stack((v[1]-v[0],v[2]-v[0],v[3]-v[0]))
    return np.eye(3)+np.dot(du,np.linalg.inv(dx)),'linear_tetra4_visualization_approximation'

def metrics(a,b):
    d=b-a; point=np.linalg.norm(d,axis=1) if d.ndim==2 else np.abs(d)
    return {'absolute_l2':float(np.linalg.norm(d)),'relative_l2':float(np.linalg.norm(d)/max(np.linalg.norm(a),np.finfo(float).eps)),
            'mean_point_difference':float(point.mean()),'maximum_point_difference':float(point.max())}
def scalar_difference(a,b):
    d=b-a
    return np.abs(d[:,0]) if d.shape[1]==1 else np.linalg.norm(d,axis=1)
def safe_correlation(a,b):
    if len(a)!=len(b) or len(a)<2 or np.std(a)==0 or np.std(b)==0: return None
    return float(np.corrcoef(a,b)[0,1])
def top_decile_overlap(a,b):
    if len(a)!=len(b) or not len(a): return None
    count=max(1,int(np.ceil(.1*len(a))))
    first=set(np.argsort(a)[-count:].tolist()); second=set(np.argsort(b)[-count:].tolist())
    return float(len(first.intersection(second))/count)
def displacement_scale_by_center(u1,u2,cells,center):
    d=np.linalg.norm(u2-u1,axis=1)
    if center.lower()=='node': return d
    return np.asarray([d[np.asarray(cell[:4],int)].mean() for cell in cells])

def comparison(points,cells,hotspots,u1,u2,a1,a2,active_meta):
    x1=points+u1; x2=points+u2; rotation,translation=rigid(x1,x2)
    residual=x2-(np.dot(x1,rotation)+translation); raw_norm=np.linalg.norm(u2-u1)
    active_delta=scalar_difference(a1,a2)
    displacement_delta=displacement_scale_by_center(u1,u2,cells,active_meta['center'])
    item={'displacement':metrics(u1,u2),'best_fit_rigid':{'rotation':rotation.tolist(),'translation':translation.tolist(),
          'residual_l2':float(np.linalg.norm(residual)),'residual_maximum':float(np.linalg.norm(residual,axis=1).max()),
          'residual_fraction_of_raw_difference':float(np.linalg.norm(residual)/max(raw_norm,np.finfo(float).eps))},
          'active_stress':metrics(a1,a2),'active_displacement_difference_correlation':safe_correlation(active_delta,displacement_delta),
          'active_displacement_top_decile_overlap_fraction':top_decile_overlap(active_delta,displacement_delta),
          'hotspots':{}}
    for gid,meta in hotspots.items():
        entry=dict(meta); ci=meta['cell']
        if ci is not None:
            b=np.asarray(meta['barycentric']); cell=cells[ci]
            f1,method=deformation_gradient(points,u1,cell,b); f2,unused=deformation_gradient(points,u2,cell,b)
            lu1=sample_node_field(u1,cell,b); lu2=sample_node_field(u2,cell,b)
            if active_meta['center'].lower()=='node': la1=sample_node_field(a1,cell,b); la2=sample_node_field(a2,cell,b)
            else: la1=a1[ci]; la2=a2[ci]
            coordinate=np.asarray(meta['coordinate']); aligned=np.dot(coordinate+lu1,rotation)+translation
            entry.update({'deformation_gradient_method':method,'F_difference':float(np.linalg.norm(f2-f1)),
                          'J_first':float(np.linalg.det(f1)),'J_second':float(np.linalg.det(f2)),
                          'displacement_difference':float(np.linalg.norm(lu2-lu1)),
                          'rigid_aligned_position_difference':float(np.linalg.norm((coordinate+lu2)-aligned)),
                          'active_stress_first':np.asarray(la1).tolist(),'active_stress_second':np.asarray(la2).tolist(),
                          'active_stress_difference':float(np.linalg.norm(la2-la1))})
        item['hotspots'][str(gid)]=entry
    return item

def analyze(args):
    if os.path.exists(args.output): raise IOError('refusing to overwrite %s'%args.output)
    os.makedirs(args.output)
    window = (min(c[0] for c in args.cycles), max(c[1] for c in args.cycles))
    points,cells,topology,fields,metadata,skipped,duplicates=read_xdmf(args.xdmf,('displacement','active_stress'),window)
    displacement=[(t,normalize_displacement(v,len(points))) for t,v in fields['displacement']]
    active=[]; representation=None
    for t,v in fields['active_stress']:
        value,current=normalize_field(v,metadata['active_stress']['center'],len(points),len(cells))
        if representation not in (None,current): raise ValueError('active_stress representation changes')
        representation=current; active.append((t,value))
    if not np.isfinite(points).all() or not all(np.isfinite(x[1]).all() for x in displacement+active): raise ValueError('nonfinite input')
    q=coordinate_lookup(args.quadrature_coordinates, args.quadrature_global_dofs); hotspots={}
    for gid in args.hotspots:
        if gid not in q:
            raise ValueError('Hotspot ID missing from saved geometry: %d' % gid)
        coordinate=np.asarray(q[gid],float); ci,b=containing_tetra(coordinate,points,cells,args.coordinate_tolerance)
        hotspots[gid]={'coordinate':coordinate.tolist(),'cell':ci,'barycentric':None if b is None else b.tolist()}
    selections=[]; rows=[]; plots=[]
    for phase in args.phases:
        per_cycle=[]
        for number,cyc in enumerate(args.cycles):
            target=cyc[0]+phase
            dn=nearest(displacement,target,cyc); di=interpolate(displacement,target,cyc)
            an=nearest(active,target,cyc); ai=interpolate(active,target,cyc)
            selections.append({'phase':phase,'cycle':number+1,'target_time':target,
              'displacement_nearest_time':dn['time'],'displacement_nearest_phase_error':dn['phase_error'],
              'displacement_interpolation_source_times':di['source_times'],'displacement_interpolation_weights':di['weights'],
              'active_nearest_time':an['time'],'active_nearest_phase_error':an['phase_error'],
              'active_interpolation_source_times':ai['source_times'],'active_interpolation_weights':ai['weights']})
            per_cycle.append((dn,di,an,ai))
        for first in (0,1):
            second=first+1; dn1,di1,an1,ai1=per_cycle[first]; dn2,di2,an2,ai2=per_cycle[second]
            near=comparison(points,cells,hotspots,dn1['value'],dn2['value'],an1['value'],an2['value'],metadata['active_stress'])
            interp=comparison(points,cells,hotspots,di1['value'],di2['value'],ai1['value'],ai2['value'],metadata['active_stress'])
            old=near['displacement']['absolute_l2']; new=interp['displacement']['absolute_l2']
            sensitivity=float((old-new)/old) if old else 0.0
            record={'phase':phase,'cycle_pair':[first+1,second+1],'nearest_snapshot':near,'phase_interpolated':interp,
                    'temporal_alignment_displacement_difference_reduction_fraction':sensitivity}
            rows.append(record)
            if not args.no_plots:
                try:
                    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
                    fig,axes=plt.subplots(1,2,figsize=(12,5));
                    axes[0].scatter(scalar_difference(ai1['value'],ai2['value']),displacement_scale_by_center(di1['value'],di2['value'],cells,metadata['active_stress']['center']),s=4)
                    axes[0].set_xlabel('|active stress difference|'); axes[0].set_ylabel('|displacement difference|')
                    axes[1].bar(['nearest','interpolated'],[near['displacement']['relative_l2'],interp['displacement']['relative_l2']])
                    axes[1].set_ylabel('relative displacement L2')
                    name='phase_%.2f_cycles_%d_%d.png'%(phase,first+1,second+1); fig.tight_layout(); fig.savefig(os.path.join(args.output,name),dpi=180); plt.close(fig); plots.append(name)
                except ImportError: pass
    table=[]
    for r in rows:
        n=r['nearest_snapshot']; i=r['phase_interpolated']
        table.append({'phase':r['phase'],'cycle_pair':'%d-%d'%tuple(r['cycle_pair']),
          'nearest_displacement_relative_l2':n['displacement']['relative_l2'],
          'interpolated_displacement_relative_l2':i['displacement']['relative_l2'],
          'nearest_rigid_residual_fraction':n['best_fit_rigid']['residual_fraction_of_raw_difference'],
          'interpolated_rigid_residual_fraction':i['best_fit_rigid']['residual_fraction_of_raw_difference'],
          'interpolated_active_stress_relative_l2':i['active_stress']['relative_l2'],
          'interpolated_active_displacement_correlation':i['active_displacement_difference_correlation'],
          'interpolated_active_displacement_top_decile_overlap':i['active_displacement_top_decile_overlap_fraction'],
          'alignment_reduction_fraction':r['temporal_alignment_displacement_difference_reduction_fraction']})
    rigid_fractions=[x['interpolated_rigid_residual_fraction'] for x in table]
    correlations=[x['interpolated_active_displacement_correlation'] for x in table
                  if x['interpolated_active_displacement_correlation'] is not None]
    ranked=[
      {'rank':1,'hypothesis':'genuine local FE deformation differences',
       'evidence':'median interpolated rigid-residual/raw ratio = %.6g'%float(np.median(rigid_fractions)),
       'limitation':'visualization sampling and interpolation do not prove instability or recover the original CG2 solution'},
      {'rank':2,'hypothesis':'nonperiodic active-state dynamics associated with deformation differences',
       'evidence':('median active/displacement difference correlation = %.6g'%float(np.median(correlations))
                   if correlations else 'correlation unavailable for the saved representation'),
       'limitation':'correlation at 0.1-second sampling cannot establish causal ordering'},
      {'rank':3,'hypothesis':'field mapping inconsistency',
       'evidence':'fields are selected by XDMF name and hotspots are mapped by immutable reference coordinates',
       'limitation':'visualization files alone cannot audit the original FE Function dofmap'}]
    report={'test':'phase_interpolated_displacement_active_stress','source_files_modified':False,'source_xdmf':os.path.abspath(args.xdmf),
      'cycles':args.cycles,'phases':args.phases,'topology_type':topology,'mesh_point_count':len(points),'cell_count':len(cells),
      'visualization_only_gradient_warning':('linear_tetra4 is not the original CG2 FE gradient; exact CG2 reconstruction requires a checkpoint with Function-space/dofmap data'),
      'active_stress':{'record_count':len(active),'center':metadata['active_stress']['center'],'attribute_type':metadata['active_stress']['attribute_type'],'representation':representation},
      'duplicate_field_records_ignored':duplicates,'skipped_field_record_counts':skipped,
      'all_hotspots_mapped':all(x['cell'] is not None for x in hotspots.values()),
      'interpolation_warning':'linear interpolation of 0.1-second samples cannot resolve rapid transients or establish causal ordering',
      'ranked_interpretation':ranked,'selections':selections,'comparison_table':table,'comparisons':rows,'plots':plots}
    report['passed']=bool(report['all_hotspots_mapped'] and len(active)>0)
    with open(os.path.join(args.output,'displacement_active_cycle_comparison.json'),'w') as h: json.dump(report,h,indent=2,sort_keys=True)
    with open(os.path.join(args.output,'comparison_table.csv'),'w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    with open(os.path.join(args.output,'interpolation_sources.csv'),'w',newline='') as h:
        flat=[]
        for s in selections:
            row=dict(s); row.update({k:json.dumps(v) for k,v in s.items() if isinstance(v,list)}); flat.append(row)
        writer=csv.DictWriter(h,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    hotspot_table=[]
    for record in rows:
        for mode in ('nearest_snapshot','phase_interpolated'):
            for gid,values in record[mode]['hotspots'].items():
                hotspot_table.append({'phase':record['phase'],'cycle_pair':'%d-%d'%tuple(record['cycle_pair']),
                  'mode':mode,'global_quadrature_dof':gid,'cell':values.get('cell'),
                  'displacement_difference':values.get('displacement_difference'),
                  'rigid_aligned_position_difference':values.get('rigid_aligned_position_difference'),
                  'active_stress_difference':values.get('active_stress_difference'),
                  'F_difference':values.get('F_difference'),'J_first':values.get('J_first'),'J_second':values.get('J_second'),
                  'deformation_gradient_method':values.get('deformation_gradient_method')})
    with open(os.path.join(args.output,'hotspot_summary.csv'),'w',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(hotspot_table[0])); writer.writeheader(); writer.writerows(hotspot_table)
    print('Phase-interpolated analysis: %s'%('PASS' if report['passed'] else 'FAIL')); print('Reports: %s'%args.output)
    return 0 if report['passed'] else 1

def cycle(value):
    start,stop=[float(x) for x in value.split(':',1)]
    if stop<=start: raise argparse.ArgumentTypeError('cycle stop must exceed start')
    return start,stop
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--xdmf',required=True); p.add_argument('--quadrature-coordinates',required=True); p.add_argument('--output',required=True)
    p.add_argument('--quadrature-global-dofs', help='default: quadrature_global_dof.npy beside coordinates')
    p.add_argument('--cycle',action='append',type=cycle,dest='cycles'); p.add_argument('--phase',action='append',type=float,dest='phases'); p.add_argument('--hotspot-dof',action='append',type=int,dest='hotspots')
    p.add_argument('--coordinate-tolerance',type=float,default=1e-7); p.add_argument('--no-plots',action='store_true'); args=p.parse_args(argv)
    args.cycles=tuple(args.cycles or CYCLES); args.phases=tuple(args.phases or PHASES); args.hotspots=tuple(args.hotspots or HOTSPOTS)
    if len(args.cycles)!=3: p.error('exactly three cycles required')
    return analyze(args)
if __name__=='__main__': sys.exit(main())
