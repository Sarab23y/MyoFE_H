#!/usr/bin/env python
"""Read-only Python-2.7 analysis of displacement across reference cycles.

Only XDMF Attributes named ``displacement`` are selected. Other fields may
legitimately occur at the same timestamp and are ignored rather than mistaken
for duplicate displacement snapshots.
"""
from __future__ import division, print_function
import argparse, csv, json, os, sys
import xml.etree.ElementTree as ET
import numpy as np

CYCLES=((37.484,38.316),(38.316,39.147),(39.147,39.978))
PHASES=(.10,.30,.50,.70)
HOTSPOTS=(1643,1694,2404,4751,4844,5004)

def descendants(e,name):
    return [x for x in e.iter() if x.tag.split('}')[-1]==name]
def children(e,name):
    return [x for x in list(e) if x.tag.split('}')[-1]==name]

def read_item(item,xdmf,handles):
    import h5py
    nested=descendants(item,'DataItem')
    if nested:
        hdf=[x for x in nested if x.get('Format','').upper()=='HDF']
        if len(hdf)!=1: raise ValueError('unsupported nested XDMF DataItem')
        item=hdf[0]
    text=(item.text or '').strip()
    if ':' not in text: raise ValueError('unsupported non-HDF DataItem %r'%text)
    filename,dataset=text.split(':',1)
    if not os.path.isabs(filename): filename=os.path.join(os.path.dirname(xdmf),filename)
    filename=os.path.abspath(filename)
    if filename not in handles: handles[filename]=h5py.File(filename,'r')
    return np.asarray(handles[filename][dataset])

def read_xdmf(path,field):
    root=ET.parse(path).getroot(); handles={}; records=[]; skipped={}
    geometry=topology=None; topology_type=''
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
            for attr in children(grid,'Attribute'):
                name=attr.get('Name','')
                if name!=field:
                    skipped[name]=skipped.get(name,0)+1; continue
                items=descendants(attr,'DataItem')
                if not items: raise ValueError('displacement Attribute has no DataItem')
                records.append((time,np.asarray(read_item(items[0],path,handles),float)))
        if geometry is None or topology is None:
            gs=descendants(root,'Geometry'); ts=descendants(root,'Topology')
            if not gs or not ts: raise ValueError('XDMF contains no mesh')
            geometry=descendants(gs[0],'DataItem')[-1]
            topology=descendants(ts[0],'DataItem')[-1]
            topology_type=ts[0].get('TopologyType','')
        points=np.asarray(read_item(geometry,path,handles),float)
        cells=np.asarray(read_item(topology,path,handles),np.int64)
    finally:
        for h in handles.values(): h.close()
    if not records: raise ValueError("no XDMF Attribute Name='%s'; other fields: %s"%(field,sorted(skipped)))
    unique={}; duplicates=0
    for time,values in records:
        if time in unique:
            duplicates+=1
            if unique[time].shape!=values.shape or not np.array_equal(unique[time],values):
                raise ValueError('conflicting displacement records at %s'%time)
        else: unique[time]=values
    return points,cells,topology_type,sorted(unique.items()),skipped,duplicates

def normalize_u(values,n):
    a=np.asarray(values,float)
    if a.ndim==1 and a.size==3*n: a=a.reshape((n,3))
    if a.shape!=(n,3): raise ValueError('displacement shape %s != (%d,3)'%(a.shape,n))
    return a

def rigid(a,b):
    ac=a.mean(0); bc=b.mean(0); left,unused,right=np.linalg.svd(np.dot((a-ac).T,b-bc))
    rotation=np.dot(left,right)
    if np.linalg.det(rotation)<0: left[:,-1]*=-1; rotation=np.dot(left,right)
    return rotation,bc-np.dot(ac,rotation)

def containing_tetra(point,points,cells,tol):
    for i,cell in enumerate(cells):
        x=points[np.asarray(cell[:4],int)]
        m=np.column_stack((x[1]-x[0],x[2]-x[0],x[3]-x[0]))
        try: tail=np.linalg.solve(m,point-x[0])
        except np.linalg.LinAlgError: continue
        bary=np.concatenate(([1-tail.sum()],tail))
        if np.all(bary>=-tol) and np.all(bary<=1+tol): return int(i),bary
    return None,None

def quadratic_derivatives(b):
    dl=np.asarray(((-1.,-1.,-1.),(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)))
    result=[(4*b[i]-1)*dl[i] for i in range(4)]
    # XDMF tetra10 convention: 01,12,20,03,13,23.
    for i,j in ((0,1),(1,2),(2,0),(0,3),(1,3),(2,3)):
        result.append(4*(dl[i]*b[j]+dl[j]*b[i]))
    return np.asarray(result)

def interpolate_displacement(u,cell,bary):
    ids=np.asarray(cell,int)
    if len(ids)>=10:
        l0,l1,l2,l3=bary
        weights=[l0*(2*l0-1),l1*(2*l1-1),l2*(2*l2-1),l3*(2*l3-1),
                 4*l0*l1,4*l1*l2,4*l2*l0,4*l0*l3,4*l1*l3,4*l2*l3]
        return np.dot(np.asarray(weights),u[ids[:10]])
    return np.dot(bary,u[ids[:4]])

def deformation_gradient(points,u,cell,bary):
    ids=np.asarray(cell,int)
    if len(ids)>=10:
        ids=ids[:10]; d=quadratic_derivatives(bary)
        dx=np.dot(points[ids].T,d); du=np.dot(u[ids].T,d)
        return np.eye(3)+np.dot(du,np.linalg.inv(dx)),'quadratic_tetra10'
    ids=ids[:4]; x=points[ids]; v=u[ids]
    dx=np.column_stack((x[1]-x[0],x[2]-x[0],x[3]-x[0]))
    du=np.column_stack((v[1]-v[0],v[2]-v[0],v[3]-v[0]))
    return np.eye(3)+np.dot(du,np.linalg.inv(dx)),'linear_tetra4'

def metrics(a,b):
    d=b-a; mag=np.linalg.norm(d,axis=1)
    return {'absolute_l2':float(np.linalg.norm(d)),
            'relative_l2':float(np.linalg.norm(d)/max(np.linalg.norm(a),np.finfo(float).eps)),
            'mean_point_difference':float(mag.mean()),'maximum_point_difference':float(mag.max())}

def analyze(args):
    if os.path.exists(args.output): raise IOError('refusing to overwrite %s'%args.output)
    os.makedirs(args.output)
    points,cells,topology,records,skipped,duplicates=read_xdmf(args.xdmf,args.field_name)
    times=np.asarray([x[0] for x in records]); us=[normalize_u(x[1],len(points)) for x in records]
    if points.shape[1]!=3 or not np.isfinite(points).all() or not all(np.isfinite(u).all() for u in us):
        raise ValueError('mesh/displacement is not finite 3-D data')
    coordinates=np.load(args.quadrature_coordinates)
    if coordinates.ndim!=2 or coordinates.shape[1]<3: raise ValueError('quadrature_dof.npy must be (N,>=3)')
    hotspots={}
    for gid in args.hotspots:
        if gid<0 or gid>=len(coordinates): raise ValueError('hotspot %d out of range'%gid)
        coordinate=np.asarray(coordinates[gid,:3],float)
        cell,bary=containing_tetra(coordinate,points,cells,args.coordinate_tolerance)
        hotspots[gid]={'coordinate':coordinate.tolist(),'cell':cell,
                       'barycentric':None if bary is None else bary.tolist()}
    selections=[]; comparisons=[]; plots=[]
    for phase in args.phases:
        selected=[]
        for number,cycle in enumerate(args.cycles):
            wanted=cycle[0]+phase; index=int(np.argmin(np.abs(times-wanted))); actual=float(times[index])
            selections.append({'phase':phase,'cycle':number+1,'requested_time':wanted,
                               'snapshot_time':actual,'phase_error':actual-wanted,
                               'within_cycle':bool(cycle[0]<=actual<=cycle[1])})
            selected.append((actual,us[index]))
        base=len(selections)-3
        for first in (0,1):
            second=first+1; t1,u1=selected[first]; t2,u2=selected[second]
            x1=points+u1; x2=points+u2; rotation,translation=rigid(x1,x2)
            residual=x2-(np.dot(x1,rotation)+translation)
            item={'phase':phase,'cycle_pair':[first+1,second+1],'snapshot_times':[t1,t2],
                  'phase_errors':[selections[base+first]['phase_error'],selections[base+second]['phase_error']],
                  'displacement':metrics(u1,u2),'best_fit_rigid':{
                      'rotation':rotation.tolist(),'translation':translation.tolist(),
                      'residual_l2':float(np.linalg.norm(residual)),
                      'residual_maximum':float(np.linalg.norm(residual,axis=1).max())},'hotspots':{}}
            methods=set()
            for gid,meta in hotspots.items():
                entry=dict(meta); cell=meta['cell']
                if cell is not None:
                    bary=np.asarray(meta['barycentric']); f1,method=deformation_gradient(points,u1,cells[cell],bary)
                    f2,unused=deformation_gradient(points,u2,cells[cell],bary); methods.add(method)
                    local_u1=interpolate_displacement(u1,cells[cell],bary)
                    local_u2=interpolate_displacement(u2,cells[cell],bary)
                    coordinate=np.asarray(meta['coordinate'])
                    deformed1=coordinate+local_u1; deformed2=coordinate+local_u2
                    aligned1=np.dot(deformed1,rotation)+translation
                    entry.update({'deformation_gradient_method':method,'F_first':f1.tolist(),'F_second':f2.tolist(),
                                  'F_difference':float(np.linalg.norm(f2-f1)),
                                  'J_first':float(np.linalg.det(f1)),'J_second':float(np.linalg.det(f2)),
                                  'displacement_first':local_u1.tolist(),
                                  'displacement_second':local_u2.tolist(),
                                  'displacement_difference':float(np.linalg.norm(local_u2-local_u1)),
                                  'rigid_aligned_deformed_position_difference':float(np.linalg.norm(deformed2-aligned1))})
                item['hotspots'][str(gid)]=entry
            item['deformation_gradient_methods']=sorted(methods); comparisons.append(item)
            if not args.no_plots:
                try:
                    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
                    fig=plt.figure(figsize=(10,7)); ax=fig.add_subplot(111,projection='3d')
                    image=ax.scatter(points[:,0],points[:,1],points[:,2],c=np.linalg.norm(u2-u1,axis=1),s=5)
                    fig.colorbar(image,ax=ax,label='|u(second)-u(first)|')
                    name='displacement_phase_%.2f_cycles_%d_%d.png'%(phase,first+1,second+1)
                    fig.savefig(os.path.join(args.output,name),dpi=180); plt.close(fig); plots.append(name)
                except ImportError: pass
    allowed=args.save_frequency/2+args.time_tolerance
    phase_ok=all(abs(r['phase_error'])<=allowed and r['within_cycle'] for r in selections)
    mapped=all(x['cell'] is not None for x in hotspots.values())
    report={'test':'existing_cycle_displacement_comparison','source_xdmf':os.path.abspath(args.xdmf),
            'source_files_modified':False,'selected_attribute_name':args.field_name,
            'skipped_xdmf_field_record_counts':skipped,
            'identical_duplicate_displacement_records_ignored':duplicates,'topology_type':topology,
            'mesh_point_count':len(points),'cell_count':len(cells),'displacement_snapshot_count':len(times),
            'snapshot_time_range':[float(times.min()),float(times.max())],
            'maximum_allowed_phase_error':allowed,'phase_selection_passed':phase_ok,
            'all_hotspots_mapped':mapped,'passed':bool(phase_ok and mapped),
            'pass_scope':'read-only field selection, phase alignment and hotspot mapping; mechanics are measurements',
            'selected_snapshots':selections,'comparisons':comparisons,'plots':plots}
    path=os.path.join(args.output,'displacement_cycle_comparison.json')
    with open(path,'w') as h: json.dump(report,h,indent=2,sort_keys=True)
    with open(os.path.join(args.output,'selected_snapshots.csv'),'wb') as h:
        columns=('phase','cycle','requested_time','snapshot_time','phase_error','within_cycle')
        writer=csv.DictWriter(h,fieldnames=columns); writer.writeheader(); writer.writerows(selections)
    print('Displacement cycle comparison: %s'%('PASS' if report['passed'] else 'FAIL')); print('Report: %s'%path)
    return 0 if report['passed'] else 1

def cycle(value):
    start,stop=[float(x) for x in value.split(':',1)]
    if stop<=start: raise argparse.ArgumentTypeError('cycle stop must exceed start')
    return start,stop

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--xdmf',required=True); p.add_argument('--quadrature-coordinates',required=True)
    p.add_argument('--output',required=True); p.add_argument('--field-name',default='displacement')
    p.add_argument('--cycle',action='append',type=cycle,dest='cycles'); p.add_argument('--phase',action='append',type=float,dest='phases')
    p.add_argument('--hotspot-dof',action='append',type=int,dest='hotspots'); p.add_argument('--save-frequency',type=float,default=.1)
    p.add_argument('--time-tolerance',type=float,default=1e-8); p.add_argument('--coordinate-tolerance',type=float,default=1e-7)
    p.add_argument('--no-plots',action='store_true'); args=p.parse_args(argv)
    args.cycles=tuple(args.cycles or CYCLES); args.phases=tuple(args.phases or PHASES); args.hotspots=tuple(args.hotspots or HOTSPOTS)
    if len(args.cycles)!=3: p.error('exactly three cycles required')
    return analyze(args)
if __name__=='__main__': sys.exit(main())
