#!/usr/bin/env python
"""Plot temporal ECM-stress diagnostics and baseline/ramp/overload PV loops."""
from __future__ import print_function
import argparse, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument('csv')
parser.add_argument('output_directory')
parser.add_argument('--baseline-end',type=float,default=20.0)
parser.add_argument('--ramp-end',type=float,default=30.0)
parser.add_argument('--simulation-end',type=float,default=None)
args=parser.parse_args()
frame=pd.read_csv(args.csv); out=args.output_directory
if not os.path.isdir(out): os.makedirs(out)
t=frame['time']
excess_valid=(frame['ecm_excess_valid'] > 0.5) if 'ecm_excess_valid' in frame else (t >= args.baseline_end)
series=[('aorta_resistance',),('heart_rate',),('pressure_ventricle',),
        ('volume_ventricle',),
        ('ecm_cycle_averaged_mean','ecm_homeostatic_mean'),
        ('ecm_signed_deviation_mean','ecm_positive_excess_mean'),
        ('ecm_cycle_averaged_mean','ecm_cycle_averaged_total_fiber_stress_mean'),
        ('ecm_phi_m_mean','ecm_phi_g_mean','ecm_phi_c_mean')]
fig,axes=plt.subplots(len(series),1,figsize=(10,20),sharex=True)
provisional=('ecm_reference_provisional' in frame and
             frame['ecm_reference_provisional'].max() > 0.5)
fig.suptitle('Continuous ECM diagnostic (%s reference)' %
             ('provisional' if provisional else 'validated'))
for ax,names in zip(axes,series):
    for name in names:
        if name in frame:
            values=frame[name]
            if name in ('ecm_homeostatic_mean','ecm_signed_deviation_mean',
                        'ecm_positive_excess_mean'):
                values=values.where(excess_valid)
            ax.plot(t,values,label=name)
    ax.axvspan(args.baseline_end,args.ramp_end,color='orange',alpha=.15)
    ax.axvline(args.baseline_end,color='black',linestyle='--',linewidth=.8)
    ax.legend(loc='best'); ax.grid(True)
axes[-1].set_xlabel('time (s)'); fig.tight_layout(); fig.savefig(os.path.join(out,'ecm_time_histories.png'),dpi=180)
fig,ax=plt.subplots()
end=args.simulation_end if args.simulation_end is not None else float(t.max())
baseline_width=min(5.0,args.baseline_end)
overload_width=min(5.0,max(0.0,end-args.ramp_end))
for label,lo,hi in [('baseline',args.baseline_end-baseline_width,args.baseline_end),
                    ('ramp',args.baseline_end,args.ramp_end),
                    ('overload',end-overload_width,end)]:
    q=frame[(t>=lo)&(t<hi)]; ax.plot(q['volume_ventricle'],q['pressure_ventricle'],label=label)
ax.set_xlabel('LV volume'); ax.set_ylabel('LV pressure'); ax.legend(); ax.grid(True)
fig.tight_layout(); fig.savefig(os.path.join(out,'pv_comparison.png'),dpi=180)
