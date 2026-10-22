#!/usr/bin/env python
"""Plot temporal ECM-stress diagnostics and baseline/ramp/overload PV loops."""
from __future__ import print_function
import os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

frame=pd.read_csv(sys.argv[1]); out=sys.argv[2]
if not os.path.isdir(out): os.makedirs(out)
t=frame['time']
series=[('aorta_resistance',),('pressure_ventricle','volume_ventricle'),
        ('ecm_cycle_averaged_mean','ecm_homeostatic_mean'),
        ('ecm_signed_deviation_mean','ecm_positive_excess_mean'),
        ('ecm_cycle_averaged_mean','ecm_cycle_averaged_total_fiber_stress_mean')]
fig,axes=plt.subplots(len(series),1,figsize=(10,14),sharex=True)
for ax,names in zip(axes,series):
    for name in names:
        if name in frame: ax.plot(t,frame[name],label=name)
    ax.axvspan(20,30,color='orange',alpha=.15); ax.legend(loc='best'); ax.grid(True)
axes[-1].set_xlabel('time (s)'); fig.tight_layout(); fig.savefig(os.path.join(out,'ecm_time_histories.png'),dpi=180)
fig,ax=plt.subplots()
for label,lo,hi in [('baseline',15,20),('ramp',20,30),('overload',55,60)]:
    q=frame[(t>=lo)&(t<hi)]; ax.plot(q['volume_ventricle'],q['pressure_ventricle'],label=label)
ax.set_xlabel('LV volume'); ax.set_ylabel('LV pressure'); ax.legend(); ax.grid(True)
fig.tight_layout(); fig.savefig(os.path.join(out,'pv_comparison.png'),dpi=180)
