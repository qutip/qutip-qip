TEXTWIDTH = 7.1398920714
LINEWIDTH = 3.48692403487
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    from quantum_plots import global_setup
    global_setup(fontsize = 10)
except:
    pass

# discrete pulse
import numpy as np
from qutip_qip.device import Processor
from qutip import sigmaz, sigmax, basis
from qutip_qip.pulse import Pulse

processor = Processor(1)
coeff = np.array([1.])
tlist = np.array([0., np.pi/2])
pulse = Pulse(
    sigmax(), targets=0, tlist=tlist,
    coeff=coeff, label="sigmax")
processor.add_pulse(pulse)
plot_time = np.linspace(0, np.pi/2, 20)
solver_result = processor.run_state(
    init_state=basis(2, 0),
    tlist=plot_time)

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#1f77b4"]) 
fig1, ax1 = processor.plot_pulses(figsize=(LINEWIDTH/2, LINEWIDTH/4))
ax1[0].set_xlim(-0.5, np.pi/2 + 0.5)
ax1[0].set_ylim(-0.1, 1.1)
ax1[0].axhline(0)
ax1[0].set_ylabel(None)
fig1.tight_layout()
fig1.savefig("discrete_pulse.pdf")
fig1.show()

# continuous pulse
processor = Processor(1)
processor.pulse_mode = "continuous"
processor.add_control(sigmax())
tlist = np.linspace(0., np.pi/2, 21)
coeff = np.array(np.sin(2*tlist) * np.pi/2)
processor.pulses[0].tlist = tlist
processor.pulses[0].coeff = coeff
solver_result2 = processor.run_state(
    init_state=basis(2, 0),
    tlist=plot_time)

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#ff7f0e"]) 
fig2, ax2 = processor.plot_pulses(figsize=(LINEWIDTH/2, LINEWIDTH/4))
ax2[0].set_xlim(-0.5, np.pi/2 + 0.5)
ax2[0].set_ylim(-0.1, 1.7)
ax2[0].axhline(0)
ax2[0].set_ylabel(None)
fig2.tight_layout()
fig2.savefig("continuous_pulse.pdf")
fig2.show()

# bloch sphere
from qutip import Bloch
fig3 = plt.figure(figsize = (LINEWIDTH, LINEWIDTH))
b = Bloch(fig3)
b.point_color = ["#1f77b4"]

theta = 2 * np.arctan(np.asarray([np.abs(s[1,0]/s[0,0]) for s in solver_result.states]))
phi = np.angle(np.asarray([s[1,0]/s[0,0] for s in solver_result.states]))
xp = np.sin(theta) * np.cos(phi)
yp = np.sin(theta) * np.sin(phi)
zp = np.cos(theta)
pnts = [xp, yp, zp]
b.add_points(pnts)
b.show()
fig3.savefig("bloch_sphere1.pdf")
fig3.show()

from qutip import Bloch
fig4 = plt.figure(figsize = (LINEWIDTH, LINEWIDTH))
b = Bloch(fig4)
b.point_color = ["#ff7f0e"]

theta = 2 * np.arctan(np.asarray([np.abs(s[1,0]/s[0,0]) for s in solver_result2.states]))
phi = np.angle(np.asarray([s[1,0]/s[0,0] for s in solver_result2.states]))
xp = np.sin(theta) * np.cos(phi)
yp = np.sin(theta) * np.sin(phi)
zp = np.cos(theta)
pnts = [xp, yp, zp]
b.add_points(pnts)
b.show()
fig4.savefig("bloch_sphere2.pdf")
fig4.show()

