import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation

# =====================================================
# MODEL PARAMETERS (TUNED TO CROSS Nc CLEANLY)
# =====================================================

M = 1.0
K_bridge = 1.0
Omega = np.sqrt(K_bridge / M)

B = 0.4          # damping
G = 0.03         # forcing per pedestrian
C = 0.08         # coupling strength
alpha = np.pi/2
sigma = 0.2      # frequency spread

# Density of frequency distribution at Omega
P_Omega = 1 / (np.sqrt(2*np.pi) * sigma)

# Theoretical critical crowd size
Nc = (2 * B * Omega) / (np.pi * G * C * P_Omega)
print("Theoretical Nc =", Nc)

# =====================================================
# CROWD RAMP SETTINGS
# =====================================================

N0 = 5
dN = 5
DeltaT = 20
Nmax = 120      # must be > Nc

T_total = 400
dt = 0.05

# =====================================================
# INITIAL CONDITIONS
# =====================================================

X0 = 1e-4
V0 = 0.0

theta = np.random.uniform(0, 2*np.pi, N0)
Omega_i = np.random.normal(Omega, sigma, N0)

state = np.concatenate(([X0, V0], theta))
current_N = N0

# =====================================================
# STORAGE
# =====================================================

time_all = []
N_all = []
A_all = []
R_all = []
X_all = []

# =====================================================
# SYSTEM OF EQUATIONS
# =====================================================

def system(t, y, Omega_i):
    X = y[0]
    V = y[1]
    theta = y[2:]

    # Bridge dynamics
    dXdt = V
    forcing = np.sum(G * np.sin(theta))
    dVdt = (1/M) * (-B*V - K_bridge*X + forcing)

    # Amplitude and phase
    A = np.sqrt(X**2 + (V/Omega)**2)
    Psi = np.arctan2(X, V/Omega)

    # Pedestrian dynamics
    dthetadt = Omega_i + C * A * np.sin(Psi - theta + alpha)

    return np.concatenate(([dXdt, dVdt], dthetadt))

# =====================================================
# CROWD RAMP SIMULATION
# =====================================================

t_current = 0

while t_current < T_total:

    t_span = (t_current, t_current + DeltaT)
    t_eval = np.arange(t_current, t_current + DeltaT, dt)

    sol = solve_ivp(
        lambda t, y: system(t, y, Omega_i),
        t_span,
        state,
        t_eval=t_eval,
        method='RK45'
    )
    theta_history = []
    for i in range(len(sol.t)):
        X = sol.y[0, i]
        V = sol.y[1, i]
        theta_vals = sol.y[2:, i]
        theta_history.append(theta_vals.copy())
        X_all.append(X)

        A = np.sqrt(X**2 + (V/Omega)**2)
        R = np.abs(np.mean(np.exp(1j * theta_vals)))

        time_all.append(sol.t[i])
        A_all.append(A)
        R_all.append(R)
        N_all.append(current_N)

    state = sol.y[:, -1]
    t_current += DeltaT

    # Add pedestrians if below Nmax
    if current_N < Nmax:
        new_theta = np.random.uniform(0, 2*np.pi, dN)
        new_Omega = np.random.normal(Omega, sigma, dN)

        theta = np.concatenate((state[2:], new_theta))
        Omega_i = np.concatenate((Omega_i, new_Omega))

        current_N += dN
        state = np.concatenate((state[:2], theta))

X_all = np.array(X_all)

theta_history = np.array(theta_history)
# =====================================================
# CONVERT TO ARRAYS
# =====================================================

time_all = np.array(time_all)
A_all = np.array(A_all)
R_all = np.array(R_all)
N_all = np.array(N_all)

# =====================================================
# FIND CRITICAL TIME SAFELY
# =====================================================

indices = np.where(N_all >= Nc)[0]

if len(indices) > 0:
    tc = time_all[indices[0]]
else:
    tc = None

# =====================================================
# PLOTS (REQUIRED STACKED FORMAT)
# =====================================================

fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# N(t)
axs[0].step(time_all, N_all, where='post')
if tc is not None:
    axs[0].axvline(tc, linestyle='--', label='N = Nc')
    axs[0].legend()
axs[0].set_ylabel("N(t)")
axs[0].set_title("Crowd Ramp Experiment")

# A(t)
axs[1].plot(time_all, A_all)
if tc is not None:
    axs[1].axvline(tc, linestyle='--', label='N = Nc')
    axs[1].legend()
axs[1].set_ylabel("Bridge Amplitude A(t)")

# R(t)
axs[2].plot(time_all, R_all)
if tc is not None:
    axs[2].axvline(tc, linestyle='--', label='N = Nc')
    axs[2].legend()
axs[2].set_ylabel("Synchronization R(t)")
axs[2].set_xlabel("Time")

plt.tight_layout()
plt.show()


def millennium_bridge_gif(
    time_all,
    theta_history,
    X_history,
    filename="millennium_bridge.gif",
    fps=20
):
    """
    Create schematic animation of bridge motion + pedestrian synchronization.
    """

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.set_xlim(-5, 5)
    ax.set_ylim(-2, 2)
    ax.set_title("Millennium Bridge Synchronization")
    ax.set_xlabel("Lateral displacement")
    ax.set_yticks([])

    # Bridge line
    bridge_line, = ax.plot([], [], 'k-', linewidth=4)

    # Pedestrians
    scatter = ax.scatter([], [], s=20)

    N_max = theta_history.shape[1]

    def update(frame):

        X = X_history[frame]
        theta = theta_history[frame]

        # Bridge position
        bridge_x = np.linspace(-4, 4, 200) + X
        bridge_y = np.zeros_like(bridge_x)

        bridge_line.set_data(bridge_x, bridge_y)

        # Pedestrian positions
        x_positions = np.linspace(-3.5, 3.5, len(theta)) + X

        # Encode phase in vertical displacement
        y_positions = 0.5 * np.sin(theta)

        scatter.set_offsets(np.column_stack((x_positions, y_positions)))

        return bridge_line, scatter

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(theta_history),
        interval=50,
        blit=True
    )

    ani.save(filename, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)

    print(f"Millennium Bridge GIF saved as {filename}")
    
def millennium_bridge_snapshots(
    time_all,
    theta_history,
    X_history,
    snapshot_times,
    filename="millennium_bridge_snapshots.png"
):
    """
    Generate static snapshots of the bridge at selected times.
    """
    max_len = min(len(time_all), len(theta_history), len(X_history))
    fig, axes = plt.subplots(1, len(snapshot_times), figsize=(4*len(snapshot_times), 4))

    if len(snapshot_times) == 1:
        axes = [axes]

    for ax, t_snap in zip(axes, snapshot_times):

        # Find closest index
        idx = np.argmin(np.abs(time_all - t_snap))

    # Then clip to valid range
        idx = min(idx, len(theta_history) - 1)

        X = X_history[idx]
        theta = theta_history[idx]

        ax.set_xlim(-5, 5)
        ax.set_ylim(-2, 2)
        ax.set_title(f"t = {time_all[idx]:.1f}")
        ax.set_xticks([])
        ax.set_yticks([])

        # Bridge
        bridge_x = np.linspace(-4, 4, 200) + X
        bridge_y = np.zeros_like(bridge_x)
        ax.plot(bridge_x, bridge_y, 'k-', linewidth=4)

        # Pedestrians
        x_positions = np.linspace(-3.5, 3.5, len(theta)) + X
        y_positions = 0.5 * np.sin(theta)

        ax.scatter(x_positions, y_positions, s=20)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)

    print(f"Snapshots saved as {filename}")

millennium_bridge_gif(
    time_all=time_all,
    theta_history=theta_history,
    X_history=X_all
)

millennium_bridge_snapshots(
    time_all,
    theta_history,
    X_all,
    snapshot_times=[
        time_all[int(0.1*len(theta_history))],
        time_all[int(0.4*len(theta_history))],
        time_all[int(0.6*len(theta_history))],
        time_all[int(0.9*len(theta_history))]
    ]
)