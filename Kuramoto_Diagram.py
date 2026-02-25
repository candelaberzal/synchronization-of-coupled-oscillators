import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import cauchy
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


# ============================================================
#  Population Initialization
# ============================================================

def generate_population(N, distribution="normal", scale=1.0, seed=None):
    """
    Generate initial phases and natural frequencies.
    """
    if seed is not None:
        np.random.seed(seed)

    theta0 = np.random.uniform(0, 2 * np.pi, N)

    if distribution == "normal":
        omega = np.random.normal(0, scale, N)
    elif distribution == "cauchy":
        omega = cauchy.rvs(loc=0, scale=scale, size=N)
    else:
        raise ValueError("Distribution must be 'normal' or 'cauchy'.")

    return theta0, omega


# ============================================================
#  Kuramoto Mean-Field Model
# ============================================================

def kuramoto_rhs(t, theta, omega, K):
    """
    Mean-field Kuramoto ODE using complex order parameter formulation.
    """
    z = np.mean(np.exp(1j * theta))
    return omega + K * np.imag(z * np.exp(-1j * theta))


def compute_order_parameter(theta):
    """
    Compute synchronization index r(t).
    """
    return np.abs(np.mean(np.exp(1j * theta), axis=0))


# ============================================================
#  PART 1 — Single K Simulation
# ============================================================

def simulate_single_K(
    N=100,
    K=3.0,
    distribution="normal",
    scale=1.0,
    t_end=50,
    dt=0.01,
    seed=1,
):
    """
    Simulate Kuramoto model for fixed K and plot r(t).
    """

    t_span = (0, t_end)
    t_eval = np.arange(0, t_end, dt)

    theta0, omega = generate_population(
        N, distribution=distribution, scale=scale, seed=seed
    )

    sol = solve_ivp(
        kuramoto_rhs,
        t_span,
        theta0,
        t_eval=t_eval,
        args=(omega, K),
    )

    theta_t = np.mod(sol.y, 2 * np.pi)
    r_t = compute_order_parameter(theta_t)

    plt.figure()
    plt.plot(t_eval, r_t)
    plt.xlabel("Time")
    plt.ylabel("Order parameter r(t)")
    plt.title(f"Order parameter evolution (K = {K})")
    plt.grid()
    plt.show()

    return t_eval, r_t, theta_t

def compare_sub_supercritical(
    N=100,
    K_values=(1.0, 7.0),
    distribution="normal",
    scale=1.0,
    t_end=50,
    dt=0.01,
    seed=1,
):
    """
    Plot r(t) for subcritical and supercritical coupling on the same axes.
    """

    t_span = (0, t_end)
    t_eval = np.arange(0, t_end, dt)

    plt.figure(figsize=(8, 5))

    for K in K_values:

        theta0, omega = generate_population(
            N, distribution=distribution, scale=scale, seed=seed
        )

        sol = solve_ivp(
            kuramoto_rhs,
            t_span,
            theta0,
            t_eval=t_eval,
            args=(omega, K),
        )

        theta_t = np.mod(sol.y, 2 * np.pi)
        r_t = compute_order_parameter(theta_t)

        label_type = "subcritical" if K < 1.6 else "supercritical"
        plt.plot(t_eval, r_t, label=f"K = {K} ({label_type})")

    plt.xlabel("Time")
    plt.ylabel("Order parameter r(t)")
    plt.title("Comparison Below and Above the Critical Coupling")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show() 

def animate_oscillators(theta_t, interval=30):
    """
    Animate oscillator motion on unit circle.
    """

    fig, ax = plt.subplots()
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.grid()

    circle = plt.Circle((0, 0), 1, fill=False)
    ax.add_artist(circle)

    scatter = ax.scatter([], [])
    centroid_line, = ax.plot([], [], "r-", linewidth=2)

    def update(frame):
        theta = theta_t[:, frame]
        x = np.cos(theta)
        y = np.sin(theta)

        scatter.set_offsets(np.column_stack((x, y)))

        z = np.mean(np.exp(1j * theta))
        centroid_line.set_data([0, np.real(z)], [0, np.imag(z)])

        return scatter, centroid_line

    ani = FuncAnimation(fig, update, frames=theta_t.shape[1], interval=interval)
    plt.show()
    import matplotlib.animation as animation

def compare_animation_gif(
    N=100,
    K_values=(1.0, 7.0),
    distribution="normal",
    scale=1.0,
    t_end=30,
    dt=0.05,
    seed=1,
    filename="kuramoto_comparison.gif"
):
    """
    Create side-by-side GIF comparing subcritical and supercritical coupling.
    """

    # Time grid
    t_span = (0, t_end)
    t_eval = np.arange(0, t_end, dt)

    # SAME initial conditions for fairness
    theta0, omega = generate_population(
        N, distribution=distribution, scale=scale, seed=seed
    )

    theta_solutions = []

    # Simulate for each K
    for K in K_values:
        sol = solve_ivp(
            kuramoto_rhs,
            t_span,
            theta0,
            t_eval=t_eval,
            args=(omega, K),
        )
        theta_solutions.append(np.mod(sol.y, 2*np.pi))

    # Setup figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    scatters = []
    centroid_lines = []

    for ax, K in zip(axes, K_values):
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_title(f"K = {K}")
        ax.add_artist(plt.Circle((0, 0), 1, fill=False))
        ax.grid()

        scatter = ax.scatter([], [])
        line, = ax.plot([], [], "r-", linewidth=2)

        scatters.append(scatter)
        centroid_lines.append(line)

    def update(frame):
        for i, theta_t in enumerate(theta_solutions):
            theta = theta_t[:, frame]
            x = np.cos(theta)
            y = np.sin(theta)

            scatters[i].set_offsets(np.column_stack((x, y)))

            z = np.mean(np.exp(1j * theta))
            centroid_lines[i].set_data([0, np.real(z)], [0, np.imag(z)])

        return scatters + centroid_lines

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(t_eval),
        interval=50,
        blit=True
    )

    # Save as GIF
    ani.save(filename, writer=animation.PillowWriter(fps=20))
    plt.close(fig)

    print(f"GIF saved as {filename}")

def compare_different_N(
    N_values=[50, 100, 500, 2000],
    K=7.0,
    distribution="normal",
    scale=1.0,
    t_end=50,
    dt=0.01,
    seed=1,
):
    """
    Compare r(t) for different population sizes N at fixed K.
    """

    t_span = (0, t_end)
    t_eval = np.arange(0, t_end, dt)

    plt.figure(figsize=(8, 5))

    for N in N_values:

        theta0, omega = generate_population(
            N, distribution=distribution, scale=scale, seed=seed
        )

        sol = solve_ivp(
            kuramoto_rhs,
            t_span,
            theta0,
            t_eval=t_eval,
            args=(omega, K),
        )

        theta_t = np.mod(sol.y, 2 * np.pi)
        r_t = compute_order_parameter(theta_t)

        plt.plot(t_eval, r_t, label=f"N = {N}")

    plt.xlabel("Time")
    plt.ylabel("Order parameter r(t)")
    plt.title(f"Effect of population size (K = {K})")
    plt.legend()
    plt.grid()
    plt.show()

def kuramoto_comparison_snapshots(
    N=100,
    K_values=(1.0, 7.0),
    distribution="normal",
    scale=1.0,
    t_end=30,
    dt=0.05,
    seed=1,
    filename="kuramoto_circle_snapshots.png"
):
    """
    Generate 3x2 snapshot figure comparing subcritical and supercritical cases.
    """

    t_span = (0, t_end)
    t_eval = np.arange(0, t_end, dt)

    # SAME initial conditions
    theta0, omega = generate_population(
        N, distribution=distribution, scale=scale, seed=seed
    )

    theta_solutions = []

    for K in K_values:
        sol = solve_ivp(
            kuramoto_rhs,
            t_span,
            theta0,
            t_eval=t_eval,
            args=(omega, K),
        )
        theta_solutions.append(np.mod(sol.y, 2*np.pi))

    r_super = np.abs(np.mean(np.exp(1j * theta_solutions[1]), axis=0))
    # Choose three time indices (early, mid, late)
    # Find meaningful indices based on synchronization level
    idx_early = np.where(r_super < 0.2)[0][0]       # small r
    idx_mid   = np.where(r_super > 0.7)[0][0]       # starting to cluster
    idx_late  = np.where(r_super > 0.98)[0][0]       # strong sync

    indices = [idx_early, idx_mid, idx_late]
    fig, axes = plt.subplots(3, 2, figsize=(8, 10))

    for row, idx in enumerate(indices):
        for col, (K, theta_t) in enumerate(zip(K_values, theta_solutions)):

            ax = axes[row, col]
            theta = theta_t[:, idx]

# Compute mean phase
            z = np.mean(np.exp(1j * theta))
            psi = np.angle(z)

# Move to rotating frame
            theta_rot = theta - psi

            x = np.cos(theta_rot)
            y = np.sin(theta_rot)

            ax.scatter(x, y, s=20)
            ax.add_artist(plt.Circle((0, 0), 1, fill=False))

# Force full unit circle view
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect("equal", adjustable="box")

            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(f"K = {K}")

            if col == 0:
                ax.set_ylabel(f"t = {t_eval[idx]:.1f}")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)

    print(f"Kuramoto snapshots saved as {filename}")

# ============================================================
#  PART 2 — Theoretical Prediction
# ============================================================

def theoretical_order_parameter(K_values, scale=1.0, distribution="normal"):
    """
    Compute theoretical r(K).
    """
    if distribution == "normal":
        g0 = 1 / (scale * np.sqrt(2 * np.pi))
    elif distribution == "cauchy":
        g0 = cauchy.pdf(0, loc=0, scale=scale)
    else:
        raise ValueError("Invalid distribution.")

    Kc = 2.0 / (np.pi * g0)

    r_theory = np.zeros_like(K_values)
    mask = K_values > Kc

    if distribution == "cauchy":
        r_theory[mask] = np.sqrt(1 - Kc / K_values[mask])
    else:
        # Approximation near onset
        r_theory[mask] = np.sqrt(1 - Kc / K_values[mask])
        r_theory = np.minimum(r_theory, 1)

    return r_theory, Kc


# ============================================================
#  PART 2 — Bifurcation Diagram
# ============================================================

def bifurcation_diagram(
    N=5000,
    distribution="cauchy",
    scale=1.0,
    t_end=100,
    dt=0.01,
    K_min=0.0,
    K_max=5.0,
    num_K=40,
    seed=1,
):
    """
    Compute and plot r∞(K) with error bars.
    """

    t_span = (0, t_end)
    t_eval = np.arange(0, t_end, dt)

    K_values = np.linspace(K_min, K_max, num_K)

    r_mean = np.zeros_like(K_values)
    r_std = np.zeros_like(K_values)

    r_theory, Kc = theoretical_order_parameter(
        K_values, scale=scale, distribution=distribution
    )

    print(f"Theoretical critical coupling Kc = {Kc:.4f}")

    theta, omega = generate_population(
        N, distribution=distribution, scale=scale, seed=seed
    )

    for i, K in enumerate(K_values):

        sol = solve_ivp(
            kuramoto_rhs,
            t_span,
            theta,
            t_eval=t_eval,
            args=(omega, K),
        )

        theta_t = np.mod(sol.y, 2 * np.pi)
        r_t = compute_order_parameter(theta_t)

        steady_index = int(0.75 * len(r_t))
        r_ss = r_t[steady_index:]

        r_mean[i] = np.mean(r_ss)
        r_std[i] = np.std(r_ss)

        theta = theta_t[:, -1]

        print(f"K = {K:.3f} | r_mean = {r_mean[i]:.4f}")

    fig, ax = plt.subplots()

    ax.plot(K_values, r_theory, label="Theoretical", linewidth=2)

    ax.errorbar(
        K_values,
        r_mean,
        yerr=r_std,
        fmt="o",
        capsize=3,
        label="Numerical",
    )

    ax.axvline(Kc, linestyle="--", color="black", label="Kc")

    ax.set_xlabel("Coupling strength K")
    ax.set_ylabel("Order parameter r")
    ax.set_title("Kuramoto Synchronization Diagram")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":

    # -------------------------
    # PART 1: Time Dynamics
    # -------------------------

    # Subcritical example
    t, r, theta_t = simulate_single_K(K=1.0)

    # Supercritical example
    t, r, theta_t = simulate_single_K(K=3.0)

    # Compare different N at fixed K
    compare_different_N(K=7.0)

    compare_sub_supercritical()

    # Optional animation (comment out if not needed)
    animate_oscillators(theta_t)

    kuramoto_comparison_snapshots()

    #compare_animation_gif()
    # -------------------------
    # PART 2: Bifurcation
    # -------------------------

    #bifurcation_diagram()