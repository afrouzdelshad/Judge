#!/usr/bin/env python3
"""
Anchored N-Lobed Lorenz Attractor
=================================

Cylindrical-coordinate generalisation of the Lorenz system with N lobes:

    phi    = (N/2) * (theta - pi/4) + pi/4

    r'     = r * [ (sigma + rho - z) * sin(phi)cos(phi)
                   - sigma * cos^2(phi)
                   - sin^2(phi) ]

    theta' = (2/N) * [ (rho - z) * cos^2(phi)
                       - sigma * sin^2(phi)
                       + (sigma - 1) * sin(phi)cos(phi) ]

    z'     = r^2 * sin(phi)cos(phi) - beta * z

For N = 2 this reduces exactly to the ordinary Lorenz system in cylindrical
coordinates.  The nonzero fixed points sit at

    theta_k = pi/4 + 2*pi*k/N,   r_k = r_L = sqrt(2*beta*(rho-1)),   z_k = rho-1

so the k = 0 fixed point always coincides with the positive fixed point of the
original Lorenz system (hence "anchored").  With the standard parameters
r_L = 12 and P+ = (6*sqrt(2), 6*sqrt(2), 27).

Usage
-----
Interactive window (default):

    python anchored_n_lorenz.py

Headless single run, save figure and data:

    python anchored_n_lorenz.py --no-gui --N 5 --time 300 --save --png out.png

Coarse measurement grid and reduced precision (much smaller CSVs):

    python anchored_n_lorenz.py --no-gui --N 5 --time 300 \
        --dt-out 0.2 --precision 4 --save

Note that --dt-out is a *measurement* interval only.  The integrator is
adaptive with its own step cap (MAX_STEP), so coarsening --dt-out subsamples
an equally accurate trajectory rather than integrating it less accurately.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0

R_LORENZ = np.sqrt(2.0 * BETA * (RHO - 1.0))   # = 12 for the standard values
Z_LORENZ = RHO - 1.0                           # = 27

IC_X_RANGE = (-25.0, 25.0)
IC_Y_RANGE = (-25.0, 25.0)
IC_Z_RANGE = (0.0, 55.0)

DT_OUT = 0.02          # default measurement (output) sampling interval
PRECISION = 17         # default significant digits written to CSV (full float64)
MAX_STEP = 0.05        # integrator step cap — independent of DT_OUT
T_PRECISION = 12       # digits for the time column (kept high on purpose)
DEFAULT_OUTDIR = "n_lorenz_saved_runs"


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def anchored_n_lorenz(t, state, N, sigma=SIGMA, rho=RHO, beta=BETA):
    """Right-hand side of the anchored N-lobed Lorenz system in (r, theta, z)."""
    r, theta, z = state

    phi = 0.5 * N * (theta - np.pi / 4.0) + np.pi / 4.0

    sp = np.sin(phi)
    cp = np.cos(phi)

    dr = r * (
        (sigma + rho - z) * sp * cp
        - sigma * cp**2
        - sp**2
    )

    dtheta = (2.0 / N) * (
        (rho - z) * cp**2
        - sigma * sp**2
        + (sigma - 1.0) * sp * cp
    )

    dz = r**2 * sp * cp - beta * z

    return [dr, dtheta, dz]


def cartesian_to_cylindrical(x0, y0, z0):
    r0 = np.hypot(x0, y0)

    if r0 < 1.0e-10:
        r0 = 1.0e-10
        theta0 = 0.0
    else:
        theta0 = np.arctan2(y0, x0)

    return np.array([r0, theta0, z0], dtype=float)


def cylindrical_to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


def fixed_points(N):
    """The N nonzero equilibria, in Cartesian coordinates."""
    k = np.arange(N)
    theta_k = np.pi / 4.0 + 2.0 * np.pi * k / N
    xk = R_LORENZ * np.cos(theta_k)
    yk = R_LORENZ * np.sin(theta_k)
    zk = np.full(N, Z_LORENZ)
    return xk, yk, zk, theta_k


def random_initial_condition(seed):
    """Uniform sample from the Cartesian box; same seed -> same point."""
    rng = np.random.default_rng(int(seed))
    x0 = rng.uniform(*IC_X_RANGE)
    y0 = rng.uniform(*IC_Y_RANGE)
    z0 = rng.uniform(*IC_Z_RANGE)

    if np.hypot(x0, y0) < 1.0e-8:
        x0 = 1.0e-6

    return float(x0), float(y0), float(z0)


def simulate(N, T, x0, y0, z0, seed=0, dt_out=DT_OUT):
    """Integrate the system and return a dict holding the whole run.

    `dt_out` is the *measurement* sampling interval: it controls only how
    densely the solution is sampled for output.  The integration itself is
    adaptive and capped by MAX_STEP, so changing dt_out does not change the
    accuracy of the trajectory — only how much of it you record.
    """
    N = int(N)
    T = float(T)
    dt_out = float(dt_out)

    if not 0.0 < dt_out <= T:
        raise ValueError(f"dt_out must be in (0, T]; got {dt_out} with T={T}")

    state0 = cartesian_to_cylindrical(x0, y0, z0)

    npts = int(np.floor(T / dt_out)) + 1
    t_eval = np.arange(npts) * dt_out

    sol = solve_ivp(
        anchored_n_lorenz,
        (0.0, T),
        state0,
        args=(N,),
        method="DOP853",
        t_eval=t_eval,
        rtol=1.0e-9,
        atol=1.0e-11,
        max_step=MAX_STEP,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    r, theta, z = sol.y
    x, y = cylindrical_to_cartesian(r, theta)

    return {
        "N": N,
        "T": T,
        "dt_out": dt_out,
        "seed": int(seed),
        "x0": float(x0),
        "y0": float(y0),
        "z0": float(z0),
        "t": sol.t,
        "x": x,
        "y": y,
        "z": z,
        "r": r,
        "theta": theta,
    }


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------

def save_run(run, outdir=DEFAULT_OUTDIR, precision=PRECISION):
    """Write three single-variable CSVs, a combined CSV, and a metadata file.

    `precision` is the number of significant digits used for x, y and z.
    The time column is always written at T_PRECISION digits: at a coarse
    `precision` a low-digit time column would collapse distinct samples
    (e.g. 299.98 and 299.96 both becoming 300.0), which would corrupt the
    sampling grid rather than merely coarsen the measurement.
    """
    save_dir = Path(outdir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dt_out = run.get("dt_out", DT_OUT)

    base = (
        f"anchored_NLorenz_N{run['N']}_seed{run['seed']}"
        f"_T{int(round(run['T']))}_dt{dt_out:g}_p{precision}"
    )

    t, x, y, z = run["t"], run["x"], run["y"], run["z"]

    t_fmt = f"%.{T_PRECISION}g"
    v_fmt = f"%.{precision}g"

    files = []
    for name, arr, header in (("x", x, "time,x"), ("y", y, "time,y"), ("z", z, "time,z")):
        path = save_dir / f"{base}_{name}_vs_time.csv"
        np.savetxt(path, np.column_stack((t, arr)), fmt=[t_fmt, v_fmt],
                   delimiter=",", header=header, comments="")
        files.append(path)

    xyz_file = save_dir / f"{base}_xyz_vs_time.csv"
    np.savetxt(xyz_file, np.column_stack((t, x, y, z)),
               fmt=[t_fmt, v_fmt, v_fmt, v_fmt],
               delimiter=",", header="time,x,y,z", comments="")
    files.append(xyz_file)

    meta_file = save_dir / f"{base}_metadata.txt"
    with open(meta_file, "w", encoding="utf-8") as f:
        f.write("Anchored N-Lorenz simulation\n")
        f.write(f"N = {run['N']}\n")
        f.write(f"T = {run['T']}\n")
        f.write(f"dt_out = {dt_out:.17g}\n")
        f.write(f"precision = {precision}\n")
        f.write(f"n_samples = {len(t)}\n")
        f.write(f"seed = {run['seed']}\n")
        f.write(f"x0 = {run['x0']:.17g}\n")
        f.write(f"y0 = {run['y0']:.17g}\n")
        f.write(f"z0 = {run['z0']:.17g}\n")
        f.write(f"sigma = {SIGMA}\n")
        f.write(f"rho = {RHO}\n")
        f.write(f"beta = {BETA:.17g}\n")
        f.write(f"r_L = {R_LORENZ:.17g}\n")
        f.write(f"integrator = DOP853, rtol=1e-9, atol=1e-11, max_step={MAX_STEP}\n")
    files.append(meta_file)

    return files


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def draw_run(run, axes):
    """Redraw all four panels for a completed run onto existing axes."""
    ax_xy, ax_xz, ax_yz, ax_t = axes

    N, t = run["N"], run["t"]
    x, y, z = run["x"], run["y"], run["z"]
    xk, yk, zk, _ = fixed_points(N)

    for ax in axes:
        ax.cla()

    ax_xy.plot(x, y, lw=0.45)
    ax_xy.scatter(xk, yk, s=35, marker="x", color="crimson", zorder=5)
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.set_title(f"x-y projection, N={N}")
    ax_xy.grid(alpha=0.25)
    ax_xy.set_aspect("equal", adjustable="box")

    ax_xz.plot(x, z, lw=0.45)
    ax_xz.scatter(xk, zk, s=35, marker="x", color="crimson", zorder=5)
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xz.set_title("x-z projection")
    ax_xz.grid(alpha=0.25)

    ax_yz.plot(y, z, lw=0.45)
    ax_yz.scatter(yk, zk, s=35, marker="x", color="crimson", zorder=5)
    ax_yz.set_xlabel("y")
    ax_yz.set_ylabel("z")
    ax_yz.set_title("y-z projection")
    ax_yz.grid(alpha=0.25)

    ax_t.plot(t, x, lw=0.75, label="x(t)")
    ax_t.plot(t, y, lw=0.75, label="y(t)")
    ax_t.plot(t, z, lw=0.75, label="z(t)")
    ax_t.set_xlabel("time")
    ax_t.set_ylabel("x, y, z")
    ax_t.set_title(
        f"Time signals   |   "
        f"IC=({run['x0']:.4g}, {run['y0']:.4g}, {run['z0']:.4g})   |   "
        f"seed={run['seed']}"
    )
    ax_t.legend(ncol=3)
    ax_t.grid(alpha=0.25)


def make_figure(with_controls):
    """Build the figure and return (fig, axes). Leaves room for widgets if asked."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10))
    bottom = 0.30 if with_controls else 0.08
    gs = fig.add_gridspec(
        2, 3,
        left=0.06, right=0.98, top=0.88, bottom=bottom,
        height_ratios=[1.0, 1.05], hspace=0.35, wspace=0.25,
    )

    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_yz = fig.add_subplot(gs[0, 2])
    ax_t = fig.add_subplot(gs[1, :])

    fig.suptitle(
        "Anchored N-Lobed Lorenz Attractor\n"
        r"$\theta_k=\pi/4+2\pi k/N,\quad r_k=r_L=12$",
        fontsize=15,
    )

    return fig, (ax_xy, ax_xz, ax_yz, ax_t)


# --------------------------------------------------------------------------
# Interactive mode
# --------------------------------------------------------------------------

def run_gui(args):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider, TextBox

    fig, axes = make_figure(with_controls=True)
    state = {"run": None}

    # ---- widget axes -----------------------------------------------------
    ax_N = fig.add_axes([0.10, 0.19, 0.34, 0.03])
    ax_T = fig.add_axes([0.58, 0.19, 0.34, 0.03])

    ax_seed = fig.add_axes([0.10, 0.12, 0.07, 0.04])
    ax_x0 = fig.add_axes([0.26, 0.12, 0.07, 0.04])
    ax_y0 = fig.add_axes([0.42, 0.12, 0.07, 0.04])
    ax_z0 = fig.add_axes([0.58, 0.12, 0.07, 0.04])

    ax_run = fig.add_axes([0.10, 0.04, 0.16, 0.05])
    ax_rand = fig.add_axes([0.30, 0.04, 0.20, 0.05])
    ax_save = fig.add_axes([0.54, 0.04, 0.24, 0.05])

    ax_status = fig.add_axes([0.80, 0.04, 0.001, 0.001])
    ax_status.set_axis_off()

    s_N = Slider(ax_N, "N lobes", 2, 20, valinit=args.N, valstep=1)
    s_T = Slider(ax_T, "Time", 100, 1000, valinit=args.time, valstep=50)

    tb_seed = TextBox(ax_seed, "seed ", initial=str(args.seed))
    tb_x0 = TextBox(ax_x0, "x0 ", initial=f"{args.x0:g}")
    tb_y0 = TextBox(ax_y0, "y0 ", initial=f"{args.y0:g}")
    tb_z0 = TextBox(ax_z0, "z0 ", initial=f"{args.z0:g}")

    b_run = Button(ax_run, "Run simulation")
    b_rand = Button(ax_rand, "Random I.C. + Run")
    b_save = Button(ax_save, "Save latest x,y,z vs time")

    status = fig.text(0.5, 0.245, "", ha="center", fontsize=9, color="dimgray")

    def set_status(msg, color="dimgray"):
        status.set_text(msg)
        status.set_color(color)
        fig.canvas.draw_idle()

    def read_float(tb, fallback):
        try:
            return float(tb.text)
        except ValueError:
            tb.set_val(f"{fallback:g}")
            return fallback

    def do_run(_event=None):
        try:
            x0 = read_float(tb_x0, 1.0)
            y0 = read_float(tb_y0, 1.0)
            z0 = read_float(tb_z0, 20.0)
            try:
                seed = int(float(tb_seed.text))
            except ValueError:
                seed = 1
                tb_seed.set_val("1")

            run = simulate(int(s_N.val), float(s_T.val), x0, y0, z0, seed,
                           dt_out=args.dt_out)
            state["run"] = run
            draw_run(run, axes)
            set_status(
                f"Latest run: N={run['N']}, T={run['T']:.0f}, "
                f"IC=({run['x0']:.6f}, {run['y0']:.6f}, {run['z0']:.6f}), "
                f"seed={run['seed']}, {len(run['t']):,} saved time points."
            )
        except Exception as exc:                      # noqa: BLE001
            set_status(f"ERROR: {exc}", color="crimson")

    def do_random(_event=None):
        try:
            seed = int(float(tb_seed.text))
        except ValueError:
            seed = 1
            tb_seed.set_val("1")

        x0, y0, z0 = random_initial_condition(seed)
        tb_x0.set_val(f"{x0:.6f}")
        tb_y0.set_val(f"{y0:.6f}")
        tb_z0.set_val(f"{z0:.6f}")
        do_run()

    def do_save(_event=None):
        if state["run"] is None:
            set_status("No completed simulation yet. Run one first.", color="crimson")
            return
        files = save_run(state["run"], args.outdir, args.precision)
        set_status(f"Saved {len(files)} files to {Path(args.outdir).resolve()}")
        print("Saved the latest completed run:")
        for f in files:
            print(f"  {f}")

    b_run.on_clicked(do_run)
    b_rand.on_clicked(do_random)
    b_save.on_clicked(do_save)

    # Sliders re-run on mouse release only, mimicking continuous_update=False.
    pending = {"N": int(s_N.val), "T": float(s_T.val)}

    def on_release(event):
        if event.inaxes in (ax_N, ax_T):
            if (int(s_N.val), float(s_T.val)) != (pending["N"], pending["T"]):
                pending["N"], pending["T"] = int(s_N.val), float(s_T.val)
                do_run()

    fig.canvas.mpl_connect("button_release_event", on_release)

    # Text boxes re-run on Enter.
    for tb in (tb_x0, tb_y0, tb_z0):
        tb.on_submit(lambda _text: do_run())

    do_run()
    plt.show()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Anchored N-lobed Lorenz attractor simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--N", type=int, default=3, help="number of lobes (>= 2)")
    p.add_argument("--time", type=float, default=200.0, help="integration time")
    p.add_argument("--x0", type=float, default=1.0)
    p.add_argument("--y0", type=float, default=1.0)
    p.add_argument("--z0", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=1,
                   help="seed for the random initial condition")
    p.add_argument("--dt-out", type=float, default=DT_OUT, dest="dt_out",
                   help="measurement sampling interval (does NOT affect "
                        "integration accuracy, only how densely the "
                        "trajectory is recorded)")
    p.add_argument("--precision", type=int, default=PRECISION,
                   help="significant digits for x, y, z in the CSV output "
                        "(1-17; the time column is unaffected)")
    p.add_argument("--random", action="store_true",
                   help="draw the initial condition from the seed instead of x0/y0/z0")
    p.add_argument("--no-gui", dest="gui", action="store_false",
                   help="run once without the interactive window")
    p.add_argument("--save", action="store_true",
                   help="write CSV/metadata/ZIP for the run")
    p.add_argument("--png", type=str, default=None,
                   help="save the figure to this path instead of showing it")
    p.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR,
                   help="directory for saved data")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.N < 2:
        raise SystemExit("N must be at least 2.")
    if not 1 <= args.precision <= 17:
        raise SystemExit("--precision must be between 1 and 17.")
    if not 0.0 < args.dt_out <= args.time:
        raise SystemExit("--dt-out must be positive and no larger than --time.")

    print(f"Original Lorenz fixed-point radius r_L = {R_LORENZ:.6f}")
    print(f"Original Lorenz fixed-point height z_L = {Z_LORENZ:.6f}")

    if args.gui and args.png is None:
        run_gui(args)
        return

    if args.random:
        args.x0, args.y0, args.z0 = random_initial_condition(args.seed)

    run = simulate(args.N, args.time, args.x0, args.y0, args.z0,
                   args.seed, dt_out=args.dt_out)
    print(
        f"Run: N={run['N']}, T={run['T']:.0f}, dt_out={run['dt_out']:g}, "
        f"precision={args.precision}, "
        f"IC=({run['x0']:.6f}, {run['y0']:.6f}, {run['z0']:.6f}), "
        f"seed={run['seed']}, {len(run['t']):,} saved time points."
    )

    import matplotlib
    if args.png is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = make_figure(with_controls=False)
    draw_run(run, axes)

    if args.png is not None:
        fig.savefig(args.png, dpi=150)
        print(f"Figure written to {args.png}")
    else:
        plt.show()

    if args.save:
        print("Saved the latest completed run:")
        for f in save_run(run, args.outdir, args.precision):
            print(f"  {f}")


if __name__ == "__main__":
    main()