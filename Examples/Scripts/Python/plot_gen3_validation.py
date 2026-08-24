"""
Tracking-level validation of the Gen3 ODD material read-back.

Reads material_validation.py output (fresh geantinos propagated through the
Gen3 geometry that now carries the mapped material) and compares against the
Geant4 truth recording. Produces four diagnostics:

  1. X0 vs eta   integrated radiation-length budget vs pseudorapidity.
  2. X0 vs phi   same, vs azimuth -- catches phi-dependent holes / non-uniformity.
  3. L0 vs eta   integrated *nuclear interaction* length -- an independent budget
                 (weights materials differently to X0; agreeing in both is a
                 stronger statement than X0 alone).
  4. (z,r) map   material-weighted position of every step, truth vs map, side by
                 side. This is the SMEARING view: truth material sits at the real
                 layer radii; a smeared map collapses it onto the designated
                 surface radii. (Plots 1-3 are integrated along the track and so
                 are blind to smearing -- this one is not.)

Run WITHOUT Python/Examples/python on PYTHONPATH (its uproot.py shim shadows the
real uproot):
    env -u PYTHONPATH bash -c 'source build/python/setup.sh; \\
      python Examples/Scripts/Python/plot_gen3_validation.py'
"""
from pathlib import Path
import uproot
import numpy as np
import matplotlib.pyplot as plt

DATA = Path("/home/themrluke/Datasets/acts")
TRUTH = DATA / "recordings" / "odd_geant4_material_andi.root"
VAL = DATA / "gen3_mapping" / "gen3_val_tracks.root"
OUTDIR = DATA / "gen3_mapping" / "plots"
TREE = "material_tracks"
NMAX = 400_000  # per-track sampling cap for the profiles
NMAX_2D = 100_000  # fewer tracks needed for a dense (z,r) density map


def load(path, branches, nmax=NMAX):
    return uproot.open(path)[TREE].arrays(branches, library="np", entry_stop=nmax)


def profile(x, y, edges):
    idx = np.digitize(x, edges) - 1
    return np.array(
        [y[idx == i].mean() if np.any(idx == i) else np.nan for i in range(len(edges) - 1)]
    )


def profile_panel(truth_xy, map_xy, edges, xlabel, ylabel, title, saveas,
                  self_normalize=False):
    """Two-panel (value + ratio) truth-vs-map profile plot.

    self_normalize divides each profile by its own mean -- use it to compare
    *shape* (e.g. phi uniformity) when the two files sample different eta
    distributions, which would otherwise bias an absolute comparison.
    """
    c = 0.5 * (edges[:-1] + edges[1:])
    pt = profile(truth_xy[0], truth_xy[1], edges)
    pm = profile(map_xy[0], map_xy[1], edges)
    if self_normalize:
        pt = pt / np.nanmean(pt)
        pm = pm / np.nanmean(pm)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, height_ratios=[3, 1], figsize=(8, 6)
    )
    ax1.plot(c, pt, label="Geant4 truth")
    ax1.plot(c, pm, "--", label="Gen3 geometry (mapped, propagated)")
    ax1.set_ylabel(ylabel)
    ax1.legend()
    ax1.set_title(title)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = pm / pt
    ax2.plot(c, ratio)
    ax2.axhline(1.0, ls=":", color="grey")
    ax2.set_ylabel("map / truth")
    ax2.set_xlabel(xlabel)
    ax2.set_ylim(0.8, 1.2)
    fig.tight_layout()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / saveas, dpi=150)
    plt.close(fig)
    mask = np.isfinite(ratio)
    print(f"wrote {OUTDIR / saveas}   mean(map/truth)={ratio[mask].mean():.3f}")


# ---- Plots 1-3: integrated profiles (use precomputed per-track totals) ----
tt = load(TRUTH, ["v_eta", "v_phi", "t_X0", "t_L0"])
vv = load(VAL, ["v_eta", "v_phi", "t_X0", "t_L0"])
print(f"truth tracks: {len(tt['v_eta'])}   validation tracks: {len(vv['v_eta'])}")

profile_panel(
    (tt["v_eta"], tt["t_X0"]), (vv["v_eta"], vv["t_X0"]),
    np.linspace(-4, 4, 41), r"$\eta$", r"$\langle X/X_0 \rangle$",
    "Gen3 read-back: integrated material vs eta", "gen3_tracking_x0_vs_eta.png",
)
# phi is a UNIFORMITY check (are there phi-dependent holes?), not an absolute
# comparison: (a) truth and our gun have different eta distributions, so a phi
# average over eta is confounded; (b) our map is phi-flat BY CONSTRUCTION (1 phi
# bin per surface). Restrict to |eta|<1 and self-normalise each curve to its own
# mean, so we compare phi *shape*. Both should be flat lines ~1.0 -- any dip is a
# real phi hole.
tmask = np.abs(tt["v_eta"]) < 1.0
vmask = np.abs(vv["v_eta"]) < 1.0
profile_panel(
    (tt["v_phi"][tmask], tt["t_X0"][tmask]), (vv["v_phi"][vmask], vv["t_X0"][vmask]),
    np.linspace(-np.pi, np.pi, 37), r"$\phi$", r"X/X$_0$ (self-normalised)",
    "Gen3 read-back: phi uniformity (|eta|<1)", "gen3_tracking_x0_vs_phi.png",
    self_normalize=True,
)
profile_panel(
    (tt["v_eta"], tt["t_L0"]), (vv["v_eta"], vv["t_L0"]),
    np.linspace(-4, 4, 41), r"$\eta$", r"$\langle L/L_0 \rangle$",
    "Gen3 read-back: nuclear interaction length vs eta", "gen3_tracking_l0_vs_eta.png",
)

# ---- Plot 4: (z, r) material-weighted position map (the smearing view) ----
def rz_weighted(path):
    # restrict to |eta|<4 (the gun's range) so both files cover the same solid
    # angle -- otherwise truth's few |eta|>4 tracks add forward material the map
    # can't, which would read as a spurious hole.
    a = load(path, ["v_eta", "mat_z", "mat_r", "mat_step_length", "mat_X0"], nmax=NMAX_2D)
    keep = np.abs(a["v_eta"]) < 4.0
    z = np.concatenate([np.asarray(x, float) for x in a["mat_z"][keep]])
    r = np.concatenate([np.asarray(x, float) for x in a["mat_r"][keep]])
    sl = np.concatenate([np.asarray(x, float) for x in a["mat_step_length"][keep]])
    x0 = np.concatenate([np.asarray(x, float) for x in a["mat_X0"][keep]])
    w = np.where(x0 > 0, sl / x0, 0.0)  # X/X0 contributed at each step
    return z, r, w


from matplotlib.colors import LogNorm

tz, tr, tw = rz_weighted(TRUTH)
vz, vr, vw = rz_weighted(VAL)
zbins = np.linspace(-3200, 3200, 320)
rbins = np.linspace(0, 1200, 240)

# shared log colour scale so truth and map are directly comparable and the faint
# inner layers are visible alongside the bright outer material
HT, _, _ = np.histogram2d(tz, tr, bins=[zbins, rbins], weights=tw)
HM, _, _ = np.histogram2d(vz, vr, bins=[zbins, rbins], weights=vw)
vmax = max(HT.max(), HM.max())
norm = LogNorm(vmax=vmax, vmin=vmax * 1e-3)

fig, (axT, axM) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(13, 5))
for ax, H, title in (
    (axT, HT, "Geant4 truth"),
    (axM, HM, "Gen3 map (loaded geometry)"),
):
    im = ax.pcolormesh(zbins, rbins, H.T, norm=norm, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("z [mm]")
axT.set_ylabel("r [mm]")
fig.colorbar(im, ax=(axT, axM), label=r"$\sum X/X_0$ (log)", shrink=0.8)
fig.suptitle("Where the material sits -- truth spreads over layer radii; map lands on the designated surfaces (note outer support at r~1160 truth vs r~1030 map)")
out = OUTDIR / "gen3_material_rz_map.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote", out)
