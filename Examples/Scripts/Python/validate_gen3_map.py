"""
Validate the Gen3 ODD material map against the Geant4 truth recording.

Two diagnostics, both geometry-free (just the ROOT track files):

  Plot 1  X0 vs eta            -- catches MISSING material (unmapped holes).
                                  Blind to smearing: material summed along a
                                  track is conserved no matter which surface it
                                  lands on, so a fully-mapped-but-smeared map
                                  still reads ~1.0 here.
  Plot 2  material vs radius   -- catches SMEARING. Truth material sits at the
                                  real layer radii (mat_r); the map collapses it
                                  onto the few designated surface radii (sur_r).

Usage:
    ACTS_SEQUENCER_DISABLE_FPEMON=1 python validate_gen3_map.py     # (env not needed here, no Sequencer)
    python validate_gen3_map.py
"""
from pathlib import Path
import uproot
import numpy as np
import matplotlib.pyplot as plt

# Dataset layout (organised 2026-08-19):
#   recordings/         Geant4 truth inputs
#   gen3_mapping/       map + mapped/unmapped tracks
#   gen3_mapping/plots/ validation PNGs written here
DATA = Path("/home/themrluke/Datasets/acts")
TRUTH = DATA / "recordings" / "odd_geant4_material_andi.root"
MAPPED = DATA / "gen3_mapping" / "gen3_mapping_mapped.root"
OUTDIR = DATA / "gen3_mapping" / "plots"
NMAX = 400_000          # tracks to sample; plenty for smooth profiles, keeps memory sane
TREE = "material_tracks"


def load(path, branches, nmax=NMAX):
    return uproot.open(path)[TREE].arrays(branches, library="np", entry_stop=nmax)


def track_totals(a):
    """Sum step_length / X0 per track -> total X0 along each track."""
    out = np.empty(len(a["v_eta"]))
    for i, (sl, x0) in enumerate(zip(a["mat_step_length"], a["mat_X0"])):
        x0 = np.asarray(x0, float)
        c = np.where(x0 > 0, np.asarray(sl, float) / x0, 0.0)
        out[i] = c.sum()
    return out


def profile(x, y, edges):
    idx = np.digitize(x, edges) - 1
    return np.array([y[idx == i].mean() if np.any(idx == i) else np.nan
                     for i in range(len(edges) - 1)])


# ---- Plot 1: X0 vs eta (truth vs map) ----
tt = load(TRUTH, ["v_eta", "mat_step_length", "mat_X0"])
mm = load(MAPPED, ["v_eta", "mat_step_length", "mat_X0"])
te, tx = tt["v_eta"], track_totals(tt)
me, mx = mm["v_eta"], track_totals(mm)

edges = np.linspace(-4, 4, 41)
c = 0.5 * (edges[:-1] + edges[1:])
pt, pm = profile(te, tx, edges), profile(me, mx, edges)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1], figsize=(8, 6))
ax1.plot(c, pt, label="Geant4 truth")
ax1.plot(c, pm, "--", label="Gen3 map (mapped tracks)")
ax1.set_ylabel(r"$\langle X/X_0 \rangle$"); ax1.legend(); ax1.set_title("Integrated material vs eta")
ax2.plot(c, pm / pt); ax2.axhline(1.0, ls=":", color="grey")
ax2.set_ylabel("map / truth"); ax2.set_xlabel(r"$\eta$"); ax2.set_ylim(0.8, 1.2)
fig.tight_layout(); fig.savefig(OUTDIR / "gen3_val_x0_vs_eta.png", dpi=150)
print("wrote", OUTDIR / "gen3_val_x0_vs_eta.png")

# ---- Plot 2: material vs radius (the smearing plot) ----
tr = load(TRUTH, ["mat_r", "mat_step_length", "mat_X0"])
mr = load(MAPPED, ["mat_r", "sur_r", "mat_step_length", "mat_X0"])

def flat(a, key):
    return np.concatenate([np.asarray(x, float) for x in a[key]])

t_r = flat(tr, "mat_r")
t_w = flat(tr, "mat_step_length") / np.where(flat(tr, "mat_X0") > 0, flat(tr, "mat_X0"), np.inf)
m_surr = flat(mr, "sur_r")
m_w = flat(mr, "mat_step_length") / np.where(flat(mr, "mat_X0") > 0, flat(mr, "mat_X0"), np.inf)

rbins = np.linspace(0, 1200, 121)
plt.figure(figsize=(9, 5))
plt.hist(t_r, bins=rbins, weights=t_w, histtype="step", label="Truth: material at real radius (mat_r)")
plt.hist(m_surr, bins=rbins, weights=m_w, histtype="step",
         label="Map: material collapsed onto surface radius (sur_r)")
plt.xlabel("radius [mm]"); plt.ylabel(r"$\sum X/X_0$ (weighted)")
plt.title("Where the material sits: truth spread vs map spikes")
plt.legend()
plt.savefig(OUTDIR / "gen3_val_material_vs_r.png", dpi=150)
print("wrote", OUTDIR / "gen3_val_material_vs_r.png")

# ---- one-line summaries ----
surr_vals, surr_cnt = np.unique(np.round(m_surr, 0), return_counts=True)
top = sorted(zip(surr_vals, surr_cnt), key=lambda z: -z[1])[:6]
print("\nMap uses these surface radii (mm):", [f"{v:.0f}({c})" for v, c in top])
print("Integrated map/truth (eta-averaged):",
      f"{np.nansum(pm)/np.nansum(pt):.3f}  <- ~1.0 = totals conserved, says nothing about placement")
