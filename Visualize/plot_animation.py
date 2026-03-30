"""
plot_animation.py — Animated Pareto front evolution (GIF + MP4).

Each frame shows:
  • All previous generations' Pareto points as grey ghost dots
  • The current generation's Pareto front in coolwarm colour
  • A "Generation X / Y" annotation in the top-right corner

Usage:
    python Visualize/plot_animation.py [path/to/all_generations_xxx.csv]

If no path is given, automatically picks the newest all_generations_*.csv
in the Results/ directory (relative to this script's parent).

Output:
    Results/pareto_animation_{ts}.gif   (always produced)
    Results/pareto_animation_{ts}.mp4   (skipped if FFmpeg is not installed)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ── Reuse helpers from the sibling module ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pareto_evolution import (
    _find_latest_csv,
    _read_generations,
    _ts_from_path,
    _pareto_front,
)


# ============================================================
# Animation
# ============================================================

def animate(csv_path):
    gen_data, last_gen = _read_generations(csv_path)
    if not gen_data:
        print("No valid data found in CSV.")
        return

    ts          = _ts_from_path(csv_path)
    results_dir = os.path.dirname(csv_path)
    out_gif     = os.path.join(results_dir, f"pareto_animation_{ts}.gif")
    out_mp4     = os.path.join(results_dir, f"pareto_animation_{ts}.mp4")

    # Build per-generation Pareto fronts once
    gens_sorted       = sorted(gen_data.keys())
    gen_pareto_fronts = [(g, _pareto_front(gen_data[g])) for g in gens_sorted]
    n_frames          = len(gen_pareto_fronts)

    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=0, vmax=last_gen)

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1.25, 1.0, 0.8))
    ax.set_xlabel("Avg Torque (Nm)",    labelpad=10)
    ax.set_ylabel("Torque Ripple (%)",  labelpad=10)
    ax.set_zlabel("Avg Efficiency (%)", labelpad=10)
    ax.view_init(elev=20, azim=145)
    fig.subplots_adjust(left=-0.02, right=0.98)

    # Annotation text (top-right corner in axes coordinates)
    gen_text = ax.text2D(0.97, 0.96, "",
                         transform=ax.transAxes,
                         ha="right", va="top",
                         fontsize=11, fontweight="bold",
                         color="black",
                         bbox=dict(boxstyle="round,pad=0.3",
                                   facecolor="white", alpha=0.7, edgecolor="none"))

    # Persistent scatter handles (updated each frame)
    ghost_sc   = ax.scatter([], [], [], s=15,  alpha=0.20,
                            color="grey", edgecolors="none", depthshade=False)
    current_sc = ax.scatter([], [], [], s=40,  alpha=0.90,
                            edgecolors="k", linewidths=0.5, depthshade=True)

    # Pre-compute axis limits from ALL valid data so the view is stable
    all_pts = [s for _, pf in gen_pareto_fronts for s in pf]
    if all_pts:
        pad = 0.02
        t_all = [s[0] for s in all_pts]
        r_all = [s[1] for s in all_pts]
        e_all = [s[2] for s in all_pts]
        ax.set_xlim(min(t_all) * (1 - pad), max(t_all) * (1 + pad))
        ax.set_ylim(min(r_all) * (1 - pad), max(r_all) * (1 + pad))
        ax.set_zlim(min(e_all) * (1 - pad), max(e_all) * (1 + pad))
    # invert_yaxis() called after set_ylim so it sets both the limits
    # and matplotlib's internal inversion flag — this controls which face
    # of the 3D box appears at the front, matching the static plot orientation.
    ax.invert_yaxis()

    # ── Frame update function ─────────────────────────────────────────────────
    def _update(frame_idx):
        gen_idx, pf = gen_pareto_fronts[frame_idx]

        # Ghost: all Pareto points from generations 0 … frame_idx-1
        ghost_pts = [s
                     for fi in range(frame_idx)
                     for s in gen_pareto_fronts[fi][1]]
        if ghost_pts:
            gx = np.array([s[0] for s in ghost_pts])
            gy = np.array([s[1] for s in ghost_pts])
            gz = np.array([s[2] for s in ghost_pts])
            ghost_sc._offsets3d = (gx, gy, gz)
        else:
            ghost_sc._offsets3d = (np.array([]), np.array([]), np.array([]))

        # Current generation
        if pf:
            cx = np.array([s[0] for s in pf])
            cy = np.array([s[1] for s in pf])
            cz = np.array([s[2] for s in pf])
            current_sc._offsets3d = (cx, cy, cz)
            color = cmap(norm(gen_idx))
            current_sc.set_facecolor(color)
        else:
            current_sc._offsets3d = (np.array([]), np.array([]), np.array([]))

        gen_text.set_text(f"Generation: {gen_idx} / {last_gen}")
        ax.set_title(f"NSGA-III Pareto Front Evolution - SPMSM\n"
                     f"({n_frames} generations)", pad=12)

        return ghost_sc, current_sc, gen_text

    # ── Build animation ───────────────────────────────────────────────────────
    anim = animation.FuncAnimation(
        fig, _update,
        frames=n_frames,
        interval=1500,          # ms per frame
        blit=False,             # blit=True is unreliable for 3D axes
        repeat=True,
    )

    # ── Save GIF (Pillow — always available with matplotlib) ──────────────────
    print(f"Saving GIF ({n_frames} frames) ...", flush=True)
    writer_gif = animation.PillowWriter(fps=1000 // 1500 or 1)
    anim.save(out_gif, writer=writer_gif, dpi=120)
    print(f"GIF saved  -> {out_gif}", flush=True)

    # ── Save MP4 (FFmpeg — optional) ──────────────────────────────────────────
    if not animation.writers.is_available("ffmpeg"):
        print("FFmpeg not available to matplotlib — MP4 skipped.", flush=True)
        print("  Ensure ffmpeg.exe is on your system PATH and restart the terminal.", flush=True)
    else:
        try:
            writer_mp4 = animation.FFMpegWriter(fps=1, bitrate=1800)
            print("Saving MP4 ...", flush=True)
            anim.save(out_mp4, writer=writer_mp4, dpi=150)
            print(f"MP4 saved  -> {out_mp4}", flush=True)
        except OSError as e:
            print(f"MP4 save failed: {e}  ->  MP4 skipped.", flush=True)

    plt.show()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.normpath(os.path.join(script_dir, "..", "Results"))
        csv_path    = _find_latest_csv(results_dir)
        print(f"Using: {csv_path}")

    animate(csv_path)
