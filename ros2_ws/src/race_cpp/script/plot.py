#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib as mpl
import numpy as np
import os

# Global bold font settings
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['legend.fontsize'] = 10
for key in ['xtick.major.width', 'ytick.major.width', 'axes.linewidth']:
    mpl.rcParams[key] = 1.5

def bold_ticks(ax):
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')

CSV_PATH = os.path.join(os.path.dirname(__file__), 'log.csv')
PDF_PATH = os.path.join(os.path.dirname(__file__), 'log_plot.pdf')
IMG_DIR  = os.path.dirname(__file__)

def save_pdf(fig, name):
    fig.savefig(os.path.join(IMG_DIR, name), bbox_inches='tight')

df = pd.read_csv(CSV_PATH)
t = df['timestamp_sec'] - df['timestamp_sec'].iloc[0]

# color per sample: blue=PD, red=MPPI
seg_colors = ['r' if m else 'b' for m in df['mppi_mode'].values[:-1]]

# Transform AI position into ego frame at each timestep
dx = df['ai_x'] - df['ego_x']
dy = df['ai_y'] - df['ego_y']
cos_yaw = np.cos(df['ego_yaw'])
sin_yaw = np.sin(df['ego_yaw'])
ai_x_ego =  dx * cos_yaw + dy * sin_yaw
ai_y_ego = -dx * sin_yaw + dy * cos_yaw

legend_world = [
    Line2D([0], [0], color='b', lw=4, label='Ego (LMPC follow)'),
    Line2D([0], [0], color='r', lw=4, label='Ego (MPPI takeover)'),
    Line2D([0], [0], color='g', lw=4, label='Opponent'),
]
legend_ego_frame = [
    Line2D([0], [0], color='g', lw=4, label='Opponent (ego frame)'),
    Line2D([0], [0], color='b', lw=4, label='Ego (LMPC follow)'),
    Line2D([0], [0], color='r', lw=4, label='Ego (MPPI takeover)'),
]


def make_lc(xarr, yarr, colors, lw=4):
    pts = np.array([xarr, yarr]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    return LineCollection(segs, colors=colors, linewidths=lw)


with PdfPages(PDF_PATH) as pdf:

    # ── Page 1: XY trajectory (world frame) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.add_collection(make_lc(df['ego_x'].values, df['ego_y'].values, seg_colors))
    ax.plot(df['ai_x'], df['ai_y'], 'g-', lw=4, alpha=0.7)
    ax.autoscale()
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Ego vs Opponent Trajectories (world frame)')
    ax.legend(handles=legend_world, loc='upper right')
    ax.set_aspect('equal', 'datalim')
    ax.grid(True, alpha=0.3)
    bold_ticks(ax)
    plt.tight_layout()
    save_pdf(fig, 'plot_01_trajectory_world.pdf')
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 2: XY trajectory (ego frame) ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(ai_x_ego, ai_y_ego, 'g-', lw=4, alpha=0.9, label='Opponent (ego frame)')
    ax.plot(0, 0, 'ko', ms=10, zorder=5, label='Ego (origin)')
    ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.5)
    ax.axvline(0, color='gray', lw=1, ls='--', alpha=0.5)
    ax.autoscale()
    ax.set_xlabel('X ego (m)')
    ax.set_ylabel('Y ego (m)')
    ax.set_title('Opponent Position in Ego Frame')
    ax.legend(loc='upper right')
    ax.set_aspect('equal', 'datalim')
    ax.grid(True, alpha=0.3)
    bold_ticks(ax)
    plt.tight_layout()
    save_pdf(fig, 'plot_02_trajectory_ego.pdf')
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 3: X vs time ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.add_collection(make_lc(t.values, df['ego_x'].values, seg_colors))
    ax.plot(t, df['ai_x'], 'g-', lw=4, alpha=0.7)
    ax.autoscale()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X world (m)')
    ax.set_title('X Position vs Time (world frame)')
    ax.legend(handles=legend_world)
    ax.grid(True, alpha=0.3)
    bold_ticks(ax)
    plt.tight_layout()
    save_pdf(fig, 'plot_03_x_world_time.pdf')
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 4: X ego frame vs time ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, ai_x_ego, 'g-', lw=4, alpha=0.9, label='Opponent X (ego frame)')
    ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.5, label='Ego X = 0')
    ax.autoscale()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X ego (m)')
    ax.set_title('Opponent X in Ego Frame vs Time  (+ = ahead)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    bold_ticks(ax)
    plt.tight_layout()
    save_pdf(fig, 'plot_04_x_ego_time.pdf')
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 5: Y vs time ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.add_collection(make_lc(t.values, df['ego_y'].values, seg_colors))
    ax.plot(t, df['ai_y'], 'g-', lw=4, alpha=0.7)
    ax.autoscale()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y world (m)')
    ax.set_title('Y Position vs Time (world frame)')
    ax.legend(handles=legend_world)
    ax.grid(True, alpha=0.3)
    bold_ticks(ax)
    plt.tight_layout()
    save_pdf(fig, 'plot_05_y_world_time.pdf')
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 6: Y ego frame vs time ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, ai_y_ego, 'g-', lw=4, alpha=0.9, label='Opponent Y (ego frame)')
    ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.5, label='Ego Y = 0')
    ax.autoscale()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Y ego (m)')
    ax.set_title('Opponent Y in Ego Frame vs Time  (+ = left, - = right)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    bold_ticks(ax)
    plt.tight_layout()
    save_pdf(fig, 'plot_06_y_ego_time.pdf')
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 7: Ego velocity vs time ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.add_collection(make_lc(t.values, df['ego_velocity'].values, seg_colors))
    ax.autoscale()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Ego Velocity vs Time')
    ax.legend(handles=legend_world[:2])
    ax.grid(True, alpha=0.3)
    bold_ticks(ax)
    plt.tight_layout()
    save_pdf(fig, 'plot_07_velocity_time.pdf')
    pdf.savefig(fig)
    plt.close(fig)

print(f'Saved → {PDF_PATH}')

