# -*- coding: utf-8 -*-
"""Shared plotting helper so every depth panel gets a consistent 'Depth (m)' colorbar."""
import matplotlib.pyplot as plt


def show_depth(ax, depth_map, title=None, fig=None):
    """Draw a depth map on `ax` with its own colorbar labelled 'Depth (m)'."""
    im = ax.imshow(depth_map, cmap="jet")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, loc="left")
    cbar = (fig or plt.gcf()).colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Depth (m)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return im
