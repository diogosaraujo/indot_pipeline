"""Map the 2026-08-12 15:00Z bridge flood alerts over Indiana.

Blue  = flow alert corroborated by NWM A&A (data assimilation agrees)
Orange= flow alert from open-loop only (A&A would NOT have fired) -> suspect
Aqua  = precipitation-trigger alert
Gray  = all other monitored bridges (context)
"""
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

SP = r"C:\Users\daraujo\AppData\Local\Temp\claude\c--Users-daraujo-Downloads-indot-pipeline\a79c81f3-50f5-40cc-bd20-ebb8bb016d75\scratchpad"
CFS = 35.3146667
CRS = 26916                                   # NAD83 / UTM 16N — good for Indiana

# palette (dataviz reference instance, slots 1-3 + chrome)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
HAIRLINE, BASELINE = "#e1e0d9", "#c3c2b7"

st = pd.read_parquet(SP + r"\alert_state.parquet")
cfg = pd.read_parquet(SP + r"\cfg.parquet")
nwm = pd.read_parquet(SP + r"\nwm15.parquet")
counties = gpd.read_file(SP + r"\in_counties.gpkg").to_crs(CRS)

m = cfg.merge(nwm, on="comid", how="left")
m["q_ol"] = m["q_ol_cms"] * CFS
m["q_aa"] = m["q_aa_cms"] * CFS
# the threshold that actually gates this bridge: Q10 if scour-critical else Q50
m["fire_thr"] = np.where(m["scour_critical"].to_numpy(bool), m["Q10_cfs"], m["Q50_cfs"])

flow_ids = set(st.loc[st["trigger_type"] == "flow", "bridge_id"])
prcp_ids = set(st.loc[st["trigger_type"] == "precip", "bridge_id"])
m["kind"] = "none"
m.loc[m["bridge_id"].isin(flow_ids) & (m["q_aa"] >= m["fire_thr"]), "kind"] = "confirmed"
m.loc[m["bridge_id"].isin(flow_ids) & ~(m["q_aa"] >= m["fire_thr"]), "kind"] = "openloop"
m.loc[m["bridge_id"].isin(prcp_ids), "kind"] = "precip"

g = gpd.GeoDataFrame(m, geometry=gpd.points_from_xy(m["lon"], m["lat"]), crs=4326).to_crs(CRS)
ctx = g[g["kind"] == "none"]
conf = g[g["kind"] == "confirmed"]
open_ = g[g["kind"] == "openloop"]
prcp = g[g["kind"] == "precip"]
print(f"context {len(ctx)}  confirmed {len(conf)}  open-loop-only {len(open_)}  precip {len(prcp)}")

fig = plt.figure(figsize=(14.6, 9.4), facecolor=SURFACE)
gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1], wspace=0.03,
                      left=0.03, right=0.975, top=0.815, bottom=0.05)
ax_main, ax_zoom = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

ZOOM = (-85.95, 39.25, -84.85, 40.45)          # lon0, lat0, lon1, lat1
zbox = gpd.GeoDataFrame(geometry=gpd.points_from_xy(
    [ZOOM[0], ZOOM[2]], [ZOOM[1], ZOOM[3]]), crs=4326).to_crs(CRS)
zx, zy = zbox.geometry.x.values, zbox.geometry.y.values


def draw(ax, sz_ctx, sz, lw, zoom=False):
    counties.plot(ax=ax, facecolor="#f4f3ef", edgecolor=HAIRLINE, linewidth=0.6, zorder=1)
    counties.dissolve().boundary.plot(ax=ax, color=BASELINE, linewidth=1.6, zorder=2)
    ctx.plot(ax=ax, color=MUTED, markersize=sz_ctx, alpha=0.30, linewidth=0, zorder=3)
    for d, c in ((conf, BLUE), (open_, ORANGE)):
        d.plot(ax=ax, color=c, markersize=sz, edgecolor=SURFACE, linewidth=lw, zorder=4)
    prcp.plot(ax=ax, color=AQUA, markersize=sz * 1.5, marker="^",
              edgecolor=SURFACE, linewidth=lw, zorder=5)
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")


draw(ax_main, 1.4, 46, 0.7)
ax_main.add_patch(plt.Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0],
                                fill=False, edgecolor=INK2, linewidth=1.4,
                                linestyle=(0, (5, 3)), zorder=6))
ax_main.annotate("detail →", xy=(zx[1], zy[1]), xytext=(6, 6),
                 textcoords="offset points", fontsize=11, color=INK2, weight="bold")

draw(ax_zoom, 7, 132, 1.1, zoom=True)
ax_zoom.set_xlim(zx[0], zx[1]); ax_zoom.set_ylim(zy[0], zy[1])
ax_zoom.set_title("Detail — east-central Indiana: two separate basins",
                  fontsize=13, color=INK2, loc="left", pad=10)
for s in ax_zoom.spines.values():
    s.set_visible(True); s.set_color(BASELINE); s.set_linewidth(1.2)

# cluster labels on the zoom panel
for lon, lat, txt in ((-85.62, 40.33, "West Fork White River\nMuncie–Yorktown–Anderson\n→ Wabash → Ohio"),
                      (-85.66, 39.62, "Whitewater River (W. Fork)\nCambridge City–Connersville\n→ Great Miami → Ohio")):
    p = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon], [lat]), crs=4326).to_crs(CRS)
    ax_zoom.annotate(txt, xy=(p.geometry.x[0], p.geometry.y[0]), fontsize=10.5,
                     color=INK, weight="bold", ha="center", va="center", zorder=7,
                     bbox=dict(boxstyle="round,pad=0.34", fc=SURFACE, ec=BASELINE, lw=0.9))

handles = [
    Line2D([], [], marker="o", ls="", mfc=BLUE, mec=SURFACE, mew=1.0, ms=11,
           label=f"Flow — A&A corroborates  ({len(conf)})"),
    Line2D([], [], marker="o", ls="", mfc=ORANGE, mec=SURFACE, mew=1.0, ms=11,
           label=f"Flow — open-loop only  ({len(open_)})"),
    Line2D([], [], marker="^", ls="", mfc=AQUA, mec=SURFACE, mew=1.0, ms=11,
           label=f"Precipitation  ({len(prcp)})"),
    Line2D([], [], marker="o", ls="", mfc=MUTED, mec="none", alpha=0.5, ms=6,
           label=f"Not alerting  ({len(ctx):,})"),
]
fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.028, 0.884),
           ncol=4, frameon=False, fontsize=11.5, handletextpad=0.6,
           columnspacing=2.0)

fig.text(0.03, 0.972, "Indiana bridge flood alerts — first live poller run",
         fontsize=21, weight="bold", color=INK, va="top")
fig.text(0.03, 0.932, "2026-08-12 15:00 UTC  ·  217 bridges alerting on 82 river reaches  ·  "
                      "NWM open-loop streamflow vs retrospective LP3 quantiles",
         fontsize=12.5, color=INK2, va="top")
fig.text(0.975, 0.015, "Counties: US Census TIGER 2022  ·  NAD83 / UTM 16N",
         fontsize=9.5, color=MUTED, ha="right")

out = SP + r"\indiana_bridge_alerts.png"
fig.savefig(out, dpi=170, facecolor=SURFACE)
print("wrote", out)
