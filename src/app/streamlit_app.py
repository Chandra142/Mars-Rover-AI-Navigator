"""Streamlit demo:  streamlit run src/app/streamlit_app.py

Upload a Mars rover image -> segmentation -> cost map -> planned route.
Assumption: the ground-level image is treated as an approximate 2D grid; the rover
starts at the bottom-center and the goal is on the top row (chosen with a slider).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import streamlit as st

from src.evaluation.evaluate import PALETTE
from src.planning.cost_map import class_grid_to_cost, load_costs, mask_to_grid
from src.planning.planners import astar, dijkstra, nearest_free, path_metrics, straight_line

IMG = 256
st.set_page_config(page_title="Mars Rover AI Navigator", layout="wide")
st.title("Mars Rover AI Navigator")


@st.cache_resource
def load_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


def synthetic_mask():
    """Fake terrain so the planner can be tried without a trained model."""
    m = np.zeros((IMG, IMG), np.uint8)
    m[:, :60] = 1
    m[100:160, 90:200] = 2
    cv2.circle(m, (128, 70), 28, 3, -1)
    cv2.circle(m, (190, 190), 20, 3, -1)
    return m


def draw_path(img, path, grid, color, thick=2):
    pts = np.array([[(c + 0.5) * IMG / grid, (r + 0.5) * IMG / grid] for r, c in path], np.int32)
    return cv2.polylines(img, [pts], False, color, thick, cv2.LINE_AA)


default_costs, default_grid, default_inflate = load_costs()
with st.sidebar:
    st.header("Input")
    model_path = st.text_input("Model file", "models/mobilenet.keras")
    demo = st.checkbox("Use synthetic demo terrain (no model needed)", value=False)
    up = st.file_uploader("Mars image", type=["jpg", "jpeg", "png"])
    st.header("Terrain costs")
    costs = {k: st.slider(k, 0.5, 20.0, float(v or 1.0), 0.5)
             for k, v in default_costs.items() if v is not None}
    costs["big_rock"] = None if st.checkbox("Block big rocks", True) else st.slider("big_rock", 5.0, 50.0, 30.0)
    inflate = st.slider("Safety margin around blocked cells", 0, 4, default_inflate)
    st.header("Planner")
    algo = st.selectbox("Algorithm", ["A*", "Dijkstra"])
    goal_pct = st.slider("Goal column (left to right, %)", 0, 100, 50)

if demo:
    image = np.full((IMG, IMG, 3), 140, np.uint8)
    mask = synthetic_mask()
elif up is not None and os.path.exists(model_path):
    arr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(cv2.resize(arr, (IMG, IMG)), cv2.COLOR_BGR2RGB)
    mask = np.argmax(load_model(model_path).predict(image[None].astype("float32"), verbose=0)[0], -1)
else:
    st.info("Upload an image and point to a trained model file, or tick the synthetic demo.")
    st.stop()

grid = default_grid
class_grid = mask_to_grid(mask, grid)
cost = class_grid_to_cost(class_grid, costs, inflate)
start = nearest_free(cost, (grid - 1, grid // 2))
goal = nearest_free(cost, (0, int(goal_pct / 100 * (grid - 1))))
path = (astar if algo == "A*" else dijkstra)(cost, start, goal)
base = straight_line(start, goal)

overlay = cv2.addWeighted(image, 0.5, PALETTE[np.where(mask > 3, 4, mask)], 0.5, 0)
route = overlay.copy()
route = draw_path(route, base, grid, (255, 255, 255), 1)          # white = straight line
if path:
    route = draw_path(route, path, grid, (0, 200, 255), 2)         # cyan = planned route
cost_vis = np.where(np.isfinite(cost), cost, cost[np.isfinite(cost)].max() * 1.3)
cost_img = cv2.applyColorMap(cv2.resize((255 * cost_vis / cost_vis.max()).astype(np.uint8), (IMG, IMG),
                                        interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_INFERNO)

c1, c2, c3, c4 = st.columns(4)
c1.image(image, caption="Input", use_container_width=True)
c2.image(overlay, caption="Segmentation (brown soil, gray bedrock, yellow sand, red rock)", use_container_width=True)
c3.image(cv2.cvtColor(cost_img, cv2.COLOR_BGR2RGB), caption="Cost map (bright = costly)", use_container_width=True)
c4.image(route, caption=f"Route: {algo} (cyan) vs straight line (white)", use_container_width=True)

if path is None:
    st.error("No safe route exists with the current costs. Lower the safety margin or unblock rocks.")
else:
    st.subheader("Route metrics")
    st.table({"Route": [algo, "Straight line"],
              **{k: [path_metrics(p, cost, class_grid)[k] for p in (path, base)]
                 for k in ["total_cost", "length", "pct_safe", "hazard_cells", "blocked_cells"]}})
