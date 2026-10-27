import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Linear Regression with Outliers", layout="centered")

MIN, MAX = 0.0, 10.0  # both axes run from 0 to 10


def shift_scale(a):
    """Shift up by 2, then scale so the mean is exactly 5."""
    shifted = np.asarray(a, dtype=float) + 2
    return shifted * (5 / shifted.mean())


def make_original_data():
    """The same ten points every time (r = 0.900)."""
    raw_x = [1.55, 3.53, 4.20, 5.71, 7.13, 8.48, 8.67, 8.89, 9.36, 9.57]
    raw_y = [2.33, 1.68, 4.15, 3.76, 7.22, 7.01, 5.41, 7.64, 6.86, 7.64]
    return shift_scale(raw_x), shift_scale(raw_y)


def regression_stats(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return {"slope": slope, "intercept": intercept, "r": r, "mx": x.mean(), "my": y.mean()}


# ---------- state ----------
if "orig" not in st.session_state:
    st.session_state.orig = make_original_data()
    st.session_state.new_point = None
    st.session_state.chart_version = 0

x0, y0 = st.session_state.orig
new = st.session_state.new_point

st.title("Linear Regression with Outliers")
st.write(
    "These ten points have a correlation of r = 0.90. "
    "Click the graph to add a new point and see how the regression line changes."
)

# ---------- stats ----------
s0 = regression_stats(x0, y0)
if new:
    x_all, y_all = np.append(x0, new[0]), np.append(y0, new[1])
    s1 = regression_stats(x_all, y_all)

# ---------- figure ----------
fig = go.Figure()

# Invisible grid of points: lets us capture clicks on empty space.
step = 0.2
g = np.arange(MIN, MAX + step / 2, step)
gx, gy = np.meshgrid(g, g)
fig.add_trace(
    go.Scatter(
        x=gx.ravel(), y=gy.ravel(), mode="markers",
        marker=dict(size=10, opacity=0.01, color="gray"),
        hoverinfo="none", showlegend=False, name="click-grid",
    )
)

xline = np.array([MIN, MAX])
BLUE, RED = "#1f77b4", "#d62728"

fig.add_trace(go.Scatter(x=xline, y=s0["slope"] * xline + s0["intercept"], mode="lines",
                         line=dict(color=BLUE, width=3), name="Original line", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=x0, y=y0, mode="markers",
                         marker=dict(size=10, color=BLUE),
                         name="Original 10 points", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=[s0["mx"]], y=[s0["my"]], mode="markers",
                         marker=dict(size=20, color=BLUE, symbol="circle-cross-open", line=dict(width=3)),
                         name="Original mean (x̄, ȳ)", hoverinfo="skip"))

if new:
    fig.add_trace(go.Scatter(x=xline, y=s1["slope"] * xline + s1["intercept"], mode="lines",
                             line=dict(color=RED, width=3, dash="dash"),
                             name="New line", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[new[0]], y=[new[1]], mode="markers",
                             marker=dict(size=14, color=RED, symbol="diamond",
                                         line=dict(width=1, color="black")),
                             name="New point", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[s1["mx"]], y=[s1["my"]], mode="markers",
                             marker=dict(size=28, color=RED, symbol="circle-cross-open", line=dict(width=3)),
                             name="New mean (x̄, ȳ)", hoverinfo="skip"))

fig.update_layout(
    # constrain="domain" keeps both ranges locked at 0-10 (the plot area shrinks to stay square
    # instead of Plotly stretching the x range to fit the page width).
    xaxis=dict(range=[MIN, MAX], autorange=False, title="x", fixedrange=True, dtick=2,
               constrain="domain"),
    yaxis=dict(range=[MIN, MAX], autorange=False, title="y", fixedrange=True, dtick=2,
               scaleanchor="x", constrain="domain"),
    height=600, dragmode=False, margin=dict(l=10, r=10, t=10, b=10),
    # Three entries per row: the original items on the first row, the new ones on the second.
    legend=dict(orientation="h", y=-0.12, entrywidthmode="fraction", entrywidth=0.33),
)

event = st.plotly_chart(
    fig,
    key=f"chart_{st.session_state.chart_version}",
    on_select="rerun",
    selection_mode="points",
    use_container_width=True,
)

# ---------- handle a click ----------
pts = event.selection.points if event and event.selection else []
if pts:
    p = pts[0]
    st.session_state.new_point = (float(p["x"]), float(p["y"]))
    st.session_state.chart_version += 1   # fresh chart => selection is cleared
    st.rerun()

if st.button("Clear new point", disabled=new is None):
    st.session_state.new_point = None
    st.session_state.chart_version += 1
    st.rerun()

# ---------- results table ----------
rows = [("Slope", "slope"), ("Y-intercept", "intercept"), ("r", "r"),
        ("Mean x (x̄)", "mx"), ("Mean y (ȳ)", "my")]

lines = [
    "| | **Original** (n = 10) | **With new point** (n = 11) | **Change** |",
    "|:--|--:|--:|--:|",
]
for label, key in rows:
    if new:
        lines.append(f"| {label} | {s0[key]:.3f} | {s1[key]:.3f} | {s1[key] - s0[key]:+.3f} |")
    else:
        lines.append(f"| {label} | {s0[key]:.3f} | — | — |")
st.markdown("\n".join(lines))

if new:
    st.caption(f"New point at ({new[0]:.2f}, {new[1]:.2f})")
else:
    st.caption("Click the graph to add a point.")
