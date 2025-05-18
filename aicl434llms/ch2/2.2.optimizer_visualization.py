"""
-- Created by: Ashok Kumar Pant
-- Email: asokpant@gmail.com
-- Created on: 01/05/2025
"""
# pip install streamlit numpy plotly

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# Define loss functions
def himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def quadratic(x, y):
    return x ** 2 + y ** 2


def custom_function(x, y, func_str):
    try:
        return eval(func_str)
    except Exception:
        return np.nan


# Gradient estimation
def compute_gradient(f, x, y, func_str=None):
    eps = 1e-5
    if func_str:
        fx = custom_function(x, y, func_str)
        fx_dx = custom_function(x + eps, y, func_str)
        fx_dy = custom_function(x, y + eps, func_str)
    else:
        fx = f(x, y)
        fx_dx = f(x + eps, y)
        fx_dy = f(x, y + eps)
    dx = (fx_dx - fx) / eps
    dy = (fx_dy - fx) / eps
    return np.array([dx, dy])


# Optimizers
def sgd(grad, pos, lr):
    return pos - lr * grad


def momentum(grad, pos, v, lr, beta=0.9):
    v = beta * v + (1 - beta) * grad
    return pos - lr * v, v


def adagrad(grad, pos, G, lr, eps=1e-8):
    G += grad ** 2
    return pos - lr * grad / (np.sqrt(G) + eps), G


def rmsprop(grad, pos, E, lr, beta=0.9, eps=1e-8):
    E = beta * E + (1 - beta) * grad ** 2
    return pos - lr * grad / (np.sqrt(E) + eps), E


def adam(grad, pos, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    return pos - lr * m_hat / (np.sqrt(v_hat) + eps), m, v


def adamw(grad, pos, m, v, t, lr, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    pos = pos - lr * m_hat / (np.sqrt(v_hat) + eps) - lr * weight_decay * pos
    return pos, m, v


st.set_page_config(layout="wide")

# Sidebar for settings
st.sidebar.title("Optimizer Visualization")
loss_choice = st.sidebar.selectbox("Choose a loss function", ["Quadratic", "Himmelblau", "Rosenbrock", "Custom"])
optimizer_choice = st.sidebar.selectbox("Choose an optimizer",
                                        ["AdamW", "SGD", "Momentum", "Adagrad", "RMSprop", "Adam"], index=0, )
lr = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1, help="Adjust the learning rate for the optimizer")
steps = st.sidebar.slider("Steps", 10, 200, 50, help="Number of optimization steps")
delay = st.sidebar.slider("Step Delay (seconds)", 0.0, 1.0, 0.0, help="Time delay between steps")

if loss_choice == "Custom":
    custom_expr = st.sidebar.text_input("Custom function f(x, y)", value="x**2 + y**2 + np.sin(x*y)")
else:
    custom_expr = None

start_x = st.sidebar.slider("Start x", -6.0, 6.0, -4.0)
start_y = st.sidebar.slider("Start y", -6.0, 6.0, 4.0)

start_button = st.sidebar.button("Start Optimization")

# Define loss function based on user choice
if loss_choice == "Himmelblau":
    loss_fn = himmelblau
elif loss_choice == "Rosenbrock":
    loss_fn = rosenbrock
elif loss_choice == "Quadratic":
    loss_fn = quadratic
else:
    loss_fn = None

if start_button:
    pos = np.array([start_x, start_y])
    v = np.zeros_like(pos)
    G = np.zeros_like(pos)
    E = np.zeros_like(pos)
    m = np.zeros_like(pos)
    v_adam = np.zeros_like(pos)

    path = [pos.copy()]

    X = np.linspace(-6, 6, 200)
    Y = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(X, Y)
    Z = custom_function(X, Y, custom_expr) if loss_fn is None else loss_fn(X, Y)
    z_func = lambda x, y: custom_function(x, y, custom_expr) if loss_fn is None else loss_fn(x, y)

    plot_placeholder = st.empty()
    progress_bar = st.progress(0)

    for t in range(1, steps + 1):
        grad = compute_gradient(loss_fn, *pos, func_str=custom_expr if loss_fn is None else None)

        if optimizer_choice == "SGD":
            pos = sgd(grad, pos, lr)
        elif optimizer_choice == "Momentum":
            pos, v = momentum(grad, pos, v, lr)
        elif optimizer_choice == "Adagrad":
            pos, G = adagrad(grad, pos, G, lr)
        elif optimizer_choice == "RMSprop":
            pos, E = rmsprop(grad, pos, E, lr)
        elif optimizer_choice == "Adam":
            pos, m, v_adam = adam(grad, pos, m, v_adam, t, lr)
        elif optimizer_choice == "AdamW":
            pos, m, v_adam = adamw(grad, pos, m, v_adam, t, lr)

        path.append(pos.copy())
        progress_bar.progress(t / steps)

        fig = go.Figure()
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', opacity=0.6))

        p = np.array(path)
        z_vals = z_func(p[:, 0], p[:, 1])

        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=z_vals,
            mode='lines+markers',
            marker=dict(size=3, color='red'),
            line=dict(color='red', width=3),
            name='Optimization Path'
        ))

        if len(p) > 1:
            for i in range(1, len(p)):
                fig.add_trace(go.Scatter3d(
                    x=[p[i - 1][0], p[i][0]],
                    y=[p[i - 1][1], p[i][1]],
                    z=[z_func(*p[i - 1]), z_func(*p[i])],
                    mode='lines+markers',
                    line=dict(color='orange', width=2, dash='dot'),
                    marker=dict(size=2),
                    showlegend=False
                ))

        fig.update_layout(
            title=f"{optimizer_choice} Optimization (Step {t})",
            scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='Loss'),
            width=1000,
            height=600,
        )

        plot_placeholder.plotly_chart(fig, use_container_width=True)

        time.sleep(delay)

# streamlit run aicl434llms/ch2/2.optimizer_visualization.py
