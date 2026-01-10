import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .config import HEATMAP_WIDTH, HEATMAP_HEIGHT

def generate_heatmap(position_data, title, hoop_positions, output_filename, width=HEATMAP_WIDTH, height=HEATMAP_HEIGHT):
    """
    Generate and save a heatmap from position data.
    position_data: list of (x, y) tuples
    """
    if not position_data:
        print(f"No data for {title}")
        return None

    x_coords = [p[0] for p in position_data]
    y_coords = [p[1] for p in position_data]

    # Set up figure
    fig, ax = plt.subplots(figsize=(width/100 + 2, height/100), dpi=100)
    ax.set_facecolor('#232c31')
    fig.patch.set_facecolor('#232c31')

    # Main KDE heatmap
    # Note: Depending on seaborn version, might need to adjust parameters
    # fill=True is for newer versions, shade=True for older. Using fill=True.
    try:
        sns.kdeplot(
            x=x_coords,
            y=y_coords,
            cmap="jet",
            fill=True,
            levels=50,
            alpha=0.95,
            thresh=0,
            ax=ax
        )
    except Exception as e:
        print(f"Error generating heatmap KDE: {e}")
        plt.close()
        return None

    # Draw hoops
    for hoop_pos in hoop_positions:
        ax.scatter(
            hoop_pos[0], hoop_pos[1],
            s=400,
            facecolors='none',
            edgecolors='black',
            linewidths=3,
            marker='o',
            zorder=5
        )

    ax.axis('off')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title, color='white', fontsize=16)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.10)
    
    mappable = plt.cm.ScalarMappable(cmap="jet")
    mappable.set_array([])
    
    cbar = plt.colorbar(mappable, cax=cax)
    cbar.set_label('Coverage', color='white')
    cbar.ax.tick_params(colors='white')
    cbar.outline.set_edgecolor('white')
    cbar.ax.yaxis.label.set_color('white')

    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1, facecolor=fig.get_facecolor())
    plt.close()
    return output_filename

import plotly.graph_objects as go
from scipy.stats import gaussian_kde

def generate_interactive_heatmap(position_data, title, hoop_positions, width=HEATMAP_WIDTH, height=HEATMAP_HEIGHT):
    """
    Generate an interactive Plotly heatmap using KDE.
    """
    if not position_data:
        return None

    x = np.array([p[0] for p in position_data])
    y = np.array([p[1] for p in position_data])

    # 1. Calculate KDE
    # Downsample grid for performance if needed, but 100x100 is usually fine for smoothness
    try:
        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[0:width:100j, 0:height:100j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        
        # Normalize to 0-100% relative intensity
        zi_norm = (zi / zi.max()) * 100
        zi_matrix = zi_norm.reshape(xi.shape).T # Transpose to match plotly's expected orientation for image/heatmap
        
        # Flip Y axis (image coords vs plot coords) - Plotly Heatmap usually requires handling axes carefully
        # In image coords, 0 is top. In plot coords, 0 is bottom. 
        # But we are drawing on a "court". Let's treat (0,0) as top-left if that's how data came in.
        # If openCV data was (x, y) with 0,0 at top-left.
        
    except Exception as e:
        print(f"KDE Error: {e}")
        return None

    fig = go.Figure()

    # Heatmap Layer
    fig.add_trace(go.Heatmap(
        z=zi_matrix,
        x=np.linspace(0, width, 100),
        y=np.linspace(0, height, 100),
        colorscale='Jet',
        zmin=0,
        zmax=100,
        hoverongaps=False,
        hovertemplate='Position: (%{x:.0f}, %{y:.0f})<br>Intensity: %{z:.1f}%<extra></extra>',
        showscale=True,
        colorbar=dict(title='Intensity %')
    ))

    # Hoop Markers
    hoop_x = [p[0] for p in hoop_positions]
    hoop_y = [p[1] for p in hoop_positions]
    
    fig.add_trace(go.Scatter(
        x=hoop_x, 
        y=hoop_y,
        mode='markers',
        marker=dict(size=12, color='white', symbol='circle-open', line=dict(width=2)),
        name='Hoop',
        hoverinfo='skip'
    ))

    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(color='white')),
        width=width,
        height=height,
        paper_bgcolor='#0e1117', # Match Streamlit dark theme
        plot_bgcolor='black',
        xaxis=dict(range=[0, width], showgrid=False, zeroline=False, visible=False),
        # Reverse Y axis to match image coordinates (0 at top)
        yaxis=dict(range=[height, 0], showgrid=False, zeroline=False, visible=False), 
        margin=dict(l=0, r=0, t=30, b=0)
    )

    return fig

def draw_text(frame, text, pos, color=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

def draw_bbox(frame, bbox, color, label=None):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        draw_text(frame, label, (x1, y1 - 10), color=color)

