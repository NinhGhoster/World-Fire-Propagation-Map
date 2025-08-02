# modules/plotly_visuals.py
import plotly.graph_objects as go

def create_grid_heatmap(grid_data, title, zmin=None, zmax=None, colorscale="Viridis"):
    """Creates a Plotly heatmap figure from a 2D grid."""
    if grid_data is None or grid_data.sum() == 0:
        return go.Figure(go.Scatter(x=[None], y=[None], mode='none')).update_layout(
            title=f"{title}<br>(No Data)", annotations=[dict(text="No data available", showarrow=False)]
        )

    fig = go.Figure(data=go.Heatmap(
        z=grid_data,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
    ))
    fig.update_layout(
        title_text=title,
        xaxis_visible=False,
        yaxis_visible=False,
        # Set a larger, fixed square size
        height=1000,
        width=1000,
        # Re-add the constraint to make the data cells square
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1,
    )
    return fig