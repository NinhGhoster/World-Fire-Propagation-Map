"""
Simple Moving Firefighter Problem Loader

A streamlined loader for MFP data with interactive Plotly visualizations
showing fire spread timesteps and defense actions.

Usage:
    loader = SimpleMFPLoader()
    loader.load('problem.json', 'solution.json')
    loader.plot_interactive_timeline()
"""

import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Tuple, Optional


class SimpleMFPLoader:
    """Simple loader for Moving Firefighter Problem data with Plotly visualization."""
    
    def __init__(self):
        """Initialize the simple loader."""
        self.problem = None
        self.solution = None
        self.graph_data = None
        self.timeline_states = []
    
    def load(self, problem_file: str, solution_file: str = None):
        """
        Load problem and solution data from JSON files.
        
        Args:
            problem_file: Path to problem JSON file
            solution_file: Path to solution JSON file (optional)
        """
        print(f"📂 Loading problem: {problem_file}")
        
        # Load problem
        with open(problem_file, 'r') as f:
            self.problem = json.load(f)
        
        # Extract key info
        self.n = self.problem['parameters']['n']
        self.lambda_d = self.problem['parameters']['lambda_d']
        self.burnt_nodes = self.problem['graph']['burnt_nodes']
        self.adjacency = np.array(self.problem['graph']['adjacency_matrix'])
        self.distances = np.array(self.problem['graph']['distance_matrix'])
        
        print(f"   ✅ Problem loaded: {self.n} vertices, λ={self.lambda_d}, fires={self.burnt_nodes}")
        
        # Load solution if provided
        self.solution = None
        if solution_file:
            try:
                with open(solution_file, 'r') as f:
                    self.solution = json.load(f)
                
                self.defense_sequence = self.solution['solution']['defense_sequence']
                self.objective = self.solution['solution']['objective']
                self.runtime = self.solution['solution']['runtime']
                
                print(f"   ✅ Solution loaded: {len(self.defense_sequence)} actions, {self.objective} burned, {self.runtime:.1f}s")
            except FileNotFoundError:
                print(f"   ⚠️  Solution file not found: {solution_file}")
        
        # Create timeline
        self._create_timeline()
        
        return self
    
    def _create_timeline(self):
        """Create timestep-by-timestep simulation of fire spread and defense."""
        if not self.solution:
            print("   ⚠️  No solution data - creating problem-only timeline")
            self._create_problem_only_timeline()
            return
        
        print("   🔥 Creating fire spread timeline...")
        
        # Initialize state
        current_burning = set(self.burnt_nodes)
        current_defended = set()
        firefighter_pos = self.defense_sequence[0][0]  # Starting position
        
        # Track timeline
        self.timeline_states = []
        time = 0.0
        
        # Initial state
        self.timeline_states.append({
            'time': time,
            'burning': current_burning.copy(),
            'defended': current_defended.copy(),
            'firefighter': firefighter_pos,
            'action': f'Initial: Fire at {list(current_burning)}, firefighter at {firefighter_pos}',
            'newly_burned': set(self.burnt_nodes),
            'newly_defended': set()
        })
        
        # Process defense sequence
        for i, (vertex, burn_round, def_round) in enumerate(self.defense_sequence[1:], 1):
            # Calculate travel time
            travel_time = self.distances[firefighter_pos, vertex]
            time += travel_time
            
            # Update firefighter position and defense
            prev_pos = firefighter_pos
            firefighter_pos = vertex
            newly_defended = {vertex} if vertex not in current_defended and vertex < self.n else set()
            current_defended.update(newly_defended)
            
            # Action description
            if prev_pos == vertex:
                action = f'Reinforce defense at vertex {vertex}'
            else:
                action = f'Move {prev_pos}→{vertex} (travel: {travel_time:.1f}), defend {vertex}'
            
            self.timeline_states.append({
                'time': time,
                'burning': current_burning.copy(),
                'defended': current_defended.copy(),
                'firefighter': firefighter_pos,
                'action': action,
                'newly_burned': set(),
                'newly_defended': newly_defended
            })
        
        # Simulate fire spread at regular intervals
        max_time = max(state['time'] for state in self.timeline_states)
        fire_spread_times = list(range(1, int(max_time) + 3))
        
        for fire_time in fire_spread_times:
            # Calculate fire spread
            newly_burned = set()
            for burning_vertex in current_burning.copy():
                if burning_vertex < self.n:  # Don't spread from anchor
                    for adjacent in range(self.n):
                        if (self.adjacency[burning_vertex, adjacent] == 1 and
                            adjacent not in current_burning and
                            adjacent not in current_defended):
                            newly_burned.add(adjacent)
            
            current_burning.update(newly_burned)
            
            if newly_burned:
                action = f'🔥 Fire spreads to {len(newly_burned)} vertices: {sorted(newly_burned)}'
            else:
                action = f'🔥 Fire contained (no spread)'
            
            self.timeline_states.append({
                'time': float(fire_time),
                'burning': current_burning.copy(),
                'defended': current_defended.copy(),
                'firefighter': firefighter_pos,
                'action': action,
                'newly_burned': newly_burned,
                'newly_defended': set()
            })
        
        # Sort by time
        self.timeline_states.sort(key=lambda x: x['time'])
        
        print(f"   ✅ Timeline created: {len(self.timeline_states)} timesteps")
    
    def _create_problem_only_timeline(self):
        """Create basic timeline for problem without solution."""
        self.timeline_states = [{
            'time': 0.0,
            'burning': set(self.burnt_nodes),
            'defended': set(),
            'firefighter': self.n,  # Start at anchor
            'action': f'Problem setup: {len(self.burnt_nodes)} initial fires',
            'newly_burned': set(self.burnt_nodes),
            'newly_defended': set()
        }]
    
    def get_summary(self) -> Dict:
        """Get a simple summary of the loaded data."""
        summary = {
            'vertices': self.n,
            'initial_fires': len(self.burnt_nodes),
            'lambda': self.lambda_d,
            'edges': int(np.sum(self.adjacency) / 2)
        }
        
        if self.solution:
            summary.update({
                'burned': self.objective,
                'saved': self.n - self.objective,
                'success_rate': (self.n - self.objective) / self.n * 100,
                'runtime': self.runtime,
                'defense_actions': len(self.defense_sequence)
            })
        
        return summary
    
    def plot_interactive_timeline(self, width: int = 900, height: int = 600, save_html: bool = True, filename: str = None):
        """
        Create interactive Plotly visualization of fire timesteps and defense steps.
        
        Args:
            width: Plot width in pixels
            height: Plot height in pixels
            save_html: Whether to save as HTML file
            filename: Custom filename for HTML (optional)
        """
        if not self.timeline_states:
            print("❌ No timeline data available")
            return
        
        print("🎬 Creating interactive timeline visualization...")
        
        # Create graph layout
        G = nx.Graph()
        for i in range(self.n):
            G.add_node(i)
        
        for i in range(self.n):
            for j in range(i+1, self.n):
                if self.adjacency[i, j] == 1:
                    G.add_edge(i, j)
        
        pos = nx.spring_layout(G, seed=42)
        
        # Position anchor vertex
        if hasattr(self, 'distances') and self.distances.size > 0:
            anchor_x = sum(pos[i][0] for i in range(self.n)) / self.n
            anchor_y = min(pos[i][1] for i in range(self.n)) - 0.5
            pos[self.n] = (anchor_x, anchor_y)
        
        # Create frames for animation
        frames = []
        
        for step, state in enumerate(self.timeline_states):
            # Node positions and colors
            node_x = []
            node_y = []
            node_colors = []
            node_text = []
            node_sizes = []
            
            # Add main vertices
            for i in range(self.n):
                node_x.append(pos[i][0])
                node_y.append(pos[i][1])
                
                # Determine color and size based on state
                if i == state['firefighter'] and i < self.n:
                    node_colors.append('blue')
                    node_sizes.append(25)
                    node_text.append(f'🚒 Firefighter V{i}')
                elif i in state['newly_burned']:
                    node_colors.append('orangered')
                    node_sizes.append(22)
                    node_text.append(f'💥 NEW FIRE V{i}')
                elif i in state['burning']:
                    node_colors.append('red')
                    node_sizes.append(20)
                    node_text.append(f'🔥 Burning V{i}')
                elif i in state['newly_defended']:
                    node_colors.append('limegreen')
                    node_sizes.append(22)
                    node_text.append(f'🛡️ NEW DEFENSE V{i}')
                elif i in state['defended']:
                    node_colors.append('green')
                    node_sizes.append(20)
                    node_text.append(f'🛡️ Defended V{i}')
                else:
                    node_colors.append('lightgray')
                    node_sizes.append(18)
                    node_text.append(f'📍 Safe V{i}')
            
            # Add anchor vertex if firefighter is there
            if state['firefighter'] == self.n and self.n in pos:
                node_x.append(pos[self.n][0])
                node_y.append(pos[self.n][1])
                node_colors.append('blue')
                node_sizes.append(25)
                node_text.append(f'🚒 Firefighter (Anchor)')
            
            # Create edges
            edge_x = []
            edge_y = []
            for i in range(self.n):
                for j in range(i+1, self.n):
                    if self.adjacency[i, j] == 1:
                        edge_x.extend([pos[i][0], pos[j][0], None])
                        edge_y.extend([pos[i][1], pos[j][1], None])
            
            # Create frame
            frame_data = []
            
            # Add edges
            frame_data.append(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=1, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))
            
            # Add nodes
            frame_data.append(go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=2, color='black')
                ),
                text=[str(i) for i in range(len(node_x))],
                textposition='middle center',
                textfont=dict(color='white', size=10, family='Arial Black'),
                hovertext=node_text,
                hoverinfo='text',
                showlegend=False
            ))
            
            frames.append(go.Frame(
                data=frame_data,
                name=str(step),
                layout=dict(
                    title=f"Time {state['time']:.1f}: {state['action']}"
                )
            ))
        
        # Create figure
        fig = go.Figure(
            data=frames[0].data if frames else [],
            frames=frames
        )
        
        # Update layout
        fig.update_layout(
            title=f"🔥 Moving Firefighter Problem Timeline<br><sub>n={self.n}, λ={self.lambda_d}, fires={self.burnt_nodes}</sub>",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False, scaleanchor="x"),
            width=width,
            height=height,
            plot_bgcolor='white',
            
            # Add slider
            sliders=[{
                "steps": [
                    {
                        "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}}],
                        "label": f"T={self.timeline_states[i]['time']:.1f}",
                        "method": "animate"
                    }
                    for i in range(len(self.timeline_states))
                ],
                "active": 0,
                "currentvalue": {"prefix": "Timestep: "},
                "pad": {"t": 50},
                "len": 0.9,
                "x": 0.05
            }],
            
            # Add play button
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶️ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True}]
                    },
                    {
                        "label": "⏸️ Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                    }
                ],
                "x": 0.1,
                "y": 0
            }]
        )
        
        # Save as HTML file
        if save_html:
            if filename is None:
                # Generate filename based on problem data
                filename = f"mfp_timeline_n{self.n}_lambda{self.lambda_d}_fires{len(self.burnt_nodes)}.html"
            
            fig.write_html(filename)
            print(f"💾 Interactive timeline saved as: {filename}")
            print("   • Open the HTML file in your browser to view")
            print("   • Use slider to navigate through timesteps")
            print("   • Click Play to auto-animate")
            print("   • Hover over nodes for details")
        else:
            print("🎯 Interactive timeline created!")
            print("   • Use slider to navigate through timesteps")
            print("   • Click Play to auto-animate")
            print("   • Hover over nodes for details")
        
        print("")
        print("🎨 Color Legend:")
        print("   🚒 Blue: Firefighter position")
        print("   💥 Orange-Red: Newly burning")
        print("   🔥 Red: Currently burning")
        print("   🛡️ Lime Green: Newly defended")
        print("   🛡️ Green: Previously defended")
        print("   📍 Gray: Safe vertices")
        
        return fig
    
    def plot_summary_stats(self, save_html: bool = True, filename: str = None):
        """Create a simple summary statistics plot."""
        summary = self.get_summary()
        
        if not self.solution:
            print("📊 Problem summary:")
            for key, value in summary.items():
                print(f"   • {key}: {value}")
            return
        
        # Create bar chart of results
        categories = ['Initial Fires', 'Final Burned', 'Defended', 'Saved']
        values = [
            len(self.burnt_nodes),
            summary['burned'],
            summary['defense_actions'] - 1,  # Exclude starting position
            summary['saved']
        ]
        colors = ['red', 'darkred', 'green', 'lightgreen']
        
        fig = go.Figure(data=[
            go.Bar(x=categories, y=values, marker_color=colors, text=values, textposition='auto')
        ])
        
        fig.update_layout(
            title=f"📊 Problem Results Summary<br><sub>Success Rate: {summary['success_rate']:.1f}%, Runtime: {summary['runtime']:.1f}s</sub>",
            yaxis_title="Number of Vertices",
            showlegend=False
        )
        
        # Save as HTML file
        if save_html:
            if filename is None:
                # Generate filename based on problem data
                filename = f"mfp_summary_n{self.n}_lambda{self.lambda_d}_fires{len(self.burnt_nodes)}.html"
            
            fig.write_html(filename)
            print(f"💾 Summary statistics saved as: {filename}")
            print(f"📈 Summary: {summary['success_rate']:.1f}% success rate, {summary['runtime']:.1f}s runtime")
        else:
            print(f"📈 Summary: {summary['success_rate']:.1f}% success rate, {summary['runtime']:.1f}s runtime")
        
        return fig


# Simple usage functions
def load_and_plot(problem_file: str, solution_file: str = None, save_html: bool = True):
    """
    Simple function to load data and create interactive plot in one call.
    
    Args:
        problem_file: Path to problem JSON
        solution_file: Path to solution JSON (optional)
        save_html: Whether to save plots as HTML files
    
    Returns:
        SimpleMFPLoader instance with data loaded
    """
    loader = SimpleMFPLoader()
    loader.load(problem_file, solution_file)
    
    # Show summary
    summary = loader.get_summary()
    print("\n📊 SUMMARY:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"   • {key}: {value:.2f}")
        else:
            print(f"   • {key}: {value}")
    
    # Create plots
    timeline_fig = loader.plot_interactive_timeline(save_html=save_html)
    
    if loader.solution:
        stats_fig = loader.plot_summary_stats(save_html=save_html)
        return loader, timeline_fig, stats_fig
    
    return loader, timeline_fig

import os

# Example usage
def is_valid_json(filename):
    """Check if a JSON file is valid and complete."""
    try:
        with open(filename, 'r') as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, FileNotFoundError):
        return False

def main():
    print("🔥 Simple MFP Data Loader Demo")
    print("=" * 40)
    
    # Find available data files
    all_problem_files = [f for f in os.listdir('.') if f.endswith('_problem.json')]
    
    # Filter for valid JSON files
    problem_files = [f for f in all_problem_files if is_valid_json(f)]
    
    if not problem_files:
        print("❌ No valid problem files found in current directory")
        if all_problem_files:
            print(f"   Found {len(all_problem_files)} problem files, but they contain errors")
            print(f"   Files found: {all_problem_files}")
        else:
            print("   Run the Moving Firefighter notebook first to save some data!")
        return
    
    # Use the first valid file
    problem_file = problem_files[0]
    solution_file = problem_file.replace('_problem.json', '_solution.json')
    
    print(f"📂 Found valid data files:")
    print(f"   Problem: {problem_file}")
    print(f"   Solution: {solution_file if os.path.exists(solution_file) else 'Not found'}")
    if len(all_problem_files) > len(problem_files):
        invalid_files = [f for f in all_problem_files if f not in problem_files]
        print(f"   ⚠️  Skipped {len(invalid_files)} invalid files: {invalid_files}")
    print()
    
    # Method 1: Step by step
    print("🔧 METHOD 1: Step by step loading")
    print("-" * 30)
    
    loader = SimpleMFPLoader()
    loader.load(problem_file, solution_file if os.path.exists(solution_file) else None)
    
    # Show summary
    summary = loader.get_summary()
    print("\n📊 Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"   • {key}: {value:.1f}")
        else:
            print(f"   • {key}: {value}")
    
    # Create interactive timeline
    print("\n🎬 Creating interactive timeline...")
    timeline_fig = loader.plot_interactive_timeline(save_html=True)
    
    # Create summary stats if solution available
    if loader.solution:
        print("\n📈 Creating summary statistics...")
        stats_fig = loader.plot_summary_stats(save_html=True)
    
    print("\n" + "=" * 40)
    
    # Method 2: One-liner (if we have multiple files)
    if len(problem_files) > 1:
        print("🚀 METHOD 2: One-liner loading")
        print("-" * 30)
        
        # Use second file if available
        problem_file2 = problem_files[1]
        solution_file2 = problem_file2.replace('_problem.json', '_solution.json')
        
        print(f"📂 Loading: {problem_file2}")
        
        # Load and plot in one call
        result = load_and_plot(problem_file2, solution_file2 if os.path.exists(solution_file2) else None, save_html=True)
        
        if len(result) == 3:
            loader2, timeline_fig2, stats_fig2 = result
            print("   ✅ Second set of plots saved as HTML files")
        else:
            loader2, timeline_fig2 = result
            print("   ✅ Second timeline saved as HTML file")
    
    print("\n✅ Demo complete!")
    print("   The interactive plots have been saved as HTML files.")
    print("   Open the HTML files in your browser to view the visualizations.")
    print("   Use the slider to see fire spread and defense actions over time!")

if __name__ == "__main__":
    main()