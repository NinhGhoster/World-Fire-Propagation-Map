"""
Simple Demo: Moving Firefighter Problem Loader

This shows how easy it is to load and visualize MFP data.
"""

from simple_mfp_loader import SimpleMFPLoader, load_and_plot
import os
import json

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