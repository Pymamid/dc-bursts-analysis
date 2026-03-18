import matplotlib.pyplot as plt
import numpy as np

# Set modern style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

# Data arrays
millisampler_r_values = [13.56, 56.42, 34.47, 25.24, 45.944, 78.67, 16.89, 31.56, 24.48, 29.34]
simulation_results = [1.02, 1.15, 1.52, 0.788, 0.93, 8.78, 6.48, 2.12, 1.8, 0.77]

# Create figure with better proportions
fig, ax = plt.subplots(figsize=(12, 9))
fig.patch.set_facecolor('white')

# Modern color palette
mill_color = '#2E86C1'  # Modern blue
sim_color = '#E74C3C'   # Modern red/coral
edge_color = '#34495E'  # Dark gray for edges

# Strip Plot with Jitter
np.random.seed(42)  # For reproducible jitter
x1 = np.random.normal(1, 0.05, len(millisampler_r_values))
x2 = np.random.normal(2, 0.05, len(simulation_results))

# Plot with improved styling - much larger dots
scatter1 = ax.scatter(x1, millisampler_r_values, alpha=0.85, s=300, color=mill_color, 
                     label='Millisampler', edgecolors=edge_color, linewidth=2, zorder=3)
scatter2 = ax.scatter(x2, simulation_results, alpha=0.85, s=300, color=sim_color, 
                     label='Simulation Results', edgecolors=edge_color, linewidth=2, zorder=3)

# Enhanced customization - much larger fonts
ax.set_xlim(0.3, 2.7)  # Slightly tighter to fill more space
ax.set_xticks([1, 2])
ax.set_xticklabels(['Millisampler', 'Simulation Results'], fontsize=24, fontweight='medium')
ax.set_ylabel('R-values', fontsize=26, fontweight='medium')
ax.tick_params(axis='y', labelsize=22)  # Much larger y-axis numbers

# Improve legend - much larger
legend = ax.legend(fontsize=20, loc='upper right', frameon=True, 
                  fancybox=True, shadow=True, framealpha=0.9)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('#BDC3C7')

# Better grid styling
ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
ax.set_facecolor('#FAFAFA')

# Calculate statistics
mill_mean = np.mean(millisampler_r_values)
mill_std = np.std(millisampler_r_values)
sim_mean = np.mean(simulation_results)
sim_std = np.std(simulation_results)

# Position stats boxes better - make them much larger for paper publication
y_top = ax.get_ylim()[1] * 0.75  # Move higher to use more space

# Millisampler stats - much larger font and padding
mill_text = f'μ = {mill_mean:.1f}\nσ = {mill_std:.1f}\nn = {len(millisampler_r_values)}'
ax.text(0.7, y_top, mill_text, fontsize=22, fontweight='bold',
        bbox=dict(boxstyle="round,pad=1.2", facecolor=mill_color, alpha=0.25,
                 edgecolor=mill_color, linewidth=3),
        ha='center', va='top')

# Simulation stats - much larger font and padding
sim_text = f'μ = {sim_mean:.1f}\nσ = {sim_std:.1f}\nn = {len(simulation_results)}'
ax.text(1.7, y_top, sim_text, fontsize=22, fontweight='bold',
        bbox=dict(boxstyle="round,pad=1.2", facecolor=sim_color, alpha=0.25,
                 edgecolor=sim_color, linewidth=3),
        ha='center', va='top')

# Add subtle median lines - thicker
mill_median = np.median(millisampler_r_values)
sim_median = np.median(simulation_results)
ax.hlines(mill_median, 0.75, 1.25, colors=mill_color, linestyles='dashed', 
          alpha=0.8, linewidth=3, label='_nolegend_')
ax.hlines(sim_median, 1.75, 2.25, colors=sim_color, linestyles='dashed', 
          alpha=0.8, linewidth=3, label='_nolegend_')

# Improve overall appearance
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['left'].set_color('#34495E')
ax.spines['bottom'].set_color('#34495E')

# Adjust layout with tighter spacing to fill more space
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95)

# Save with high quality
plt.savefig('r_values_strip_plot.png', dpi=300, bbox_inches='tight', 
           facecolor='white', edgecolor='none', format='png')
plt.show()

print("✨ Aesthetically enhanced strip plot saved as 'r_values_strip_plot.png'")
print(f"\n📊 Summary Statistics:")
print(f"   Millisampler     - Mean: {mill_mean:.2f}, Std: {mill_std:.2f}, Median: {mill_median:.2f}")
print(f"   Simulation Results - Mean: {sim_mean:.2f}, Std: {sim_std:.2f}, Median: {sim_median:.2f}")
print(f"   Difference in means: {mill_mean - sim_mean:.2f}")