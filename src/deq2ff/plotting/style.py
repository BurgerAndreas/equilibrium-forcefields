import matplotlib.pyplot as plt
import seaborn as sns

def prep_plot_style(do_legend=True, xrange=None, legend_framealpha=0, legend_loc=None):
  # A function that sets some simple matplotlib style options re-used in each plot
  ax = plt.gca()

  if do_legend:
    plt.legend(framealpha=legend_framealpha, loc=legend_loc)

  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  plt.tick_params(
    axis='both',        # changes apply to the x-axis
    which='both',       # both major and minor ticks are affected
    bottom=False,       # ticks along the bottom edge are off
    top=False,          # ticks along the top edge are off
    right=False,
    left=False)         # labels along the bottom edge are off

  if xrange:
    ax.set_xticks([xrange[0], (xrange[0] + xrange[1])/2, xrange[1]])
  #plt.tight_layout()


import matplotlib.style as style

# Define custom style parameters
custom_style = {
    'figure.figsize': (8, 6),     # Adjust the figure size as needed
    'font.size': 20,               # Increase font size
    'lines.linewidth': 3.5,        # Thicker lines
    'lines.markersize': 8,        # Thicker lines
    'legend.fontsize': 12,         # Legend font size
    'legend.frameon': True,        # Display legend frame
    'legend.loc': 'upper right',   # Adjust legend position
}

# Apply the custom style
style.use(custom_style)