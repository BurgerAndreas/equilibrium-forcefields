import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, pathlib

# andreas-burger/EquilibriumEquiFormer
entity = "andreas-burger"
project = "EquilibriumEquiFormer"

# parent folder of the plot
plotfolder = pathlib.Path(__file__).parent.absolute()
plotfolder = os.path.join(plotfolder, "plots")

def set_seaborn_style(style="poster", figsize=(8, 6), font_scale=0.8):
    sns.set_style(style="whitegrid") # whitegrid white
    sns.set_context(
        style, # {paper, notebook, talk, poster}
        font_scale=font_scale,
        rc={
            'figure.figsize': figsize,     # Adjust the figure size as needed
            'font.size': 20,               # Increase font size
            'lines.linewidth': 3.5,        # Thicker lines
            'lines.markersize': 8,        # Thicker lines
            'legend.fontsize': 12,         # Legend font size
            'legend.frameon': True,        # Display legend frame
            'legend.loc': 'upper right',   # Adjust legend position
            "axes.spines.right": False, "axes.spines.top": False
        },
    )

def combine_legend(ax, colorstyle_dict, markerstyle):
    # https://stackoverflow.com/questions/68591271/how-can-i-combine-hue-and-style-groups-in-a-seaborn-legend

    # seaborn used to have the legend title as element 0, but it seems to have been removed
    offset = 1

    # create a dictionary mapping the colorstyle to their color
    handles, labels = ax.get_legend_handles_labels()
    index_item_title = labels.index(markerstyle)
    color_dict = {
        label: handle.get_color()
        for handle, label in zip(handles[offset:index_item_title], labels[offset:index_item_title])
    }
    print('color_dict:', color_dict)

    # loop through the items, assign color via the colorstyle of the item
    for handle, label in zip(handles[index_item_title + offset:], labels[index_item_title + offset:]):
        handle.set_color(color_dict[colorstyle_dict[label]])

    # create a legend only using the items
    ax.legend(
        handles=handles[index_item_title + 1:], 
        labels=labels[index_item_title + 1:], 
        # title='Item',
        # bbox_to_anchor=(1.03, 1.02), fontsize=10
    )


# Kevin Xi

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

if __name__ == "__main__":
    # Apply the custom style
    style.use(custom_style)