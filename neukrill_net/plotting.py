#!/usr/bin/env python
"""
Misc utilities for plotting easily in the notebook,
often using Holoviews.
"""

def model_weights(model):
    """
    Takes a model and returns the weights as a grid
    of heatmaps using Holoviews.
    """
    # get the weights out of the model
    weights = model.get_weights_topo()

    # make heatmaps using holoviews
    heatmaps = None
    for w in weights:
        w = {(i,j):w[i,j][0] for i in range(w.shape[0]) for j in range(w.shape[1])}
        if heatmaps == None:
            heatmaps = hl.HeatMap(w)
        else:
            heatmaps += hl.HeatMap(w)
    return heatmaps

def monitor_channels(model, *channels, x_axis="example"):
    """
    Takes a model and some strings indicating the channels
    to print and returns a grid of curves plotting the 
    channels. Keyword option to plot against either example,
    batch, epoch or time.
    """
    curves = None
    for c in channels:
        channel = model.monitor.channels[c]
        # holoviews demands capitalisation...
        c = c[0].upper() + c[1:]
        if x_axis == 'example':
            x = channel.example_record
        elif x_axis == 'batch':
            x = channel.batch_record
        elif x_axis == 'epoch':
            x = channel.epoch_record
        elif x_axis == 'time':
            x = channel.time_record
        else:
            raise ValueError("Invalid choice for x_axis: {0}".format(x_axis))
        if not curves:
            curves = hl.Curve(zip(x,channel.value_record))
        else:
            curves += hl.Curve(zip(x,channel.value_record))
    return curves
