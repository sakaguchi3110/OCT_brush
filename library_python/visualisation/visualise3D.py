import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

def visualiseVolume(data, hFig, hAxes, minThreshold, maxThreshold, title_base):
    hAxes.clear()

    x, y, z = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]), np.arange(data.shape[2]), indexing='ij')
    mask = (minThreshold <= data) & (data <= maxThreshold)

    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
    values_filtered = data[mask]

    scatter = hAxes.scatter(x_filtered, y_filtered, z_filtered, c=values_filtered, cmap='turbo', marker='o')

    hFig.colorbar(scatter, ax=hAxes, shrink=0.5, aspect=5)
    hAxes.set_xlim([0, data.shape[0]])
    hAxes.set_ylim([0, data.shape[1]])
    hAxes.set_zlim([0, data.shape[2]])

    if title_base:
        title = f'{title_base}: Threshold Range: [{minThreshold:.2f}, {maxThreshold:.2f}]'
    else:
        title = f'Threshold Range: [{minThreshold:.2f}, {maxThreshold:.2f}]'
    hAxes.set_title(title)

    plt.draw()
    

def visualise3D(data, hFig=None, title_base=""):
    if data.ndim != 3:
        raise ValueError('Input must be a 3D matrix.')

    if hFig is None:
        hFig = plt.figure()
    hFig.suptitle('3D Volume Visualiser', fontsize=16)

    minValue = np.min(data)
    maxValue = np.max(data)

    hAxes = hFig.add_subplot(111, projection='3d')

    minThreshold = minValue
    maxThreshold = maxValue
    visualiseVolume(data, hFig, hAxes, minThreshold, maxThreshold, title_base)

    axcolor = 'lightgoldenrodyellow'
    axmin = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axmax = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    slider_min = Slider(axmin, 'Min Threshold', minValue, maxValue, valinit=minValue)
    slider_max = Slider(axmax, 'Max Threshold', minValue, maxValue, valinit=maxValue)

    def update(val):
        minThreshold = slider_min.val
        maxThreshold = slider_max.val
        if minThreshold >= maxThreshold:
            maxThreshold = minThreshold + 0.01
            slider_max.set_val(maxThreshold)
        visualiseVolume(data, hFig, hAxes, minThreshold, maxThreshold, title_base)

    slider_min.on_changed(update)
    slider_max.on_changed(update)

    plt.show()