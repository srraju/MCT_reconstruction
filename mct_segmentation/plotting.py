import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize_with_slider(volume=None, cmap='gray'):
    '''    
    if volume is None:
        volume = self.volume
    assert volume.ndim == 3, "Volume must be 3D"
    '''
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    slice_idx = 0
    img = ax.imshow(volume[slice_idx], cmap=cmap)
    ax.set_title(f"Slice {slice_idx + 1}/{volume.shape[0]}")
    
    # Slider axis and slider object
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, volume.shape[0] - 1, valinit=slice_idx, valstep=1)

    # Update function
    def update(val):
        idx = int(slider.val)
        img.set_data(volume[idx])
        ax.set_title(f"Slice {idx + 1}/{volume.shape[0]}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
