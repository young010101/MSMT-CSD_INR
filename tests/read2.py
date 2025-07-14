import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def interactive_viewer(filename):
    img = nib.load(filename)
    data = img.get_fdata()

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)

    # 初始显示
    initial_slice = data.shape[2] // 2
    initial_time = 0
    im = ax.imshow(data[:, :, initial_slice, initial_time], cmap='gray')
    ax.set_title(f'切片: {initial_slice}, 时间点: {initial_time}')

    # 添加滑块
    ax_slice = plt.axes([0.2, 0.1, 0.6, 0.03])
    ax_time = plt.axes([0.2, 0.05, 0.6, 0.03])

    slider_slice = Slider(ax_slice, '切片', 0, data.shape[2] - 1,
                          valinit=initial_slice, valfmt='%d')
    slider_time = Slider(ax_time, '时间点', 0, data.shape[3] - 1,
                         valinit=initial_time, valfmt='%d')

    def update(val):
        slice_idx = int(slider_slice.val)
        time_idx = int(slider_time.val)
        im.set_array(data[:, :, slice_idx, time_idx])
        ax.set_title(f'切片: {slice_idx}, 时间点: {time_idx}')
        fig.canvas.draw()

    slider_slice.on_changed(update)
    slider_time.on_changed(update)

    plt.show()


# 使用
interactive_viewer('../runs/mwu100408_von/grads_test.nii.gz')
