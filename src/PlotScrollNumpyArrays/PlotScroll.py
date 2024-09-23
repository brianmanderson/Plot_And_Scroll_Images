import copy
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
import SimpleITK as sitk

load_tkag = False
try:
    from ipywidgets import interactive, IntSlider
except:
    load_tkag = True
if load_tkag:
    try:
        matplotlib.use('TkAgg')
    except:
        matplotlib.use('Agg')
    xxx = 1


class scroll_bar_class(object):
    ##  Original code provided by Tucker Netherton (@tnetherton), modified by @cecardenas, modularized by @bmanderson
    def __init__(self, images, title=None):
        self.title = title
        self.selected_images = sitk.GetImageFromArray(images, isVector=False)
        self.size = self.selected_images.GetSize()

    def custom_myshow1(self, img, margin=0.05, dpi=80):
        nda = sitk.GetArrayFromImage(img)
        if nda.ndim == 3:
            # fastest dim, either component or x
            c = nda.shape[-1]

            # the the number of components is 3 or 4 consider it an RGB image
            if not c in (3, 4):
                nda = nda[nda.shape[0] // 2, :, :]

        elif nda.ndim == 4:
            c = nda.shape[-1]

            if not c in (3, 4):
                raise ReferenceError("Unable to show 3D-vector Image")

            # take a z-slice
            nda = nda[nda.shape[0] // 2, :, :, :]

        ysize = nda.shape[1]
        xsize = nda.shape[1]

        # Make a figure big enough to accomodate an axis of xpixels by ypixels
        # as well as the ticklabels, etc...

        plt.close('all')
        plt.clf()

        figsize = ((1 + margin) * ysize / dpi) * 2, (1 + margin) * xsize / dpi
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plt.subplots_adjust(left=0.25, bottom=0.25)

        t = ax.imshow(nda, cmap='gray', interpolation=None)
        ax.set_title('use scroll bar to navigate images')
        plt.show()

    def update(self, Z, view='2D'):
        if view == '2D':
            slices = [self.selected_images[:, :, Z]]
            dpi = 50
        self.custom_myshow1(sitk.Tile(slices, [3, 1]), dpi=dpi, margin=0.05)


def plot_Image_Scroll_Bar_Image(x):
    images = np.squeeze(x)
    if len(images.shape) == 2:
        images = images[None, ...]
    if images.shape[-1] == 3:
        images = images[..., 1]
    if len(images.shape) == 4:
        images = np.argmax(images, axis=-1)
    k = scroll_bar_class(images)
    interactive_plot = interactive(k.update, Z=IntSlider(min=0, max=images.shape[0] - 1))
    output = interactive_plot.children[-1]
    output.layout.height = '600px'
    return interactive_plot


def preprocess_input(x):
    '''
    :param x: numpy array of [rows, columns, # images]
    :return:
    '''
    if x is not None:
        if x.dtype not in ['float32', 'float64', 'int8']:
            x = copy.deepcopy(x).astype('float32')
        if len(x.shape) > 3:
            x = np.squeeze(x)
        if len(x.shape) == 3:
            if x.shape[0] != x.shape[1]:
                x = np.transpose(x, [1, 2, 0])
            elif x.shape[0] == x.shape[2]:
                x = np.transpose(x, [1, 2, 0])
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=-1)

    return x


def plot_scroll_Image(img, mask=None, dose=None, alpha=0.3):
    '''
    :param img: input to view of form [rows, columns, # images] or [#images,512,512,#channels]
    :param mask: binary or int image of masks
    :param dose: float image of the dose
    inputs should be of the same size
    :return:
    '''

    img, mask, dose = map(preprocess_input, [img, mask, dose])

    if mask is not None:
        mask = np.ma.masked_where(mask.astype(np.int8) == 0, mask.astype(np.int8))

    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, img, mask, dose, alpha=alpha)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    return fig, tracker


class IndexTracker(object):
    def __init__(self, ax, img, mask=None, dose=None, alpha=0.3, fontsize=10):
        '''
        :param ax:
        :param img: input to view of form [rows, columns, # images] or [#images,512,512,#channels]
        :param mask: binary or int image of masks
        :param dose: float image of the dose
        :param alpha: transparency level for both mask and dose (0: not visible, 1: opaque)
        :param fontsize: colormap label and tick label size (default: 10)
        '''

        self.ax = ax
        ax.set_title('Use scroll wheel to navigate images')

        self.img = img
        self.mask = mask
        self.dose = dose
        rows, cols, self.slices = img.shape
        self.ind = np.where((np.min(self.img, axis=(0, 1)) != np.max(self.img, axis=(0, 1))))[-1]

        if len(self.ind) > 0:
            self.ind = self.ind[len(self.ind) // 2]
        else:
            self.ind = self.slices // 2
        self.im_img = ax.imshow(self.img[:, :, self.ind], cmap='gray')

        if self.mask is not None:
            mmin = np.min(self.mask)
            mmax = np.max(self.mask)
            LinearSegmentedColormap = matplotlib.colors.LinearSegmentedColormap.from_list
            mask_cm = LinearSegmentedColormap('colormap', plt.cm.Set1(range(mmin, mmax)), mmax)
            self.im_mask = ax.imshow(self.mask[:, :, self.ind], cmap=mask_cm, alpha=alpha, vmin=mmin, vmax=mmax)

        if self.dose is not None:
            self.im_dose = ax.imshow(self.dose[:, :, self.ind], cmap='jet', alpha=alpha)
            dmin = np.min(self.dose)
            dmax = np.max(self.dose)
            sm_dose = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=dmin, vmax=dmax))
            sm_dose.set_array([])
            cbar_dose = plt.colorbar(sm_dose, ticks=np.arange(dmin, dmax + 5, 5))
            cbar_dose.set_label('Dose (Gy)', fontsize=fontsize)
            cbar_dose.ax.tick_params(labelsize=fontsize)

        self.update()

    def onscroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im_img.set_data(self.img[:, :, self.ind])
        if self.mask is not None:
            self.im_mask.set_data(self.mask[:, :, self.ind])
        if self.dose is not None:
            self.im_dose.set_data(self.dose[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im_img.axes.figure.canvas.draw()


if __name__ == '__main__':
    xxx = 1
