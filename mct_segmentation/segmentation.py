import os
os.environ['SKIMAGE_NO_POOCH'] = '1'
os.environ['POOCH_BASE_DIR'] = r'C:/Raju/Codes/MCT_Analysis/mct_segmentation_package/scikit_cache'

from skimage import data
_ = data.camera()
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import time
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage import filters

class HistogramAnalyzer:
    @staticmethod
    def plot_histogram(image):
        histogram, bins = np.histogram(image.flatten(), bins=1000, range=[image.min(), image.max()])
        fig, ax = plt.subplots()
        ax.plot(bins[:-1], histogram, color='b')
        ax.set_title('Histogram with Peaks')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.set_xlim([image.min(), image.max()])
        plt.show()

    @staticmethod
    def select_thresholds(image):
        histogram, bins = np.histogram(image.flatten(), bins=1000, range=[image.min(), image.max()])
        thresholds = []

        def onclick(event):
            if event.xdata is not None:
                x = event.xdata
                thresholds.append(x)
                print(f"Threshold added: {x}")
                thresholds.sort()

        fig, ax = plt.subplots()
        ax.plot(bins[:-1], histogram, color='b')
        ax.set_title('Histogram with Peaks')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.set_xlim([image.min(), image.max()])
        ax.set_ylim(0, histogram.max())

        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        return thresholds
        
class Segment:

    def __init__(self, image_handler, separate_background, segment_images,do_watershed_3d,threshold_option,thr1,thr2,thr3,thr4,roi,chunksize,ncores):
        #global checkfolder
        self.checkfolder = False
        self.image_handler = image_handler
        self.separate_background = separate_background
        self.segment_images = segment_images
        self.do_watershed_3d = do_watershed_3d
        self.segmented_images = []
        self.threshold_option = threshold_option
        self.thr1=thr1
        self.thr2=thr2
        self.thr3=thr3
        self.thr4=thr4
        self.thresholds = []
        self.chunksize = chunksize
        self.ncores = ncores
        self.volume = []
        self.roi=roi

    def create_masks(self, image):
        masks = []
        if len(self.thresholds) % 2 == 0:
            for i in range(0, len(self.thresholds), 2):
                lt = self.thresholds[i]
                ut = self.thresholds[i + 1]
                mask = np.logical_and(image >= lt, image <= ut).astype(np.uint8)
                masks.append(mask)
        else:
            print("Number of thresholds should be even.")
        return masks

    def watershed_segment(self, image, masks):
        markers = np.zeros_like(image, dtype=int)
        for i, mask in enumerate(masks, start=1):
            markers[mask == 1] = i

        edges = sobel(image)
        segmented_image = np.zeros_like(image, dtype=int)
        local_maxi = peak_local_max(edges, footprint=np.ones((3, 3)), labels=segmented_image)
        markers[local_maxi] = len(masks) + 1
        segmented_image = watershed(edges, markers, mask=image)
        
        return segmented_image



    def remove_border_regions(self, segmented_image):
        segmented_image[:, 0] = 0
        segmented_image[:, -1] = 0
        segmented_image[0, :] = 0
        segmented_image[-1, :] = 0
        return segmented_image

    def separate_background_roi(self, images):
        modified_images = []
        for original_image, segmented_image in zip(images,self.segmented_images):
            img = segmented_image == 2
            filled_image = binary_fill_holes(img)
            background = np.logical_not(filled_image)
            modified_image = np.where(background, original_image.min(), original_image)
            modified_images.append(modified_image)
        images = modified_images.copy()
        return images

    def define_thresholds(self, image):
        print('Histogram')
        analyzer = HistogramAnalyzer()
        self.thresholds = analyzer.select_thresholds(image)


    def process_images(self):
        t1 = time.time()

        num_cores = cpu_count()
        ncores = np.clip(self.ncores, 4, num_cores - 1)

        # Extract individual slices from 3D volume
        input_slices = [self.image_handler.image_volume[i, ...]
                        for i in range(self.image_handler.image_volume.shape[0])]

        if self.separate_background:
            print('Separating background')

            if self.threshold_option == 1:
                print("Using predefined thresholds from thr1, thr2, thr3 and thr4")
                self.thresholds = [self.thr1, self.thr2, self.thr3, self.thr4]

            elif self.threshold_option == 2:
                print("Using Lower Otsu thresholding")
                otsu_thresholds = [filters.threshold_otsu(img) for img in input_slices]
                median_otsu_threshold = np.median(otsu_thresholds)

                min_val = np.min(self.image_handler.image_volume)
                max_val = np.max(self.image_handler.image_volume)

                self.thresholds = [min_val, median_otsu_threshold * 0.9,
                                   median_otsu_threshold * 0.91, max_val]

            else:
                print("Select four threshold points from Histogram")
                self.define_thresholds(input_slices[0])

            print('Selected Thresholds:', self.thresholds)

            self.segmented_images = Parallel(n_jobs=ncores)(
                delayed(process_single_image)(img, self.thresholds)
                for img in tqdm(input_slices, desc="Processing Images")
            )

            self.segmented_images = [np.squeeze(img) for img in self.segmented_images]
            self.volume = np.stack(self.segmented_images, axis=0)
            self.image_volume = self.image_handler.image_volume.copy()

        if self.segment_images:
            print('Segmenting matrix')

            if self.threshold_option == 1:
                print("Using predefined thresholds from thr1, thr2, thr3 and thr4")
                self.thresholds = [self.thr1, self.thr2, self.thr3, self.thr4]

            elif self.threshold_option == 2:
                print("Using Lower Otsu thresholding")
                otsu_thresholds = [filters.threshold_otsu(img) for img in input_slices]
                median_otsu_threshold = np.median(otsu_thresholds)

                min_val = np.min(self.image_handler.image_volume)
                max_val = np.max(self.image_handler.image_volume)

                self.thresholds = [min_val, median_otsu_threshold * 0.94,
                                   median_otsu_threshold * 0.945, max_val]

            else:
                print("Select four threshold points from Histogram")
                self.define_thresholds(input_slices[0])
            print('Selected Thresholds:', [float(t) for t in self.thresholds])
            #print('Selected Thresholds:', self.thresholds)

            if not self.separate_background:
                segmented_images = Parallel(n_jobs=ncores)(
                    delayed(process_single_image)(img, self.thresholds)
                    for img in tqdm(input_slices, desc="Processing Images")
                )

            segmented_images = [np.squeeze(img) for img in segmented_images]
            self.image_handler.label_volume = np.stack(segmented_images, axis=0)

            print(f'The segmented volume is of type: {self.image_handler.label_volume.dtype} with dimensions: {self.image_handler.label_volume.shape}')

        
        if self.do_watershed_3d:                        
            '''
            # Call the watershed segmentation function on the 3D volume again if needed
            self.volume = self.watershed_segmentation_3d(self.image_volume,self.volume).astype(np.int32)  # Process the 3D volume

            print(f'The type of self.volume after 3d watershed is {self.volume.dtype} and shape is {self.volume.shape} and size is {self.volume.size}')
            
            if self.volume is None or (isinstance(self.volume, np.ndarray) and self.volume.size == 0):
                print("No segmented images found.")
                return
            '''
            pass

        print(f'Processing time for segmentation: {(time.time()-t1)/60} mins')
        #print(f'The shape of the segmented images is {self.segmented_images.shape}')
        '''
        unique_values = np.unique(self.segmented_images[0])  
        print(f'Unique values are: {unique_values}')      
        '''
        #self.plot_watershed_result(unique_values)
        # creating the volume of segmented images fo measurements


def process_single_image(image, thresholds):
    # create masks locally
    masks = []
    if len(thresholds) % 2 == 0:
        for i in range(0, len(thresholds), 2):
            lt = thresholds[i]
            ut = thresholds[i + 1]
            mask = np.logical_and(image >= lt, image <= ut).astype(np.uint8)
            masks.append(mask)
    else:
        print("Number of thresholds should be even.")



    markers = np.zeros_like(image, dtype=int)
    for i, mask in enumerate(masks, start=1):
        markers[mask == 1] = i

    edges = sobel(image)
    segmented_image = np.zeros_like(image, dtype=int)
    local_maxi = peak_local_max(edges, footprint=np.ones((3, 3)), labels=segmented_image)
    markers[tuple(local_maxi.T)] = len(masks) + 1
    segmented_image = watershed(edges, markers, mask=image)

    return segmented_image

            
if __name__ == "__main__":
    segment= Segment(image_handler, separate_background, segment_images,do_watershed_3d,threshold_option,thr1,thr2,thr3,thr4,roi,chunksize,ncores)
