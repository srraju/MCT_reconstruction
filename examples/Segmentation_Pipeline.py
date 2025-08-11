from mct_segmentation_package.mct_segmentation.ExtractROI import ExtractROIs
print("Loaded ExtractROIs from:", ExtractROIs.__module__)

from mct_segmentation_package.mct_segmentation.segmentation import Segment
from mct_segmentation_package.mct_segmentation.measures import Measure
class Pipeline:
    def __init__(self,root_dir, image_handler, separate_background, segment_images, do_watershed_3d,
                 threshold_option, thr1, thr2, thr3, thr4, roi, chunksize, ncores,
                 boxsize, extract_images, extract_label_images, X, Y, Z, positions_file_path=None):
        self.root_dir=root_dir
        self.handler=image_handler
        self.roi=roi
        self.segmenter = Segment(
            image_handler=image_handler,
            separate_background=separate_background,
            segment_images=segment_images,
            do_watershed_3d=do_watershed_3d,
            threshold_option=threshold_option,
            thr1=thr1, thr2=thr2, thr3=thr3, thr4=thr4,
            roi=roi,
            chunksize=chunksize,
            ncores=ncores
        )

        self.measurement = Measure(
            #volume=self.segmenter.volume,
            image_handler=self.handler,
            segmenter=self.segmenter,
            roi=roi,
            ncores=ncores,
            chunksize=chunksize,
            checkfolder=self.segmenter.checkfolder,
            outfolder=self.handler.output_folder
            )


        self.extractROI=ExtractROIs(
            self.root_dir,
            self.handler,
            self.segmenter,
            folder_path=image_handler.folder_path,
            label_folder_path=self.handler.label_directory,
            #read_labels=False,
            boxsize=boxsize,
            extract_images=True,
            extract_label_images=False,
            positions_file_path = positions_file_path,X=50,Y=50,Z=50,
            measure=self.measurement

        )


    def run_segmentation(self):
        print("Starting segmentation process...")
        self.segmenter.process_images()

        #self.measurement.volume = self.segmenter.volume

    def save_segmented_data(self,save_image_volume=False, save_image_slices=False,save_segmented_volume=False,save_segmented_slices=False ):

        if save_image_volume:
            self.handler.save_volume(self.handler.image_volume, "images",'tif')
        if save_image_slices:
            self.handler.save_images(self.handler.image_volume, "images")

        if save_segmented_volume:
            self.handler.save_volume(self.handler.label_volume, "labeled_images", 'tif')
        if save_segmented_slices:
            self.handler.save_images(self.handler.label_volume,"labeled_images")

    def run_measurements(self, mode="3d"):
        print("Connected component analysis...")
        if mode == "3d":
            self.measurement.Feret_size_horizontal_vertical_3d_check(self.handler.label_volume,self.handler.image_volume,self.roi)
        elif mode == "2d":
            self.measurement.Feret_size_horizontal_vertical(self.handler.label_volume,self.roi)
        else:
            raise ValueError("Invalid measurement mode. Use '2d' or '3d'.")

    def extractROIs(self):
        self.extractROI.extract_rois()

    def run_all(self):
        self.run_segmentation()
        self.save_segmented_data()
        self.run_measurements("3d")


