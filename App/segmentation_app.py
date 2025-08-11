import tkinter as tk
from tkinter import filedialog, messagebox
from mct_segmentation_package.mct_segmentation.plotting import *
from mct_segmentation_package.mct_segmentation.ExtractROI import *
from mct_segmentation_package.mct_segmentation.segmentation import Segment
#from mct_segmentation_package.mct_segmentation.ExtractROI import ExtractROIs 
from mct_segmentation_package.mct_segmentation.measures import Measure
from mct_segmentation_package.IO.IO import ImageHandler


class Segmenter:
    def __init__(self, root,segment):
        self.root = root
        self.segment = segment
        self.image_handler = None
        self.folder_path = ''
        self.file_names = []
        self.separate_background = tk.BooleanVar()
        self.segment_images = tk.BooleanVar()
        self.do_watershed_3d = tk.BooleanVar()
        self.roi_positions = []
        self.ImagesLabel = ''
        self.choice_var = tk.IntVar(value=1)
        self.properties_2d = tk.BooleanVar()
        self.properties_3d = tk.BooleanVar()
        self.save_2d_slices = tk.BooleanVar()
        self.save_3d_stack = tk.BooleanVar()
        self.threshold_option = tk.BooleanVar()


        self.setup_gui()
        #self.image_handler = ImageHandler(folder_path)
        self.positionsdata=None
        self.startprocessing.config(state='normal')
        self.plot_frame = None

    def setup_gui(self):
        self.root.title('Image Processing Tool_V11')
        self.root.geometry("1000x1000")
        
        # LabelFrame for file selection and processing options
        file_frame = tk.LabelFrame(self.root, text="File Selection & Processing")
        file_frame.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

        # File selection
        tk.Label(file_frame, text='Select files:').grid(row=0, column=0, sticky=tk.W)
        self.ImagesLabel = tk.Label(file_frame, text=' ')
        self.ImagesLabel.grid(row=0, column=1, sticky=tk.W)
        tk.Button(file_frame, text='Browse', command=ImageHandler.select_files).grid(row=0, column=2, padx=5, pady=5)
        # Entry box from start and end
        self.StartindexLabel = tk.Label(file_frame, text='Start index:')
        self.StartindexLabel.grid(row=0, column=3, sticky=tk.W)
        
        self.start_index = tk.Entry(file_frame,width=15)
        self.start_index.grid(row=0,column=4,sticky=tk.W)
        self.start_index.insert(0, "0")  # Default value
        self.EndindexLabel = tk.Label(file_frame, text='End index:')
        self.EndindexLabel.grid(row=0, column=5, sticky=tk.W)
        
        self.end_index = tk.Entry(file_frame,width=15)
        self.end_index.grid(row=0,column=6,sticky=tk.W)
        self.end_index.insert(0, "100")  # Default value
        self.thr1 = tk.DoubleVar(value=22000)
        self.thr2 = tk.DoubleVar(value = 24000)
        self.thr3 = tk.DoubleVar(value = 24500)
        self.thr4 = tk.DoubleVar(value = 27000)

        # Background separation
        tk.Checkbutton(file_frame, text='Separate Background (To remove background from slices, \n do not use in conjunction with Segment images.)', variable=self.separate_background).grid(row=1, column=1, sticky=tk.W,columnspan=20)

        # Image segmentation
        tk.Checkbutton(file_frame, text='Segment Images', variable=self.segment_images).grid(row=2, column=1, sticky=tk.W,columnspan=4)
        # Image 3d segmentation
        tk.Checkbutton(file_frame, text='3d Segment Images', variable=self.do_watershed_3d).grid(row=3, column=1, sticky=tk.W,columnspan=4)        
        
        tk.Label(file_frame, text='Chunk Size').grid(row=4, column=0, sticky=tk.W)
        self.chunksize_entry_box = tk.Entry(file_frame,width=15)
        self.chunksize_entry_box.grid(row=4,column=1,sticky=tk.W)
        self.chunksize_entry_box.insert(0, "100")  # Default value

        tk.Label(file_frame, text='ncores').grid(row=4, column=2, sticky=tk.W)
        self.ncores_entry_box = tk.Entry(file_frame,width=15)
        self.ncores_entry_box.grid(row=4,column=3,sticky=tk.W)
        self.ncores_entry_box.insert(0, "20")  # Default value

        threshold_options_frame = tk.LabelFrame(file_frame, text="Threshold options")
        threshold_options_frame.grid(row=5, column=0, padx=10, pady=10, columnspan=3,sticky=tk.W)

        # Variable to store the selected threshold option
        self.threshold_option = tk.IntVar(value=3)  # Default option: Use predefined threshold
        
        # Option 1: Use predefined threshold
        tk.Radiobutton(threshold_options_frame, text='Use predefined threshold', variable=self.threshold_option, value=1, command=self.enable_threshold_entries).grid(row=0, column=0, columnspan=3,sticky=tk.W)
        self.thr1Entry=tk.Entry(threshold_options_frame, textvariable=self.thr1,width=10,state='disabled')
        self.thr1Entry.grid(row=1, column=0)
        self.thr2Entry=tk.Entry(threshold_options_frame, textvariable=self.thr2,width=10,state='disabled')
        self.thr2Entry.grid(row=1, column=1)
        self.thr3Entry=tk.Entry(threshold_options_frame, textvariable=self.thr3,width=10,state='disabled')
        self.thr3Entry.grid(row=1, column=2)
        self.thr4Entry=tk.Entry(threshold_options_frame, textvariable=self.thr4,width=10,state='disabled')
        self.thr4Entry.grid(row=1, column=3)        
        
        # Option 2: From lower Otsu
        tk.Radiobutton(threshold_options_frame, text='From lower Otsu', variable=self.threshold_option, value=2, command=self.enable_threshold_entries).grid(row=2, column=0, sticky=tk.W, columnspan=2)
        
        # Option 3: Choose from histogram
        tk.Radiobutton(threshold_options_frame, text='Choose from histogram', variable=self.threshold_option, value=3, command=self.enable_threshold_entries).grid(row=3, column=0, sticky=tk.W, columnspan=4)


        # LabelFrame for Property Measurements
        Measurement_frame = tk.LabelFrame(file_frame, text="Measures")
        Measurement_frame.grid(row=5, column=3, padx=10, pady=10, sticky=tk.W)

        # 2D measures
        tk.Checkbutton(Measurement_frame, text='slice measures', variable=self.properties_2d).grid(row=0, column=0, sticky=tk.W)
        self.Measure_2d_Button = tk.Button(Measurement_frame, text='Measure', command = Measure.Feret_size_horizontal_vertical_Calc)
        self.Measure_2d_Button.grid(row=0, column=1,  pady=5)
        self.Measure_2d_Button.config(state='disabled')

        # 3D measures
        tk.Checkbutton(Measurement_frame, text='3d measures', variable=self.properties_3d).grid(row=1, column=0, sticky=tk.W)
        self.Measure_3d_Button = tk.Button(Measurement_frame, text='Measure', command = Measure.Feret_size_horizontal_vertical_3d_Calc)
        self.Measure_3d_Button.grid(row=1, column=1, pady=5)
        self.Measure_3d_Button.config(state='disabled')

        # LabelFrame for saving data
        Save_frame = tk.LabelFrame(file_frame, text="Save")
        Save_frame.grid(row=5, column=42, padx=10, pady=10, sticky=tk.W)
        # save 2d segmented slices
        tk.Checkbutton(Save_frame, text='Segmented slices', variable=self.save_2d_slices).grid(row=0, column=0, sticky=tk.W)
        self.Save_Seg_slices_Button = tk.Button(Save_frame, text='Save', command = Measure.save_segmented_slices)
        self.Save_Seg_slices_Button.grid(row=0, column=1,  pady=5)
        self.Save_Seg_slices_Button.config(state='disabled')

        # 2D measures
        tk.Checkbutton(Save_frame, text='Segmented stack', variable=self.save_3d_stack).grid(row=1, column=0, sticky=tk.W)
        self.Save_Seg_stack_Button = tk.Button(Save_frame, text='Save', command = Measure.save_segmented_stack)
        self.Save_Seg_stack_Button.grid(row=1, column=1,  pady=5)
        self.Save_Seg_stack_Button.config(state='disabled')


        # Start button
        self.startprocessing=tk.Button(file_frame, text='Start Processing', command=self.start_processing_all, font=('Arial', 15) )
        self.startprocessing.grid(row=8, column=0, pady=10)

        # Exit button
        exit_button = tk.Button(file_frame, text="Exit", command=self.quit, font=('Arial', 15) )
        exit_button.grid(row=8, column=1, padx=10, pady=10)


        plot_frame = tk.LabelFrame(file_frame, text="Plotting options")
        plot_frame.grid(row=9, column=0,columnspan = 9, padx=10, pady=10, sticky=tk.W)
        
        tk.Label(plot_frame, text='Plot input image',width=50).grid(row=0, column=0, sticky=tk.W)
        self.plot_image_entry = tk.Entry(plot_frame,width=15)
        self.plot_image_entry.grid(row=0,column=1,sticky=tk.W)
        self.plot_image_entry.insert(0, "0")  # Default value
        self.plotimagebutton=tk.Button(plot_frame, text="Plot input image", command=lambda:plot_image(self))
        self.plotimagebutton.grid(row=0, column=2, pady=5)
        self.plotimagebutton.config(state='disabled')


        tk.Label(plot_frame, text='Plot Watershed segmented image',width = 50).grid(row=1, column=0, sticky=tk.W)
        self.plot_watershed_entry = tk.Entry(plot_frame,width=15)
        self.plot_watershed_entry.grid(row=1,column=1,sticky=tk.W)
        self.plot_watershed_entry.insert(0, "0")  # Default value
        self.plotwatershedbutton=tk.Button(plot_frame, text="Plot Watershed", command=lambda:plot_watershed(self))
        self.plotwatershedbutton.grid(row=1, column=2, pady=5)
        self.plotwatershedbutton.config(state='disabled')

        # Creating an instance of ExtractROIs 
        self.erois=ExtractROIs(root,self.segment)

    def enable_threshold_entries(self):
        if self.threshold_option.get() == 1:
            self.thr1Entry.config(state='normal')
            self.thr2Entry.config(state='normal')
            self.thr3Entry.config(state='normal')
            self.thr4Entry.config(state='normal')
        else:
            self.thr1Entry.config(state='disabled')
            self.thr2Entry.config(state='disabled')
            self.thr3Entry.config(state='disabled')
            self.thr4Entry.config(state='disabled')

    def quit(self):
        self.root.quit()
        self.root.destroy()
        gc.collect()
    def start_processing_all(self):
        global t1
        self.startprocessing.config(state='disabled')
        if not self.folder_path:
            positionFileButton.config(state='normal')
            messagebox.showerror('Folder Path Required', 'Please select a folder path.')
            return
        if not self.file_names:
            messagebox.showerror('Files Required', 'Please select files to process.')
            return
        chunksize = int(self.chunksize_entry_box.get())
        ncores = int(self.ncores_entry_box.get())
        t1=time.time()
        self.start_processing(
            self.segment,
            self.folder_path,
            self.file_names,
            int(self.start_index.get()),
            int(self.end_index.get()),
            self.separate_background.get(),
            self.segment_images.get(),
            self.do_watershed_3d.get(),
            self.threshold_option.get(),
            self.thr1.get(),
            self.thr2.get(),
            self.thr3.get(),
            self.thr4.get(),
            self.save_2d_slices.get(),
            self.save_3d_stack.get(),
            self.properties_2d.get(),
            self.properties_3d.get(),
            chunksize,
            ncores)
        
        print(f'Total time taken: {time.time()-t1} (s)')




    def start_processing(self,segment,folder_path, file_names, start, end, separate_background, segment_images, do_watershed_3d,threshold_option,thr1,thr2,thr3,thr4,save_2d_slices,save_3d_stack,properties2d,properties3d,chunksize,ncores):
        t=time.time()
        filtered_files=self.select_subset_files(start,end)

        # Output the filtered filenames
        [print(f"Filtered Filename: {filename}") for filename in filtered_files]


        #setting an instance for handling images and assigning it to a variable in segment class
        image_handler = ImageHandler(folder_path, filtered_files,chunksize)
        image_handler.open_images()
        self.plotimagebutton.config(state='normal')

        segment.separate_background = separate_background
        segment.segment_images = segment_images
        segment.do_watershed_3d = do_watershed_3d
        segment.threshold_option = threshold_option
        segment.thr1=thr1
        segment.thr2=thr2
        segment.thr3=thr3
        segment.thr4=thr4
        segment.chunksize=chunksize
        segment.ncores = ncores

        print(f'Splitting data into {chunksize} and using {ncores}')
        print(f'Separate background is {separate_background}')
        print(f'Segment is {segment_images}')
        print(f'do watershed is {do_watershed_3d}')

        if do_watershed_3d:
            segment_images = True

        if (separate_background or segment_images):
            print('processing images')
            segment.process_images()
        self.plotwatershedbutton.config(state='normal')


        #self.plot_watershed()
        
        if save_2d_slices:
            if segment.segmented_images:
                segment.save_segmented_slices()
            else:
                print('Segmented images is Null')

        if save_3d_stack:
            if segment.segmented_images:
                segment.save_segmented_stack() 
            else:
                print('Segmented images is Null')
        segment.roi = 1

        if properties2d:
            if segment.segmented_images:
                segment.Feret_size_horizontal_vertical_Calc()
            else:
                print('Segmented images is Null')
        if properties3d:
            if segment.segmented_images:
                segment.Feret_size_horizontal_vertical_3d_Calc()
            else:
                print('Segmented images is Null')
        print(f'Total processing time: {time.time()-t}')

        messagebox.showinfo('Processing Complete', 'Segmentation complete.')
        self.startprocessing.config(state='normal')
        self.Measure_2d_Button.config(state='normal')
        self.Measure_3d_Button.config(state='normal')
        self.Save_Seg_slices_Button.config(state='normal')
        self.Save_Seg_stack_Button.config(state='normal')

class ExtractROIs:
    def __init__(self, root, segment):
        # Initialize with root (for GUI), image_handler, and segment instances
        self.root = root
        self.segment = segment
        #self.segment.image_handler.output_folder = tk.StringVar()
        # Initialize default variables for extracting ROIs
        self.folder_path = tk.StringVar()
        self.label_folder_path = tk.StringVar()
        self.boxsize = tk.IntVar(value = 100)

        self.positions_file = tk.StringVar()
        self.output_directory = tk.StringVar()
        self.start_idx = tk.IntVar(value=0)
        self.total = tk.IntVar(value=2000)
        self.extract_images = tk.BooleanVar(value=True)
        self.extract_label_images = tk.BooleanVar(value=False)
        self.X = tk.IntVar(value = 50)
        self.Y = tk.IntVar(value = 50)
        self.Z = tk.IntVar(value = 50)
        self.isolated_volume = None
        self.grayscale_subsets = None
        self.ROIexport_option = tk.IntVar(value=1)  # Default option: Use predefined threshold

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Creating widgets in Extract ROI frame
        roi_frame = tk.LabelFrame(self.root, text="ROI Export Options")
        roi_frame.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)


        tk.Radiobutton(roi_frame, text='Extract ROIs from positions', variable=self.ROIexport_option,
                       value=1, command=self.enable_ROIexport_entries).grid(row=0, column=0, columnspan=3, sticky=tk.W)
        tk.Radiobutton(roi_frame, text='Extract single feature from label', variable=self.ROIexport_option,
                       value=2, command=self.enable_ROIexport_entries).grid(row=1, column=0, columnspan=3, sticky=tk.W)

        # Folder path selection
        tk.Label(roi_frame, text="Grayscale images:").grid(row=4, column=0, sticky="e")
        tk.Entry(roi_frame, textvariable=self.folder_path).grid(row=4, column=1)
        self.grayscalebutton=tk.Button(roi_frame, text="Browse", command=lambda:browse_folder(self),state='normal')
        self.grayscalebutton.grid(row=4, column=2)
        # Parameters for reading TIFF stack
        tk.Label(roi_frame, text="Start Index:").grid(row=5, column=0, sticky="e")
        tk.Entry(roi_frame, textvariable=self.start_idx).grid(row=5, column=1)

        tk.Label(roi_frame, text="Total Images:").grid(row=6, column=0, sticky="e")
        tk.Entry(roi_frame, textvariable=self.total).grid(row=6, column=1)

        tk.Label(roi_frame, text="Label images:").grid(row=7, column=0, sticky="e")
        tk.Entry(roi_frame, textvariable=self.label_folder_path).grid(row=7, column=1)
        self.labelimagebutton = tk.Button(roi_frame, text="Browse", command=lambda:browse_label_folder(self))
        self.labelimagebutton.grid(row=7, column=2)

        # Positions file selection
        tk.Label(roi_frame, text="Positions File:").grid(row=8, column=0, sticky="e")
        self.positionsfileEntry = tk.Entry(roi_frame, textvariable=self.positions_file)
        self.positionsfileEntry.grid(row=8, column=1)
        self.positionsfilebutton = tk.Button(roi_frame, text="Browse", command=lambda:browse_file(self))
        self.positionsfilebutton.grid(row=8, column=2)

        # Output directory selection
        tk.Label(roi_frame, text="Output Directory:").grid(row=9, column=0, sticky="e")
        self.outputDirectoryEntry = tk.Entry(roi_frame, textvariable=self.output_directory)
        self.outputDirectoryEntry.grid(row=9, column=1)
        self.outputDirectorybutton = (tk.Button(roi_frame, text="Browse", command=lambda:browse_output_directory(self)))
        self.outputDirectorybutton.grid(row=9, column=2)

        # Options for extraction
        tk.Checkbutton(roi_frame, text="Extract Grayscale Images", variable=self.extract_images).grid(row=10, column=0, sticky="w")
        tk.Checkbutton(roi_frame, text="Extract Label Images", variable=self.extract_label_images).grid(row=10, column=1, sticky="w")
        tk.Label(roi_frame, text="ROI size:").grid(row=10, column=2, sticky="e")
        tk.Entry(roi_frame, textvariable=self.boxsize).grid(row=10, column=3)

        # Process Button
        self.ExtractROIsButton = tk.Button(roi_frame, text="Extract ROIs", command=lambda:extract_rois(self))
        self.ExtractROIsButton.grid(row=11, column=1)

        # Save single Label volume as per input x,y,z
        SaveLabel_frame = tk.LabelFrame(roi_frame, text="Export single Label")
        SaveLabel_frame.grid(row=0, column=5,rowspan=8, padx=10, pady=10, sticky=tk.W)
        #tk.Button(SaveLabel_frame, text="Import Label volume", command=self.browse_labeled_folder).grid(row=0, column=2)
        
        tk.Label(SaveLabel_frame, text="X").grid(row=1, column=0, sticky="e")
        tk.Label(SaveLabel_frame, text="Y").grid(row=1, column=1, sticky="e")
        tk.Label(SaveLabel_frame, text="Z").grid(row=1, column=2, sticky="e")

        self.XEntry=tk.Entry(SaveLabel_frame, textvariable=self.X,width=10)
        self.XEntry.grid(row=2, column=0)
        self.YEntry=tk.Entry(SaveLabel_frame, textvariable=self.Y,width=10)
        self.YEntry.grid(row=2, column=1)
        self.ZEntry=tk.Entry(SaveLabel_frame, textvariable=self.Z,width=10)
        self.ZEntry.grid(row=2, column=2)
        self.exportlabel = tk.Button(SaveLabel_frame, text="Export Label", command=lambda:isolate_volume(self),state='disabled')
        self.exportlabel.grid(row=3, column=0,columnspan=3)
        self.measure3d = tk.Button(SaveLabel_frame, text="Measure3d", command=lambda:isolate_volume_measure3d(self),state='disabled')
        self.measure3d.grid(row=3, column=4)
        self.plotorthogonalbutton = tk.Button(SaveLabel_frame, text="Plot Orthogonal Projections", command=lambda:plot_orthogonal_projections(self),state='disabled')
        self.plotorthogonalbutton.grid(row=4, column=0,columnspan=3)

        self.plotvolrenderbutton = tk.Button(SaveLabel_frame, text="Plot volume rendered \nOrthogonal Projections", command=lambda: plot_volume_rendered_projections(self.isolated_volume),state='disabled')
        self.plotvolrenderbutton.grid(row=5, column=0,columnspan=3)

    def enable_ROIexport_entries(self):
        if self.ROIexport_option.get() == 2:
            self.grayscalebutton.config(state='normal')
            self.labelimagebutton.config(state='normal')
            self.positionsfilebutton.config(state='disabled')
            self.positionsfileEntry.config(state='disabled')
            self.outputDirectorybutton.config(state='disabled')
            self.outputDirectoryEntry.config(state='disabled')
            self.exportlabel.config(state='normal')
            self.measure3d.config(state='normal')
            self.plotorthogonalbutton.config(state='normal')
            self.plotvolrenderbutton.config(state='normal')
            self.ExtractROIsButton.config(state='disabled')

        else:
            self.grayscalebutton.config(state='normal')
            self.labelimagebutton.config(state='normal')
            self.positionsfilebutton.config(state='normal')
            self.positionsfileEntry.config(state='normal')
            self.outputDirectorybutton.config(state='normal')
            self.outputDirectoryEntry.config(state='normal')
            self.exportlabel.config(state='disabled')
            self.measure3d.config(state='disabled')
            self.plotorthogonalbutton.config(state='disabled')
            self.plotvolrenderbutton.config(state='disabled')
            self.ExtractROIsButton.config(state='normal')

if __name__ == "__main__":
    root = tk.Tk()
    segment = Segment(image_handler=None, separate_background=False, segment_images=False, do_watershed_3d = False ,threshold_option=False,thr1=0,thr2=0,thr3=0,thr4=0,roi=0,chunksize=100,ncores=4)
    app = Segmenter(root, segment)
    app.setup_gui()  # <- parentheses to actually call it
    root.mainloop()
