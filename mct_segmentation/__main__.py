if __name__ == '__main__':
    root = tk.Tk()
    segment = Segment(image_handler=None, separate_background=False, segment_images=False, do_watershed_3d = False ,threshold_option=False,thr1=0,thr2=0,thr3=0,thr4=0,roi=0,chunksize=100,ncores=4)
    extract_rois = ExtractROIs(root, segment)
    app = App(root,segment)

    root.mainloop()
