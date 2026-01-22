# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from pylibCZIrw import czi as pyczi
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import datetime
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
#import cv2
import scipy
import numpy as np 
from scipy.optimize import curve_fit



def estimate_background(img2, plot=False, smoothing=False):
    """
    Estimate vertical background wave from an image with NaNs marking removed pixels.
    Scales the column background (x_profile) per row to match the min/max of each row.
    """
    # average along columns (x_profile)
    x_profile = np.nanmean(img2, axis=0)  # mean intensity per column
    y_profile = np.nanmean(img2, axis=1)  # mean intensity per row
    x = np.arange(len(x_profile))

    
    # interpolate missing values if NaNs remain (x_profile)
    x_mask = ~np.isnan(x_profile)
    if np.sum(x_mask) < 5:
        x_profile = np.zeros_like(x_profile)
    else:
        f = scipy.interpolate.interp1d(x[x_mask], x_profile[x_mask], kind="linear", 
                                       fill_value="extrapolate")
        x_profile = f(x)
    if smoothing: 
        x_profile = scipy.ndimage.uniform_filter1d(x_profile, size=5, mode="reflect")
      
    # normalize x_profile to [0,1]
    x_norm = (x_profile - np.min(x_profile)) / (np.max(x_profile) - np.min(x_profile) + 1e-12)

    # expand and scale per row using each row's min/max
    y_background = np.zeros_like(img2)
    for i, row_val in enumerate(y_profile):
        row_min = np.nanquantile(img2[i, :],0.05)
        row_max = np.nanquantile(img2[i, :],0.95)
        if np.isnan(row_min) or np.isnan(row_max) or row_min == row_max:
            y_background[i, :] = x_norm * row_val  # fallback scaling
            print("Got NaNs up my arse! ")
        else:
            y_background[i, :] = row_min + x_norm * (row_max - row_min)

    background = y_background
    
    if plot:
        fig, ax = plt.subplots(dpi=600)
        ax.plot(x_norm)
        plt.title("Estimated X Background Profile")
        plt.show()

    return background




def estimate_background_simple(img2, plot=False,smoothing=False):
    """
    Estimate vertical background wave from an image with NaNs marking removed pixels.
    """
    # average along columns (vertical profile)
    #plt.imshow(img2)
    #plt.show()
    x_profile = np.nanmean(img2, axis=0) # mean intensity per row
    y_profile = np.nanmean(img2, axis=1) # mean intensity per column
    x = np.arange(len(x_profile))
    y = np.arange(len(y_profile))
    
    # interpolate missing values if NaNs remain
    x_mask = ~np.isnan(x_profile)
    if np.sum(x_mask) < 5:
        # fallback: flat background
        x_profile = np.zeros_like(x_profile)
    else:
        f = scipy.interpolate.interp1d(x[x_mask], x_profile[x_mask], kind="linear", 
                                        fill_value="extrapolate")
        x_profile = f(x)
        
    y_profile = np.nanmean(img2, axis=1)  # mean intensity per row
    y = np.arange(len(y_profile))

    # interpolate missing values if NaNs remain
    y_mask = ~np.isnan(y_profile)
    if np.sum(y_mask) < 2:
        # fallback: flat background
        y_profile = np.zeros_like(y_profile)
    else:
        f = scipy.interpolate.interp1d(y[y_mask], y_profile[y_mask], kind="linear", 
                                        fill_value="extrapolate")
        y_profile = f(y)
    
    #x_profile = np.ma.average(x_profile,axis=0,keepdims=True)
    #y_profile = np.ma.average(y_profile,axis=0,keepdims=True)
    if smoothing: 
        x_profile = scipy.ndimage.uniform_filter1d(x_profile, size=5, mode="reflect")
        y_profile = scipy.ndimage.uniform_filter1d(y_profile, size=5, mode="reflect")
    
    
    # expand back into image
    x_background = np.tile(x_profile, (img2.shape[0],1))
    y_background = np.tile(y_profile.reshape((y_profile.size,1)), (1,img2.shape[1]))

    #background = y_background + x_background
    background = x_background
    
    if plot:
        fig,ax = plt.subplots(dpi=600)
        ax.plot(x_profile)
        plt.title("Estimated X Background Profile")
        plt.show()
        fig,ax = plt.subplots(dpi=600)
        ax.plot(y_profile)
        plt.title("Estimated Y Background Profile")
        plt.show()

    return background


def your_awb(im):
    """
    Auto white balance on the whole image.
    Each channel is scaled so its mean value is mapped to 128.
    """
    im = im.astype(float)
    h, w, c = im.shape

    for i in range(c):
        mean_val = np.mean(im[:, :, i])
        if mean_val > 0:
            scale = 224 / mean_val
            im[:, :, i] *= scale

    im = np.clip(im, 0, 255)
    return im.astype(np.uint8)

def destripe(im,quantile=0.15,debug=False,method='simple'):
    
    backgroundPix = []
    #im = your_awb(im)
    #fig,ax = plt.subplots()
    #ax.imshow(im)
    #plt.show()
    nu_im = np.tile(0,im.shape)
    
    for i in range(im.shape[2]):
        subIm = im[:,:,i]
        
        idx = subIm<np.quantile(subIm,quantile)
        backgroundPix.append(idx)
        img2= copy.copy(im[:,:,i])

        img2 = img2.astype(float)
        img2[idx] = 255

        img2[idx] = np.nan
        signal = np.nanmean(img2,axis=0)
        t = np.arange(0,signal.size)
        if method == 'simple':
            corr_im = estimate_background_simple(img2, plot=debug)
        else:
            corr_im = estimate_background(img2, plot=debug)
        #fig,ax = plt.subplots()
        #ax.imshow(corr_im)
        #plt.show()
        
        c2 = corr_im.astype(int)
        c3 = c2-np.min(c2)
        #fig,ax = plt.subplots()
        #ax.imshow(c3)
        #plt.show()
        nu_im[:,:,i] = subIm-c3
    nu_im = your_awb(nu_im)
    return nu_im


class CarlScheissViewer:
    def __init__(self, root,fname):
        self.root = root
        self.root.title("THE CARL SCHEISS VIEWER")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        target_height = int(screen_height * 0.65)
        target_width = int(screen_width * 0.8)
        export_name = os.path.basename(fname)[0:-4]
        
        self.root.geometry(f"{target_width}x{target_height}")
        self.fname = fname
        self.xpos = 0
        self.ypos = 0
        self.xwidth = 5200
        self.ywidth = 5200
        self.resolution = 0.6
        self.increment = 128
        self.savefolder = "./"
        self.save_name = export_name
        self.export_count = 0
        self.roi_mode = False
        self.roi_rect = None
        self.fetch_dims()
        self.fetch_overview()
        self._build_ui()
        self._load_overview()
        self._load_subimage()
        self._update_subimage_display()
        self.root.bind("<Key>", self._on_keypress)
        
       
    def fetch_dims(self):
        with pyczi.open_czi(self.fname) as czidoc:
            # get the image dimensions as a dictionary, where the key identifies the dimension
            total_bounding_box = czidoc.total_bounding_box

            # get the total bounding box for all scenes
            total_bounding_rectangle = czidoc.total_bounding_rectangle
        self.image_width = total_bounding_rectangle[2]
        self.image_height = total_bounding_rectangle[3]

    def fetch_overview(self):
        
        with pyczi.open_czi(self.fname) as czidoc:
            # get the image dimensions as a dictionary, where the key identifies the dimension
            total_bounding_box = czidoc.total_bounding_box
            

            # get the total bounding box for all scenes
            total_bounding_rectangle = czidoc.total_bounding_rectangle
            print("TOTAL BOUNDING BOX IS ", total_bounding_rectangle)
            
            # get the bounding boxes for each individual scene
            scenes_bounding_rectangle = czidoc.scenes_bounding_rectangle

            # read a 2D image plane and optionally specify planes, zoom levels and ROIs
            image2d = czidoc.read(plane={"T": 0, "Z": 0, "C": 0}, zoom=0.01,
                                  roi=(total_bounding_rectangle[0], 
                                       total_bounding_rectangle[1],
                                       self.image_width, 
                                       self.image_height,))
        self.full_image = Image.fromarray(image2d[:,:,[2,1,0]])

    #def _generate_dummy_image(self):
    #    data = np.random.randint(0, 256, (1024, 2048, 3), dtype=np.uint8)
    #    self.full_image = Image.fromarray(data)

    def _build_ui(self):
        tk.Label(self.root, text="THE CARL SCHEISS VIEWER", font=("Arial", 22),
                 fg="blue").pack()

        settings_frame = tk.Frame(self.root)
        settings_frame.pack()

        self._add_setting_input(settings_frame, "X Pos", 'xpos')
        self._add_setting_input(settings_frame, "Y Pos", 'ypos')
        self._add_setting_input(settings_frame, "X Width", 'xwidth')
        self._add_setting_input(settings_frame, "Y Width", 'ywidth')
        self._add_setting_input(settings_frame, "Resolution", 'resolution')
        self._add_setting_input(settings_frame, "Increment", 'increment')

        tk.Button(settings_frame, text="Set Save Folder", command=self._set_save_folder).grid(row=0, column=6)
        self.save_name_entry = tk.Entry(settings_frame)
        self.save_name_entry.insert(0, self.save_name)
        self.save_name_entry.grid(row=1, column=6)

        # Save/Quit controls moved up under settings
        top_control_frame = tk.Frame(self.root)
        top_control_frame.pack(pady=(5, 10))
        tk.Button(top_control_frame, text="Save Subimage", command=self._save_subimage).pack(side="left", padx=10)
        tk.Button(top_control_frame, text="Quit", command=self._quit_all).pack(side="right", padx=10)

        image_frame = tk.Frame(self.root)
        image_frame.pack(fill="both", expand=True)

        self.canvas_overview = tk.Canvas(image_frame)
        self.canvas_overview.pack(side="left", expand=True, fill="both")
        self.canvas_overview.bind("<Button-1>", self._on_overview_click)

        self.canvas_subimage = tk.Canvas(image_frame)
        self.canvas_subimage.pack(side="right", expand=True, fill="both")
        self.canvas_subimage.bind("<Button-1>", self._start_roi)
        self.canvas_subimage.bind("<B1-Motion>", self._draw_roi)
        self.canvas_subimage.bind("<ButtonRelease-1>", self._end_roi)

    def _add_setting_input(self, frame, label, attr):
        row = (list(vars(self)).index(attr)) % 6
        tk.Label(frame, text=label).grid(row=0, column=row)
        entry = tk.Entry(frame)
        entry.insert(0, str(getattr(self, attr)))
        entry.grid(row=1, column=row)
        entry.bind("<Return>", lambda e, a=attr, en=entry: self._update_setting(a, en))
        setattr(self, f"entry_{attr}", entry)

    def _update_setting(self, attr, entry):
        try:
            value_str = entry.get()
            if attr == 'resolution':
                value = float(value_str)
                if not (0 < value <= 1):
                    raise ValueError("Resolution must be between 0 and 1.")
            else:
                value = int(value_str)
            setattr(self, attr, value)
            self._load_subimage()
            self._update_subimage_display()
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"{attr}: {e}")

    def _set_save_folder(self):
        self.savefolder = filedialog.askdirectory()

    def _load_overview(self):
        
        self.overview_scale = 512 / self.full_image.width
        overview_height = int(self.full_image.height * self.overview_scale)
        self.overview_img = self.full_image.resize((512, overview_height))
        self.tk_overview = ImageTk.PhotoImage(self.overview_img)
        self.canvas_overview.config(width=512, height=overview_height)
        self.canvas_overview.create_image(0, 0, anchor="nw", image=self.tk_overview)
        
        self.conversion_x = 512/self.image_width
        self.conversion_y = overview_height/self.image_height

    def _on_overview_click(self, event):
        x_full = int(event.x / self.conversion_x)
        y_full = int(event.y / self.conversion_y)

        self.xpos = max(0, min(self.image_width - self.xwidth, x_full - self.xwidth // 2))
        self.ypos = max(0, min(self.image_height - self.ywidth, y_full - self.ywidth // 2))

        self._load_subimage()
        self._update_subimage_display()

    

    def load_image(self, x, y, xw, yw, resolution):
        
        with pyczi.open_czi(self.fname) as czidoc:
            # get the image dimensions as a dictionary, where the key identifies the dimension
            total_bounding_box = czidoc.total_bounding_box

            # get the total bounding box for all scenes
            total_bounding_rectangle = czidoc.total_bounding_rectangle

            # get the bounding boxes for each individual scene
            scenes_bounding_rectangle = czidoc.scenes_bounding_rectangle

            # read a 2D image plane and optionally specify planes, zoom levels and ROIs
            image2d = czidoc.read(plane={"T": 0, "Z": 0, "C": 0}, zoom=resolution,
                                  roi=(total_bounding_rectangle[0]+ x,
                                       total_bounding_rectangle[1]+ y,
                                       xw, 
                                       yw,))
        return image2d[:,:,[2,1,0]]




    def _load_subimage(self):
        subimage = self.load_image(
            self.xpos, self.ypos, self.xwidth, self.ywidth, self.resolution
        )
        self.subimage = Image.fromarray(subimage)

    def _update_subimage_display(self):
        target_width = 512
        #aspect_ratio = self.subimage.height / self.subimage.width
        aspect_ratio = self.subimage.height / self.subimage.width
        target_height = int(target_width * aspect_ratio)

        self.tk_sub = ImageTk.PhotoImage(self.subimage.resize((target_width, target_height)))
        self.canvas_subimage.config(width=target_width, height=target_height)
        self.canvas_subimage.create_image(0, 0, anchor="nw", image=self.tk_sub)

        self.canvas_overview.delete("rect")
        xscale = self.conversion_x
        yscale = self.conversion_y
        self.canvas_overview.create_rectangle(
            int(self.xpos * xscale), int(self.ypos * yscale),
            int((self.xpos + self.xwidth) * xscale),
            int((self.ypos + self.ywidth) * yscale),
            outline="red", width=2, tag="rect"
        )

    def _on_keypress(self, event):
        key = event.keysym
        if key == "Left":
            self.xpos = self.xpos - self.increment
        elif key == "Right":
            self.xpos = self.xpos + self.increment
        elif key == "Up":
            self.ypos = self.ypos - self.increment
        elif key == "Down":
            self.ypos = self.ypos + self.increment
        else:
            return  # ignore other keys

        self._load_subimage()
        self._update_subimage_display()
        self._sync_position_inputs()
        
    def _get_pixel_size_um(self):
        with pyczi.open_czi(self.fname) as czidoc:
            metadata = czidoc.metadata  # ← no parentheses!
            try:
                imPixels = metadata['ImageDocument']['Metadata']['ImageScaling']['ImagePixelSize']
                div = imPixels.index(',')
                px_um_x = float(imPixels[0:div])
                px_um_y = float(imPixels[div+1:])
            except Exception:
                px_um_x = px_um_y = 1.0  # fallback to 1 µm if unavailable
                
            print('#######################################################')
            print('resolution X Y:' + str(px_um_x) + str(px_um_y))
        return px_um_x, px_um_y
        
    def _save_subimage(self):
        name = self.save_name_entry.get()
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{now}_{self.export_count}.png"
        path = os.path.join(self.savefolder, filename)

        if self.roi_rect:
            x0, y0, x1, y1 = self.canvas_subimage.coords(self.roi_rect)
            x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])

            canvas_w = self.canvas_subimage.winfo_width()
            canvas_h = self.canvas_subimage.winfo_height()
            img_w, img_h = self.subimage.size
            scale_x = img_w / canvas_w
            scale_y = img_h / canvas_h

            x0_img = int(x0 * scale_x)
            y0_img = int(y0 * scale_y)
            x1_img = int(x1 * scale_x)
            y1_img = int(y1 * scale_y)

            x0_img, x1_img = sorted((max(0, x0_img), min(img_w, x1_img)))
            y0_img, y1_img = sorted((max(0, y0_img), min(img_h, y1_img)))

            roi = self.subimage.crop((x0_img, y0_img, x1_img, y1_img))
            roi.save(path)
            self.canvas_subimage.delete(self.roi_rect)
            self.roi_rect = None
        else:
            # Save full subimage with pixel size metadata
            px_um_x, px_um_y = self._get_pixel_size_um()
            #dpi_x = int(25400 * px_um_x*self.resolution)  # 1 inch = 25.4 mm → convert µm to DPI
            #dpi_y = int(25400 * px_um_y*self.resolution)
            dpi_x = px_um_x*self.resolution # 1 inch = 25.4 mm → convert µm to DPI
            dpi_y = px_um_y*self.resolution
            #self.subimage.save(path, dpi=(dpi_x, dpi_y))
            try: 
                backup = copy.copy(self.subimage)
                img = np.asarray(self.subimage)
                print("size is:", img.shape)
                img = destripe(img).astype(np.uint8)
                self.subimage = Image.fromarray(img)
                self.subimage.save(path)
            except:
                self.subimage = backup
                self.subimage.save(path)
            
        self.export_count += 1
        messagebox.showinfo("Saved", f"Saved {path}")
        
    def _sync_position_inputs(self):
        self.entry_xpos.delete(0, tk.END)
        self.entry_xpos.insert(0, str(self.xpos))
        self.entry_ypos.delete(0, tk.END)
        self.entry_ypos.insert(0, str(self.ypos))

    def _start_roi(self, event):
        if self.roi_rect:
            self.canvas_subimage.delete(self.roi_rect)
            self.roi_rect = None
        self.roi_start = (event.x, event.y)
        self.roi_rect = self.canvas_subimage.create_rectangle(
            event.x, event.y, event.x, event.y, outline="blue"
        )

    def _draw_roi(self, event):
        if self.roi_rect:
            self.canvas_subimage.coords(self.roi_rect, *self.roi_start, event.x, event.y)

    def _end_roi(self, event):
        pass

    def _quit_all(self):
        self.root.quit()
        self.root.destroy()



def get_file_path():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select CZI File",
        filetypes=[("CZI files", "*.czi"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path


if __name__ == "__main__":
    #filePath = '/home/wormulon/Downloads/R3G3_H10Adults_310325rep2.czi'
    filePath = get_file_path()
    root = tk.Tk()
    app = CarlScheissViewer(root,filePath)
    root.mainloop()

