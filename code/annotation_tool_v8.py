import os,json,cv2,csv,sys
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import networkx as nx
from networkx.exception import NetworkXNoPath
from skimage.draw import polygon as sk_polygon

def get_worm_centerline(poly, plot=False, padding=10):
    """
    Extract centerline from an open or closed worm polygon.
    
    Parameters:
    - poly: Nx2 array of xy points (polygon of worm outline)
    - plot: bool, if True plot polygon and centerline
    - padding: int, pixels to pad around polygon in mask
    
    Returns:
    - centerline: Mx2 array of xy coords along centerline
    - length: float, length of centerline in pixels
    """
    def convert2array(plist):
        print("THIS IS NOT A NP.ARRAY!!!!!!!!!!!!!!!!!!!!")
        worm = np.empty((int(len(plist)/2),2))
        w = np.asarray(plist)
        worm[:,0] = w[np.arange(0,len(plist),2)]
        worm[:,1] = w[np.arange(1,len(plist),2)]
        return worm
    
    if isinstance(poly,list):
        poly = convert2array(poly)
    
    poly = np.array(poly)
    
    # 1. Close polygon if open
    if not np.array_equal(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    
    # Shift polygon coords to positive with padding
    min_xy = poly.min(axis=0) - padding
    poly_shifted = poly - min_xy
    
    # Define mask size with padding
    max_xy = poly_shifted.max(axis=0) + padding
    img_shape = (int(np.ceil(max_xy[1]))+1, int(np.ceil(max_xy[0]))+1)  # rows, cols (y,x)
    
    # 2. Create binary mask by rasterizing polygon
    rr, cc = sk_polygon(poly_shifted[:,1], poly_shifted[:,0], img_shape)
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask[rr, cc] = 1
    area = np.sum(mask)
    
    # 3. Morphological close to clean mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 4. Skeletonize mask (convert to boolean)
    skeleton = skeletonize(mask > 0)
    
    # 5. Build graph from skeleton pixels
    G = nx.Graph()
    ys, xs = np.where(skeleton)
    pixels = list(zip(xs, ys))
    for x,y in pixels:
        G.add_node((x,y))
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor = (x+dx, y+dy)
                if neighbor in G.nodes:
                    G.add_edge((x,y), neighbor)
    
    # 6. Find endpoints (nodes with degree 1)
    endpoints = [n for n,d in G.degree() if d == 1]
    
    if len(endpoints) < 2:
        raise ValueError("Could not find two endpoints in skeleton.")
    
    # If more than 2 endpoints, pick farthest apart
    if len(endpoints) > 2:
        max_dist = 0
        for i in range(len(endpoints)):
            for j in range(i+1, len(endpoints)):
                dist = np.linalg.norm(np.array(endpoints[i]) - np.array(endpoints[j]))
                if dist > max_dist:
                    max_dist = dist
                    end1, end2 = endpoints[i], endpoints[j]
    else:
        end1, end2 = endpoints
    
    # 7. Shortest path = centerline
    

# 7. Try to find shortest path = centerline
    try:
        path = nx.shortest_path(G, source=end1, target=end2)
    except NetworkXNoPath:
        print(f"⚠️ No path between {end1} and {end2}. Trying fallback method...")
    
        # Fallback: use the longest shortest path among all endpoint pairs
        fallback_path = []
        max_len = 0
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                try:
                    p = nx.shortest_path(G, source=endpoints[i], target=endpoints[j])
                    if len(p) > max_len:
                        max_len = len(p)
                        fallback_path = p
                except NetworkXNoPath:
                    continue
    
        if not fallback_path:
            raise ValueError(f"No connected path found in skeleton for polygon.")
        path = fallback_path
    
    centerline = np.array(path)
    # Shift back to original coordinate system
    centerline = centerline + min_xy[::-1]  # note: (x,y) vs (row,col)
    # centerline is currently (x,y) but path nodes are (x,y) with y=row, x=col
    
    # Correct order of coordinates to xy (flip axis)
    centerline_xy = centerline[:, [0,1]]  # x,y
    
    # 8. Calculate length along centerline
    diffs = np.diff(centerline_xy, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    length = np.sum(segment_lengths)
    
    dx = np.min(poly[:,0]) - padding/2
    dy = np.min(poly[:,1]) - padding/2
    ex = np.min(centerline_xy[:,0]) - padding
    ey = np.min(centerline_xy[:,1]) - padding
    
    print(dx,dy)
    centerline_xy = centerline + np.tile([dx,dy],(centerline_xy.shape[0],1)) - np.tile([ex,ey],(centerline_xy.shape[0],1))

                    
    # Plot if requested
    if plot:
        plt.figure(figsize=(8,6))
        plt.plot(poly[:,0], poly[:,1], 'b-', label='Worm outline')
        plt.plot(centerline_xy[:,0], centerline_xy[:,1], 'r-', linewidth=2, label='Centerline')
        plt.axis('equal')
        plt.legend()
        plt.title(f'Worm centerline length: {length:.2f} px')
        plt.show()
    
    return centerline_xy, length, area

def merge_annotation_folder(anno_dir, color_csv, output_point_csv, output_area_json):
    """
    Merges all annotation CSV and JSON files in a folder and saves them as single combined files.
    Cleans up the JSON files using global class mapping.
    """
    all_points = []
    all_json = {
        "images": [],
        "annotations": [],
        "categories": []  # Will be overwritten during cleanup
    }
    image_id = 1
    annotation_id = 1

    # Temporary directory to hold cleaned JSONs
    cleaned_jsons = []

    # Process files
    for file in os.listdir(anno_dir):
        full_path = os.path.join(anno_dir, file)

        if file.endswith(".csv"):
            df = pd.read_csv(full_path, header=None)
            all_points.append(df)

        elif file.endswith(".json"):
            # Clean JSON in-place
            class_clean_up(full_path, color_csv)
            with open(full_path, 'r') as f:
                data = json.load(f)

            img_id_map = {}
            for img in data.get("images", []):
                new_img = img.copy()
                new_img["id"] = image_id
                img_id_map[img["id"]] = image_id
                all_json["images"].append(new_img)
                image_id += 1

            for ann in data.get("annotations", []):
                new_ann = ann.copy()
                new_ann["id"] = annotation_id
                new_ann["image_id"] = img_id_map[ann["image_id"]]
                all_json["annotations"].append(new_ann)
                annotation_id += 1

            # Overwrite categories later, just pick one from last cleaned JSON
            all_json["categories"] = data.get("categories", [])

    # Save merged CSV
    if all_points:
        merged_df = pd.concat(all_points, ignore_index=True)
        merged_df.to_csv(output_point_csv, header=False, index=False)

    # Save merged JSON
    with open(output_area_json, "w") as f:
        json.dump(all_json, f, indent=2)

    print(f"✅ Merged CSV saved to: {output_point_csv}")
    print(f"✅ Merged JSON saved to: {output_area_json}")


def class_clean_up(json_path, csv_path, output_path=None):
    """
    Syncs class IDs in a COCO-style JSON annotation file with global IDs from a CSV file.
    Preserves class names and updates both 'annotations' and 'categories' sections.
    
    Parameters:
        json_path (str): Path to the JSON annotation file.
        csv_path (str): Path to the CSV file containing class names and global IDs.
        output_path (str, optional): Path to save the corrected JSON. If None, overwrites input JSON.
    """
    # Load CSV and create name → global ID mapping
    class_df = pd.read_csv(csv_path, header=None)
    name_to_id = dict(zip(class_df[0], class_df[4]))

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Build name → original ID mapping from JSON
    original_id_to_name = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    name_to_original_id = {v: k for k, v in original_id_to_name.items()}

    # Update 'annotations' with new global IDs
    for ann in data.get("annotations", []):
        old_id = ann["category_id"]
        class_name = original_id_to_name.get(old_id)
        if class_name in name_to_id:
            ann["category_id"] = name_to_id[class_name]
        else:
            print(f"⚠️ Warning: Class name '{class_name}' not found in CSV. Skipping.")

    # Rebuild 'categories' with correct IDs
    updated_categories = []
    used_names = set()
    for class_name, global_id in name_to_id.items():
        if class_name not in used_names:
            updated_categories.append({
                "id": int(global_id),
                "name": class_name,
                "supercategory": ""
            })
            used_names.add(class_name)

    data["categories"] = updated_categories

    # Save corrected JSON
    save_path = output_path if output_path else json_path
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Class cleanup completed and saved to: {save_path}")

class AnnotationTool:
    def __init__(self, image_dir, point_csv, area_json, color_csv):
        
        #syncing the area data csv with global class ids........................
        #class_clean_up(area_json, color_csv)
        self.image_dir = image_dir
        self.point_csv_path = point_csv
        self.area_json_path = area_json

        print(point_csv)
        self.point_df = pd.read_csv(point_csv, header=None,
                                    names=["class", "x", "y", "image_name", "width", "height"])
        self.point_df = self.point_df.astype({"x": float, "y": float})

        with open(area_json, 'r') as f:
            self.area_data = json.load(f)
           
        self.class_colors, self.class_names, self.class_ids = self.load_colors_and_classes(color_csv)

        self.image_files = sorted(list(set(self.point_df['image_name'].tolist())))
        self.current_index = 0
        self.selection_box = None
        self.root = tk.Tk()
        self.inspector = None
        self.inspector_canvas = None
        self.inspector_zoom = tk.DoubleVar(master=self.root, value=4.0)
        self.inspector_image = None
        self.root.title("Annotation Tool")

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.max_width = self.screen_width - 100
        self.max_height = self.screen_height - 150

        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.class_var = tk.StringVar(self.root)
        self.class_dropdown = tk.OptionMenu(self.root, self.class_var, *self.class_colors.keys(), command=self.update_class)
        self.class_dropdown.pack(side=tk.BOTTOM)

        self.exit_button = tk.Button(self.root, text="Exit", command=self.root.destroy)
        self.exit_button.pack(side=tk.BOTTOM)
        
        self.prev_button = tk.Button(self.root, text="Previous", command=self.prev_image)
        self.prev_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.next_button = tk.Button(self.root, text="Next", command=self.next_image)
        self.next_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        
        
        
        self.pixelsize = 1.1374
        self.pixelsize_var = tk.StringVar(value=str(self.pixelsize))
        self.pixelsize_label = tk.Label(self.root, text="Pixel size:")
        self.pixelsize_label.pack(side=tk.LEFT, padx=(10, 2), pady=5)
        self.pixelsize_entry = tk.Entry(self.root, textvariable=self.pixelsize_var, width=6)
        self.pixelsize_entry.pack(side=tk.LEFT, pady=5)
        self.pixelsize_entry.bind("<Return>", self.update_pixelsize)
        
        self.reporter_button = tk.Button(self.root, text="Open Report", command=lambda: AnnotationReporter(self))
        self.reporter_button.pack(side=tk.RIGHT, padx=(10, 2), pady=5)

        self.selected_point = None
        self.selected_polygon = None
        self.selected_ann_idx = None
        self.dragging = False
        self.scale_factor = 1.0
        self.active_selection = None
        
        self.load_image()
        self.root.mainloop()
    
    def open_inspector(self):
        if self.inspector is not None:
            return
    
        self.inspector = tk.Toplevel(self.root)
        self.inspector.title("Instance Inspector")
        self.inspector.geometry("450x450")
    
        # --- layout frames ---
        container = tk.Frame(self.inspector)
        container.pack(fill=tk.BOTH, expand=True)
    
        canvas_frame = tk.Frame(container)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
    
        control_frame = tk.Frame(container)
        control_frame.pack(fill=tk.X)
    
        # --- scrollbars ---
        vbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        hbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
    
        self.inspector_canvas = tk.Canvas(
            canvas_frame,
            bg="black",
            xscrollcommand=hbar.set,
            yscrollcommand=vbar.set
        )
    
        vbar.config(command=self.inspector_canvas.yview)
        hbar.config(command=self.inspector_canvas.xview)
    
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.inspector_canvas.pack(fill=tk.BOTH, expand=True)
    
        # --- zoom controls (ALWAYS visible) ---
        tk.Label(control_frame, text="Zoom").pack(side=tk.LEFT, padx=5)
    
        zoom_slider = tk.Scale(
            control_frame,
            from_=1,
            to=12,
            resolution=0.5,
            orient=tk.HORIZONTAL,
            variable=self.inspector_zoom,
            command=lambda _: self.update_inspector_from_selection()
        )
        zoom_slider.pack(fill=tk.X, expand=True, padx=5)
    
        # --- panning bindings ---
        self.inspector_canvas.bind("<ButtonPress-2>", self._start_pan)
        self.inspector_canvas.bind("<B2-Motion>", self._do_pan)
    
        # Linux / Windows mouse wheel
        self.inspector_canvas.bind("<MouseWheel>", self._on_mousewheel)
        # macOS
        self.inspector_canvas.bind("<Button-4>", self._on_mousewheel)
        self.inspector_canvas.bind("<Button-5>", self._on_mousewheel)
    
        self.inspector.protocol("WM_DELETE_WINDOW", self.close_inspector)
    
    def _start_pan(self, event):
        self.inspector_canvas.scan_mark(event.x, event.y)

    def _do_pan(self, event):
        self.inspector_canvas.scan_dragto(event.x, event.y, gain=1)
    
    def _on_mousewheel(self, event):
        if event.num == 5 or event.delta < 0:
            self.inspector_canvas.yview_scroll(1, "units")
        else:
            self.inspector_canvas.yview_scroll(-1, "units")
    
    def close_inspector(self):
        if self.inspector:
            self.inspector.destroy()
        self.inspector = None
    
    def update_inspector_from_selection(self):
        if self.active_selection is None or self.inspector is None:
            return
    
        zoom = self.inspector_zoom.get()
        img = self.cv_image.copy()
    
        sel_type, data = self.active_selection
    
        if sel_type == "point":
            i = self.point_items[data]
            x = int(self.point_df.at[i, "x"])
            y = int(self.point_df.at[i, "y"])
            half = 30
            crop = img[
                max(0, y-half):min(img.shape[0], y+half),
                max(0, x-half):min(img.shape[1], x+half)
            ]
    
        else:  # polygon or polygon_ann
            if sel_type == "polygon":
                _, ann_idx, _ = data
            else:
                ann_idx = data
    
            coords = self.area_data["annotations"][ann_idx]["segmentation"][0]
            xs = np.array(coords[0::2]).astype(int)
            ys = np.array(coords[1::2]).astype(int)
    
            pad = 10
            x0, x1 = max(0, xs.min()-pad), min(img.shape[1], xs.max()+pad)
            y0, y1 = max(0, ys.min()-pad), min(img.shape[0], ys.max()+pad)
    
            crop = img[y0:y1, x0:x1]
    
        if crop.size == 0:
            return
    
        h, w = crop.shape[:2]
        crop = cv2.resize(
            crop,
            (int(w*zoom), int(h*zoom)),
            interpolation=cv2.INTER_NEAREST
        )
    
        pil = Image.fromarray(crop)
        self.inspector_image = ImageTk.PhotoImage(pil)
    
        self.inspector_canvas.delete("all")
        self.inspector_canvas.delete("all")

        img_id = self.inspector_canvas.create_image(
            0, 0, anchor=tk.NW, image=self.inspector_image
        )
        
        # VERY IMPORTANT: define scrollable area
        self.inspector_canvas.config(
            scrollregion=(0, 0,
                          self.inspector_image.width(),
                          self.inspector_image.height())
        )
        self.inspector_canvas.create_image(0, 0, anchor=tk.NW, image=self.inspector_image)
    
    
    
    def clear_selection_box(self):
        if self.selection_box is not None:
            self.canvas.delete(self.selection_box)
            self.selection_box = None


    def draw_point_selection_box(self, x, y, size=50):
        self.clear_selection_box()
        half = size // 2
        self.selection_box = self.canvas.create_rectangle(
            x - half, y - half, x + half, y + half,
            outline="red", width=2
        )
    
    
    def draw_polygon_selection_box(self, coords_scaled):
        """
        coords_scaled: list of (x,y) in *canvas coordinates*
        """
        self.clear_selection_box()
        xs = [p[0] for p in coords_scaled]
        ys = [p[1] for p in coords_scaled]
    
        self.selection_box = self.canvas.create_rectangle(
            min(xs), min(ys), max(xs), max(ys),
            outline="red", width=2
        )
    
    
    def next_image(self):
        self.current_index = (self.current_index + 1) % len(self.image_files)
        self.load_image()

    def prev_image(self):
        self.current_index = (self.current_index - 1) % len(self.image_files)
        self.load_image()

    def load_colors_and_classes(self, color_csv):
        df = pd.read_csv(color_csv, header=None)
        colors = {row[0]: (int(row[1]), int(row[2]), int(row[3])) for _, row in df.iterrows()}
        class_names = df[df.columns[0]].to_numpy()
        class_ids = df[df.columns[4]].to_numpy()
        return colors,class_names, class_ids
    
    def update_pixelsize(self, event=None):
        try:
            val = float(self.pixelsize_var.get())
            self.pixelsize = val
            print(f"✅ Pixel size updated to: {self.pixelsize}")
        except ValueError:
            print("⚠️ Invalid pixel size. Must be a float.")
            self.pixelsize_var.set(str(self.pixelsize))  # Reset to last valid value
    

    def load_image(self):
        self.canvas.delete("all")
        self.image_name = self.image_files[self.current_index]
        path = os.path.join(self.image_dir, self.image_name)
        print(path)
        self.cv_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        self.orig_height, self.orig_width = self.cv_image.shape[:2]

        scale_w = self.max_width / self.orig_width
        scale_h = self.max_height / self.orig_height
        self.scale_factor = min(scale_w, scale_h, 1.0)

        new_width = int(self.orig_width * self.scale_factor)
        new_height = int(self.orig_height * self.scale_factor)
        resized_image = cv2.resize(self.cv_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        self.pil_image = Image.fromarray(resized_image)
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        self.display_annotations()
        self.active_selection = None
        self.active_selection = None
        self.clear_selection_box()
        self.clear_selection_box()
        if self.inspector:
            self.inspector_canvas.delete("all")

    def display_annotations(self):
        self.point_items = {}
        self.polygon_items = {}
        self.vertex_handles = {}

        for i, row in self.point_df[self.point_df["image_name"] == self.image_name].iterrows():
            x, y = row["x"] * self.scale_factor, row["y"] * self.scale_factor
            class_name = row["class"]
            color = self.class_colors.get(class_name, (255, 255, 255))
            hex_color = self.rgb_to_hex(color)
            item = self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5,
                                           fill=hex_color, outline="black", tags=("point", f"point_{i}"))
            self.point_items[item] = i

        image_id = None
        for img in self.area_data["images"]:
            if img["file_name"] == self.image_name:
                image_id = img["id"]
                break
        if image_id is None:
            return

        for ann_idx, ann in enumerate(self.area_data["annotations"]):
            if ann["image_id"] == image_id:
                class_id = ann["category_id"]
                class_name = next((c["name"] for c in self.area_data["categories"] if c["id"] == class_id), "")
                color = self.class_colors.get(class_name, (255, 255, 255))
                hex_color = self.rgb_to_hex(color)
                coords = ann["segmentation"][0]
                points = [(coords[i] * self.scale_factor, coords[i + 1] * self.scale_factor)
                          for i in range(0, len(coords), 2)]
                item = self.canvas.create_polygon(*[v for p in points for v in p], fill='', outline=hex_color, width=2, tags="polygon")
                self.polygon_items[item] = (ann_idx, coords)
                for i, (vx, vy) in enumerate(points):
                    handle = self.canvas.create_oval(vx-2, vy-2, vx+2, vy+2, fill=hex_color, outline="black", tags="vertex")
                    self.vertex_handles[handle] = (item, ann_idx, i*2)

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % rgb

    def on_click(self, event):
        x, y = event.x, event.y
    
        # Check point annotations
        for item in self.point_items:
            i = self.point_items[item]
            px = self.point_df.at[i, "x"] * self.scale_factor
            py = self.point_df.at[i, "y"] * self.scale_factor
            if abs(x - px) <= 5 and abs(y - py) <= 5:
                self.active_selection = ("point", item)
                if self.inspector is None:
                    self.open_inspector()
                
                self.update_inspector_from_selection()
                self.class_var.set(self.point_df.at[i, "class"])
                self.dragging = True
                
                # Draw 50x50 selection box
                self.draw_point_selection_box(px, py)
                
                return
    
        # Check polygon vertex handles
        for handle in self.vertex_handles:
            vx, vy, _, _ = self.canvas.coords(handle)
            if abs(x - (vx+4)) <= 5 and abs(y - (vy+4)) <= 5:
                selection = self.vertex_handles[handle]
                item, ann_idx, _ = selection
                class_id = self.area_data["annotations"][ann_idx]["category_id"]
                class_name = next((c["name"] for c in self.area_data["categories"] if c["id"] == class_id), "")
                self.active_selection = ("polygon", selection)
                if self.inspector is None:
                    self.open_inspector()
                
                self.update_inspector_from_selection()
                self.class_var.set(class_name)
                self.dragging = True
                
                # Draw tight bbox around polygon
                coords = self.area_data["annotations"][ann_idx]["segmentation"][0]
                scaled = [(coords[i]*self.scale_factor, coords[i+1]*self.scale_factor)
                          for i in range(0, len(coords), 2)]
                self.draw_polygon_selection_box(scaled)
                
                return
    
        # Check full polygon body
        for item in self.polygon_items:
            if self.canvas.type(item) == "polygon" and self.canvas.find_withtag(tk.CURRENT):
                ann_idx, _ = self.polygon_items[item]
                class_id = self.area_data["annotations"][ann_idx]["category_id"]
                class_name = next((c["name"] for c in self.area_data["categories"] if c["id"] == class_id), "")
                self.active_selection = ("polygon_ann", ann_idx)
                if self.inspector is None:
                    self.open_inspector()
                
                self.update_inspector_from_selection()
                self.class_var.set(class_name)
                
                return
            
        self.active_selection = None
        self.clear_selection_box()

    def on_drag(self, event):
        if self.dragging:
            x, y = event.x, event.y
            
            
            if self.active_selection and self.active_selection[0] == "point":
                item = self.active_selection[1]
                i = self.point_items[item]
                self.canvas.coords(item, x - 5, y - 5, x + 5, y + 5)
                self.point_df.at[i, "x"] = float(x / self.scale_factor)
                self.point_df.at[i, "y"] = float(y / self.scale_factor)
                self.draw_point_selection_box(x, y)
                self.save_annotations()
    
            elif self.active_selection and self.active_selection[0] == "polygon":
                item, ann_idx, coord_idx = self.active_selection[1]
                coords = self.area_data["annotations"][ann_idx]["segmentation"][0]
                coords[coord_idx] = x / self.scale_factor
                coords[coord_idx + 1] = y / self.scale_factor
                scaled_coords = [(coords[i] * self.scale_factor, coords[i + 1] * self.scale_factor)
                                 for i in range(0, len(coords), 2)]
                self.canvas.coords(item, *[v for p in scaled_coords for v in p])
                handle = None
                for h, (it, aid, idx) in self.vertex_handles.items():
                    if it == item and aid == ann_idx and idx == coord_idx:
                        handle = h
                        break
                if handle:
                    self.canvas.coords(handle, x - 4, y - 4, x + 4, y + 4)
                self.save_annotations()
                
                if hasattr(self, 'reporter_update_callback'):
                    self.reporter_update_callback()
                self.draw_polygon_selection_box(scaled_coords)
                self.update_inspector_from_selection()

    def on_release(self, event):
        self.dragging = False
        # Do NOT reset self.active_selection here!

    def update_class(self, new_class):
        updated = False
        print("Class update for selection:", self.active_selection)
        
            
        if self.active_selection:
            sel_type, data = self.active_selection
    
            if sel_type == "point":
                i = self.point_items[data]
                self.point_df.at[i, "class"] = new_class
                updated = True
    
            elif sel_type == "polygon":
                _, ann_idx, _ = data
                cat = next((c for c in self.area_data["categories"] if c["name"] == new_class), None)
                if cat:
                    self.area_data["annotations"][ann_idx]["category_id"] = cat["id"]
                    updated = True
    
            elif sel_type == "polygon_ann":
                ann_idx = data
                cat = next((c for c in self.area_data["categories"] if c["name"] == new_class), None)
                if cat:
                    self.area_data["annotations"][ann_idx]["category_id"] = cat["id"]
                    updated = True
    
        if updated:
            print("Updating annotation")
            self.save_annotations()
    
            # ✅ Reload updated data from disk
            self.point_df = pd.read_csv(self.point_csv_path, header=None,
                                        names=["class", "x", "y", "image_name", "width", "height"])
            self.point_df = self.point_df.astype({"x": float, "y": float})
            with open(self.area_json_path, 'r') as f:
                self.area_data = json.load(f)
    
            self.load_image()
            
            if hasattr(self, 'reporter_update_callback'):
                self.reporter_update_callback()


    def save_annotations(self):
        self.point_df.to_csv(self.point_csv_path, header=False, index=False)
        with open(self.area_json_path, "w") as f:
            json.dump(self.area_data, f, indent=2)


class AnnotationReporter:
    def __init__(self, tool_instance):
        self.tool = tool_instance

        self.window = tk.Toplevel()
        self.window.title("Annotation Report")
        self.window.geometry("800x400")

        self.tree = ttk.Treeview(self.window, columns=("class", "x", "y", "area", "length"), show="headings")
        for col in ("class", "x", "y", "area", "length"):
            self.tree.heading(col, text=col.capitalize())
            self.tree.column(col, width=100, anchor='center')
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.save_button = tk.Button(self.window, text="Save as CSV", command=self.save_table_as_csv)
        self.save_button.pack(pady=10)

        self.update_table()
        # Set up callback so UI can update the table when needed
        self.tool.reporter_update_callback = self.update_table

    def update_table(self):
        # Clear existing table
        for row in self.tree.get_children():
            self.tree.delete(row)

        # Reload updated annotations
        with open(self.tool.area_json_path, 'r') as f:
            area_data = json.load(f)

        image_name = self.tool.image_name
        image_id = next((img['id'] for img in area_data['images'] if img['file_name'] == image_name), None)
        if image_id is None:
            return

        for ann in area_data["annotations"]:
            if ann["image_id"] != image_id:
                continue

            # Skip annotations that are points
            if "segmentation" not in ann or not ann["segmentation"] or not isinstance(ann["segmentation"][0], list):
                continue

            class_id = ann["category_id"]
            class_name = next((c["name"] for c in area_data["categories"] if c["id"] == class_id), "")

            try:
                coords = ann["segmentation"][0]
                poly = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                poly = np.array(poly)
                center_x, center_y = poly[:,0].mean(), poly[:,1].mean()

                centerline, length, area = get_worm_centerline(coords,plot=False)
                length /= self.tool.pixelsize
                area /= self.tool.pixelsize

                self.tree.insert("", "end", values=(
                    class_name,
                    f"{center_x:.1f}",
                    f"{center_y:.1f}",
                    f"{area:.1f}",
                    f"{length:.1f}"
                ))

            except Exception as e:
                print(f"Error processing annotation: {e}")

    def save_table_as_csv(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["class", "x", "y", "area", "length"])
            for row_id in self.tree.get_children():
                row_data = self.tree.item(row_id)['values']
                writer.writerow(row_data)

        # Image export logic
        export_dir = os.path.dirname(file_path)

        with open(self.tool.area_json_path, 'r') as f:
            area_data = json.load(f)

        image_name = self.tool.image_name
        image_id = next((img['id'] for img in area_data['images'] if img['file_name'] == image_name), None)
        if image_id is None:
            return

        image_path = os.path.join(self.tool.image_dir, image_name)
        cv_image = cv2.imread(image_path)
        instance_counter = {}

        for ann in area_data["annotations"]:
            if ann["image_id"] != image_id:
                continue

            if "bbox" not in ann:
                continue

            class_id = ann["category_id"]
            class_name = next((c["name"] for c in area_data["categories"] if c["id"] == class_id), "")
            x, y, w, h = map(int, ann["bbox"])
            cutout = cv_image[y:y+h, x:x+w]

            instance_counter[class_name] = instance_counter.get(class_name, 0) + 1
            export_name = f"{class_name}_{image_name}_{instance_counter[class_name]}.png"
            export_path = os.path.join(export_dir, export_name)
            cv2.imwrite(export_path, cutout)


if __name__ == "__main__":
    # Allow command-line usage: python script.py base_dir image_dir output_dir
    base_dir = sys.argv[1]
    image_dir = base_dir + "/images/"
 
    point_csv = base_dir + "/output/" + "annotations_points.csv"
    area_json = base_dir + "/output/" + "annotations_areas.json"
    color_csv = base_dir + '/' + "class_colors.csv" 
    print(base_dir)
    print(image_dir)
    print(area_json)
    print(color_csv)
    
    tool = AnnotationTool( image_dir=image_dir,
                                  point_csv=point_csv, 
                                  area_json=area_json, color_csv=color_csv )
            
