import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, Label, Entry
from PIL import Image, ImageTk
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import variable
from image_processing import process_image

# error_folder = variable.error_folder
preview_output_res = variable.preview_output_res
preview_debug_max_res = variable.preview_debug_max_res

class ImageProcessingGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Image Processing GUI")
        self.geometry("800x600")

        self.selected_crop_type = tk.StringVar()
        self.crop_type = 1
        self.show_preview = tk.BooleanVar()
        self.parallel_processing = tk.BooleanVar()
        self.output_res = 1080

        self.create_widgets()

    def create_widgets(self):
        # Select Crop Type
        crop_type_frame = tk.LabelFrame(self, text="Crop Type")
        crop_type_frame.pack(pady=10)

        crop_type_options = [
            ("Upper Body", "1"),
            ("Face", "2"),
            ("Full Body", "3")
        ]

        for text, value in crop_type_options:
            radio_button = tk.Radiobutton(crop_type_frame, text=text, variable=self.selected_crop_type, value=value)
            radio_button.pack(anchor="w", padx=10, pady=5)

        # Show Preview
        preview_frame = tk.LabelFrame(self, text="Preview")
        preview_frame.pack(pady=10)

        preview_checkbox = tk.Checkbutton(preview_frame, text="Show Preview Window", variable=self.show_preview)
        preview_checkbox.pack(anchor="w", padx=10, pady=5)

        # Parallel Processing
        parallel_frame = tk.LabelFrame(self, text="Parallel Processing")
        parallel_frame.pack(pady=10)

        parallel_checkbox = tk.Checkbutton(parallel_frame, text="Enable Parallel Processing", variable=self.parallel_processing)
        parallel_checkbox.pack(anchor="w", padx=10, pady=5)

        # Select Input Folder
        input_folder_frame = tk.LabelFrame(self, text="Input Folder")
        input_folder_frame.pack(pady=10)

        input_folder_label = tk.Label(input_folder_frame, text="Select Input Folder:")
        input_folder_label.pack(side="left", padx=10)

        self.input_folder_entry = tk.Entry(input_folder_frame, width=50)
        self.input_folder_entry.pack(side="left", padx=5)

        input_folder_button = tk.Button(input_folder_frame, text="Browse", command=self.browse_input_folder)
        input_folder_button.pack(side="left", padx=5)

        # Select output Folder
        output_folder_frame = tk.LabelFrame(self, text="Output Folder")
        output_folder_frame.pack(pady=10)

        output_folder_label = tk.Label(output_folder_frame, text="Select Output Folder:")
        output_folder_label.pack(side="left", padx=10)

        self.output_folder_entry = tk.Entry(output_folder_frame, width=50)
        self.output_folder_entry.pack(side="left", padx=5)

        output_folder_button = tk.Button(output_folder_frame, text="Browse", command=self.browse_output_folder)
        output_folder_button.pack(side="left", padx=5)


        output_res_label = Label(self, text="Output Resolution:")
        output_res_label.pack()

        self.output_res_entry = Entry(self)
        self.output_res_entry.insert(0, str(self.output_res))
        self.output_res_entry.pack()

        # Start Processing
        process_button = tk.Button(self, text="Start Processing", command=self.start_processing)
        process_button.pack(pady=20)

    def browse_input_folder(self):
        input_folder = filedialog.askdirectory()
        self.input_folder_entry.delete(0, tk.END)
        self.input_folder_entry.insert(0, input_folder)
    def browse_output_folder(self):
        output_folder = filedialog.askdirectory()
        self.output_folder_entry.delete(0, tk.END)
        self.output_folder_entry.insert(0, output_folder)

    def start_processing(self):
        crop_type = self.selected_crop_type.get()
        show_preview = self.show_preview.get()
        input_folder = self.input_folder_entry.get()
        output_folder = self.output_folder_entry.get()

        if not input_folder:
            messagebox.showerror("Error", "Please select an input folder.")
            return

        if not os.path.isdir(input_folder):
            messagebox.showerror("Error", "Invalid input folder.")
            return

        self.output_res = int(self.output_res_entry.get())
        error_folder = os.path.join(output_folder, "error_folder")

        if crop_type == "1":
            top_margin_value = 1
            bottom_margin_value = 3
            debug_output = os.path.join(output_folder, "upperbody_debug")
            output_folder = os.path.join(output_folder, "upperbody_cropped")
        elif crop_type == "2":
            top_margin_value = 1
            bottom_margin_value = 1
            debug_output = os.path.join(output_folder, "face_debug")
            output_folder = os.path.join(output_folder, "face_cropped")
        elif crop_type == "3":
            top_margin_value = 1
            bottom_margin_value = 12
            debug_output = os.path.join(output_folder, "fullbody_debug")
            output_folder = os.path.join(output_folder, "fullbody_cropped")

        if show_preview:
            self.process_images_single(input_folder, output_folder, debug_output, crop_type, show_preview, top_margin_value, bottom_margin_value, error_folder)
        else:
            self.process_images_parallel(input_folder, output_folder, debug_output, crop_type, show_preview, top_margin_value, bottom_margin_value, error_folder)


    def process_images_parallel(self, input_folder, output_folder, debug_output, crop_type, show_preview, top_margin_value, bottom_margin_value, error_folder):
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(debug_output, exist_ok=True)
        os.makedirs(error_folder, exist_ok=True)

        input_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]
        total_files = len(input_files)

        output_res = self.output_res
        with ThreadPoolExecutor() as executor:
            futures = []
            progress_bar = tqdm(total=total_files, desc="Processing Images", dynamic_ncols=True)

            for image_path in input_files:
                args = (
                    image_path,
                    error_folder,
                    output_folder,
                    debug_output,
                    output_res,
                    preview_output_res,
                    preview_debug_max_res,
                    show_preview,
                    crop_type,
                    top_margin_value,
                    bottom_margin_value,
                )
                future = executor.submit(self.process_image_worker, args)
                future.add_done_callback(lambda f: progress_bar.update(1))
                futures.append(future)

            for future in futures:
                future.result()

        cv2.destroyAllWindows()
        self.display_summary(total_files, output_folder)

    def process_image_worker(self, args):
        error_count = process_image(*args)
        return error_count

    def process_images_single(self, input_folder, output_folder, debug_output, crop_type, show_preview, top_margin_value, bottom_margin_value, error_folder):
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(debug_output, exist_ok=True)
        os.makedirs(error_folder, exist_ok=True)

        input_files = [os.path.join(input_folder, file) for file in os.listdir(input_folder)]
        total_files = len(input_files)

        progress_bar = tqdm(total=total_files, desc="Processing Images", dynamic_ncols=True)
        error_count = 0
        output_res = self.output_res
        for image_path in input_files:
            error_occurred = process_image(
                image_path,
                error_folder,
                output_folder,
                debug_output,
                output_res,
                preview_output_res,
                preview_debug_max_res,
                show_preview,
                crop_type,
                top_margin_value,
                bottom_margin_value,
            )
            if error_occurred:
                error_count += 1
            progress_bar.update(1)

        cv2.destroyAllWindows()
        self.display_summary(total_files, output_folder, error_count)

    def display_summary(self, total_files, output_folder, error_count=0):
        total_output_images = len(os.listdir(output_folder))

        summary_text = f"Total images: {total_files}\n"
        summary_text += f"Processed images: {total_files - error_count}\n"
        summary_text += f"Total faces detected: {total_output_images}\n"
        summary_text += f"Error images: {error_count}"

        messagebox.showinfo("Processing Complete", summary_text)


if __name__ == '__main__':
    app = ImageProcessingGUI()
    app.mainloop()
