import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter import filedialog

class ImageViewer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Image Viewer")
        self.geometry("800x600")
        self.image_list = []
        self.current_image_index = 0
        self.image_label = tk.Label(self)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        self.create_widgets()

    def create_widgets(self):
        # Button to open folder and load images
        open_button = tk.Button(self, text="Open Folder", command=self.open_folder)
        open_button.pack(side=tk.TOP, fill=tk.X)

        # Navigation buttons
        prev_button = tk.Button(self, text="Previous", command=self.show_previous_image)
        prev_button.pack(side=tk.LEFT, padx=10, pady=5)

        next_button = tk.Button(self, text="Next", command=self.show_next_image)
        next_button.pack(side=tk.LEFT, padx=10, pady=5)

        exit_button = tk.Button(self, text="Exit", command=self.quit)
        exit_button.pack(side=tk.RIGHT, padx=10, pady=5)

    def open_folder(self):
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return

        # Get all images (JPEG, PNG, BMP) in the folder
        self.image_list = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if not self.image_list:
            messagebox.showerror("Error", "No images found in this folder.")
            return
        
        # Reset index to show the first image
        self.current_image_index = 0
        self.display_image(folder_path)

    def display_image(self, folder_path):
        image_path = os.path.join(folder_path, self.image_list[self.current_image_index])
        try:
            # Open the image using Pillow
            img = Image.open(image_path)
            
            # Use LANCZOS resampling for high-quality resizing
            img = img.resize((self.winfo_width(), self.winfo_height()), Image.Resampling.LANCZOS)  # Resize image to fit window
            img = ImageTk.PhotoImage(img)  # Convert image for Tkinter display
            self.image_label.config(image=img)
            self.image_label.image = img  # Keep a reference to avoid garbage collection
        except Exception as e:
            # Show an error message if image loading fails
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            print(f"Error loading image {image_path}: {str(e)}")  # Print error to console

    def show_previous_image(self):
        if self.image_list and self.current_image_index > 0:
            self.current_image_index -= 1
            folder_path = filedialog.askdirectory()
            self.display_image(folder_path)

    def show_next_image(self):
        if self.image_list and self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            folder_path = filedialog.askdirectory()
            self.display_image(folder_path)

if __name__ == "__main__":
    app = ImageViewer()
    app.mainloop()
