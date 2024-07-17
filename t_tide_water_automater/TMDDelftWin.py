import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import json
import os

CONFIG_FILE = "config.json"
TIDAL_CONSTITUENTS = ['2N2 ', 'K1  ', 'K2  ', 'M2  ', 'M4  ', 'MF  ', 'MM  ', 'MN4 ', 'MS4 ', 'N2  ', 'O1  ', 'P1  ', 'Q1  ', 'S1  ', 'S2  ']

class TidalModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tidal Model Driver")

        self.config = self.load_config()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both')

        self.file_paths_frame = ttk.Frame(self.notebook)
        self.tidal_constituents_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.file_paths_frame, text='File Paths')
        self.notebook.add(self.tidal_constituents_frame, text='Tidal Constituents')

        self.setup_file_paths_tab()
        self.setup_tidal_constituents_tab()

        self.run_button = tk.Button(root, text="Run with this Profile", command=self.run_profile)
        self.run_button.pack(pady=10)

    def setup_file_paths_tab(self):
        self.file_entries = {}
        for key in ["Input Path", "Output Path"]:
            frame = tk.Frame(self.file_paths_frame)
            frame.pack(fill=tk.X, pady=5)

            label = tk.Label(frame, text=key, width=20)
            label.pack(side=tk.LEFT)

            entry = tk.Entry(frame)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            entry.insert(0, self.config.get(key, ""))
            self.file_entries[key] = entry

            button = tk.Button(frame, text="Browse", command=lambda k=key: self.browse_file(k))
            button.pack(side=tk.RIGHT)

    def setup_tidal_constituents_tab(self):
        self.constituent_vars = {}
        for constituent in TIDAL_CONSTITUENTS:
            var = tk.BooleanVar(value=self.config.get(constituent.strip(), False))
            checkbutton = tk.Checkbutton(self.tidal_constituents_frame, text=constituent.strip(), variable=var)
            checkbutton.pack(anchor='w')
            self.constituent_vars[constituent.strip()] = var

    def browse_file(self, key):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_entries[key].delete(0, tk.END)
            self.file_entries[key].insert(0, file_path)

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as file:
                return json.load(file)
        return {}

    def save_config(self):
        config = {key: entry.get() for key, entry in self.file_entries.items()}
        config.update({key: var.get() for key, var in self.constituent_vars.items()})
        with open(CONFIG_FILE, 'w') as file:
            json.dump(config, file)

    def save_output(self):
        self.save_config()
        output_path = self.file_entries["Output Path"].get()
        if output_path:
            with open(output_path, 'w') as file:
                file.write("Output based on selected options\n")
                for key, entry in self.file_entries.items():
                    file.write(f"{key}: {entry.get()}\n")
                for key, var in self.constituent_vars.items():
                    file.write(f"{key}: {var.get()}\n")
            messagebox.showinfo("Saved", f"Output saved to {output_path}")
        else:
            messagebox.showwarning("No Output Path", "Please specify an output path")

    def run_profile(self):
        self.save_config()
        messagebox.showinfo("Run", "Running with the current profile...")

if __name__ == "__main__":
    root = tk.Tk()
    app = TidalModelApp(root)
    root.mainloop()

