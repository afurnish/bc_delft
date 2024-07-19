import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import json
import os
import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import ttide as tt
import threading
from tkcalendar import DateEntry
import sys

CONFIG_DIR = "confs"
MAIN_CONFIG_FILE = os.path.join(CONFIG_DIR, "config.conf")
TIDAL_CONSTITUENTS = {'2N2': '2N2  ', 'K1': 'K1  ', 'K2': 'K2  ', 'M2': 'M2  ', 'M4': 'M4  ', 'MF': 'MF  ', 'MM': 'MM  ', 'MN4': 'MN4 ', 'MS4': 'MS4 ', 'N2': 'N2  ', 'O1': 'O1  ', 'P1': 'P1  ', 'Q1': 'Q1  ', 'S1': 'S1  ', 'S2': 'S2  '}
OUTPUT_OPTIONS = ['Delft 3D .csv Outputs', 'Delft 3D Ocean Input Files .bc Outputs', 'Delft 3D Harmonic Outputs']

class TidalModelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tidal Model Driver")

        if not os.path.exists(CONFIG_DIR):
            os.makedirs(CONFIG_DIR)

        self.config = self.load_config(MAIN_CONFIG_FILE)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both')

        self.date_timestep_frame = ttk.Frame(self.notebook)
        self.tidal_constituents_frame = ttk.Frame(self.notebook)
        self.input_files_frame = ttk.Frame(self.notebook)
        self.output_options_frame = ttk.Frame(self.notebook)
        self.progress_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.date_timestep_frame, text='Dates & Timestep')
        self.notebook.add(self.tidal_constituents_frame, text='Tidal Constituents')
        self.notebook.add(self.input_files_frame, text='Input Files')
        self.notebook.add(self.output_options_frame, text='Outputs')
        self.notebook.add(self.progress_frame, text='Progress Information')

        self.setup_date_timestep_tab()
        self.setup_tidal_constituents_tab()
        self.setup_input_files_tab()
        self.setup_output_options_tab()
        self.setup_progress_tab()

        self.run_button = tk.Button(root, text="Run with this Profile", command=self.run_profile, font=('Helvetica', 14, 'bold'), pady=10, padx=10)
        self.run_button.pack(pady=10)

        self.save_button = tk.Button(root, text="Save Configuration", command=self.save_config_as)
        self.save_button.pack(pady=10)

        self.load_button = tk.Button(root, text="Load Configuration", command=self.load_config_dialog)
        self.load_button.pack(pady=10)

    def setup_date_timestep_tab(self):
        self.date_entries = {}
        for key in ["Start Date", "End Date"]:
            frame = tk.Frame(self.date_timestep_frame)
            frame.pack(fill=tk.X, pady=5)

            label = tk.Label(frame, text=key, width=20)
            label.pack(side=tk.LEFT)

            date_str = self.config.get(key, "")
            if not date_str:
                date_str = datetime.now().strftime('%Y-%m-%d')

            date_entry = DateEntry(frame, date_pattern='yyyy-mm-dd')
            date_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            date_entry.set_date(datetime.strptime(date_str, '%Y-%m-%d'))
            self.date_entries[key] = date_entry

            time_entry = ttk.Combobox(frame, values=[f"{h:02}:00" for h in range(24)])
            time_entry.set(self.config.get(f"{key} Time", "00:00"))
            time_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.date_entries[f"{key} Time"] = time_entry

        frame = tk.Frame(self.date_timestep_frame)
        frame.pack(fill=tk.X, pady=5)

        label = tk.Label(frame, text="Timestep", width=20)
        label.pack(side=tk.LEFT)

        timestep_entry = tk.Entry(frame)
        timestep_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        timestep_entry.insert(0, self.config.get("Timestep", ""))
        self.date_entries["Timestep"] = timestep_entry

    def setup_tidal_constituents_tab(self):
        self.constituent_vars = {}
        for key in TIDAL_CONSTITUENTS.keys():
            var = tk.BooleanVar(value=self.config.get(key, False))
            checkbutton = tk.Checkbutton(self.tidal_constituents_frame, text=key, variable=var)
            checkbutton.pack(anchor='w')
            self.constituent_vars[key] = var

    def setup_input_files_tab(self):
        self.file_entries = {}
        for key in ["TPXO9 Directory", ".pli File"]:
            frame = tk.Frame(self.input_files_frame)
            frame.pack(fill=tk.X, pady=5)

            label = tk.Label(frame, text=key, width=20)
            label.pack(side=tk.LEFT)

            entry = tk.Entry(frame)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            entry.insert(0, self.config.get(key, ""))
            self.file_entries[key] = entry

            button = tk.Button(frame, text="Browse", command=lambda k=key: self.browse_directory(k) if key == "TPXO9 Directory" else self.browse_file(k))
            button.pack(side=tk.RIGHT)

        pli_description = tk.Label(self.input_files_frame, text="The .pli file generated by Delft that contains the location information of boundary points")
        pli_description.pack(fill=tk.X, pady=5)

    def setup_output_options_tab(self):
        self.output_vars = {}
        for option in OUTPUT_OPTIONS:
            var = tk.BooleanVar(value=self.config.get(option, False))
            checkbutton = tk.Checkbutton(self.output_options_frame, text=option, variable=var)
            checkbutton.pack(anchor='w')
            self.output_vars[option] = var

        frame = tk.Frame(self.output_options_frame)
        frame.pack(fill=tk.X, pady=5)

        label = tk.Label(frame, text="Output Path", width=20)
        label.pack(side=tk.LEFT)

        output_entry = tk.Entry(frame)
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        output_entry.insert(0, self.config.get("Output Path", ""))
        self.file_entries["Output Path"] = output_entry

        button = tk.Button(frame, text="Browse", command=lambda: self.browse_directory("Output Path"))
        button.pack(side=tk.RIGHT)

        output_description = tk.Label(self.output_options_frame, text="Select the directory for output files")
        output_description.pack(fill=tk.X, pady=5)

    def setup_progress_tab(self):
        self.progress_text = tk.Text(self.progress_frame, wrap='word', state='disabled')
        self.progress_text.pack(expand=True, fill='both')

    def browse_file(self, key):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.file_entries[key].delete(0, tk.END)
            self.file_entries[key].insert(0, file_path)

    def browse_directory(self, key):
        directory_path = filedialog.askdirectory()
        if directory_path:
            self.file_entries[key].delete(0, tk.END)
            self.file_entries[key].insert(0, directory_path)

    def load_config(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as file:
                return json.load(file)
        return {}

    def save_config(self):
        config = {key: entry.get() for key, entry in self.date_entries.items()}
        config.update({key: entry.get() for key, entry in self.file_entries.items()})
        config.update({key: var.get() for key, var in self.constituent_vars.items()})
        config.update({key: var.get() for key, var in self.output_vars.items()})
        with open(MAIN_CONFIG_FILE, 'w') as file:
            json.dump(config, file)
        return config

    def save_config_as(self):
        config = self.save_config()
        file_path = filedialog.asksaveasfilename(defaultextension=".conf", initialdir=CONFIG_DIR, filetypes=[("Config files", "*.conf")])
        if file_path:
            with open(file_path, 'w') as file:
                json.dump(config, file)
            self.log(f"Configuration saved to {file_path}")

    def load_config_dialog(self):
        file_path = filedialog.askopenfilename(initialdir=CONFIG_DIR, filetypes=[("Config files", "*.conf")])
        if file_path:
            self.config = self.load_config(file_path)
            self.update_entries()
            self.log(f"Configuration loaded from {file_path}")

    def update_entries(self):
        for key, entry in self.date_entries.items():
            if 'Time' in key:
                entry.set(self.config.get(key, "00:00"))
            else:
                entry.set_date(datetime.strptime(self.config.get(key, datetime.now().strftime('%Y-%m-%d')), '%Y-%m-%d'))
        for key, entry in self.file_entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, self.config.get(key, ""))
        for key, var in self.constituent_vars.items():
            var.set(self.config.get(key, False))
        for key, var in self.output_vars.items():
            var.set(self.config.get(key, False))

    def run_profile(self):
        self.save_config()
        config = self.load_config(MAIN_CONFIG_FILE)
        selected_constituents = [k for k, v in config.items() if v is True and k in TIDAL_CONSTITUENTS.keys()]
        print(f"Selected Constituents: {selected_constituents}")
        self.log(f"Selected Constituents: {selected_constituents}")

        # Run tidal model in a separate thread
        thread = threading.Thread(target=self.run_tidal_model, args=(config,))
        thread.start()

    def run_tidal_model(self, config):
        tide_dir = config["TPXO9 Directory"]
        output_path = config["Output Path"]
        pli_file = config[".pli File"]
        start_date = f"{config['Start Date']} {config['Start Date Time']}"
        end_date = f"{config['End Date']} {config['End Date Time']}"
        timestep = config["Timestep"]
        selected_constituents = [TIDAL_CONSTITUENTS[k] for k, v in config.items() if v is True and k in TIDAL_CONSTITUENTS.keys()]
        output_options = {k: v for k, v in config.items() if k in OUTPUT_OPTIONS}

        if not selected_constituents:
            self.log("No tidal constituents selected.")
            messagebox.showwarning("Warning", "No tidal constituents selected.")
            return

        self.log(f"Running Tidal Model with the following settings:")
        self.log(f"TPXO9 Directory: {tide_dir}")
        self.log(f"Output Path: {output_path}")
        self.log(f".pli File: {pli_file}")
        self.log(f"Start Date: {start_date}")
        self.log(f"End Date: {end_date}")
        self.log(f"Timestep: {timestep}")
        self.log(f"Selected Constituents: {selected_constituents}")
        self.log(f"Output Options: {output_options}")

        try:
            self.generate_tidal_model(tide_dir, output_path, pli_file, start_date, end_date, timestep, selected_constituents, output_options)
        except Exception as e:
            self.log(f"Error: {e}")

        messagebox.showinfo("Run", "Running with the current profile...")

    def generate_tidal_model(self, tide_dir, output_path, pli_file, start_date, end_date, timestep, selected_constituents, output_options):
        script_path = os.path.abspath(sys.argv[0])[:-7]
        tidal_consts_list = [i.replace(' ','') for i in ['2N2 ','K1  ','K2  ','M2  ','M4  ','MF  ','MM  ','MN4 ','MS4 ','N2  ','O1  ','P1  ','Q1  ','S1  ','S2  ']]

        self.log('Loading grid dataset...')
        grid_ds = xr.open_dataset(glob.glob(tide_dir + r'/grid*.nc')[0])
        self.log('Grid metadata loaded:')
        self.log(grid_ds)

        lat_y = np.array(grid_ds['lat_z'])
        lon_x = np.array(grid_ds['lon_z'])
        hz = np.array(np.fliplr(np.rot90((grid_ds['hz']), axes=(1, 0))))
        self.log("Processed grid data.")

        neg_lon = (360 - lon_x[int(len(lon_x)/2):(len(lon_x))]) * -1
        pos_lon = lon_x[0 : int(len(lon_x)/2)]
        new_lon = np.concatenate((neg_lon, pos_lon), axis=0)
        new_lon[5399] = 0  # some reason it thinks it's not 0

        neg_hz = hz[:, int(len(lon_x)/2):(len(lon_x))]
        pos_hz = hz[:, 0 : int(len(lon_x)/2)]
        new_hz = np.concatenate((neg_hz, pos_hz), axis=1)
        new_lat = np.array(grid_ds['lat_z'])

        uk = [-11, 2, 49, 61]
        imin = self.find_nearest(new_lon, uk[0])
        imax = self.find_nearest(new_lon, uk[1])
        jmin = self.find_nearest(lat_y, uk[2])
        jmax = self.find_nearest(lat_y, uk[3])

        mx = int(np.array(np.where(new_lon == imin))[0][0])  # min x
        MX = int(np.array(np.where(new_lon == imax))[0][0])  # max x
        my = int(np.array(np.where(lat_y == jmin))[0][0])  # min y
        MY = int(np.array(np.where(lat_y == jmax))[0][0])  # max y

        UK_hz = new_hz[my:MY, mx:MX]  # slice the UK
        UK_lon = new_lon[mx:MX]
        UK_lat = new_lat[my:MY]

        self.log("Generated location data.")

        locs_of_boundary = pd.read_csv(pli_file, delimiter=(' '), header=1, usecols=[0, 2, 3], names=['Lon', 'Lat', 'Name'])
        # self.log(locs_of_boundary)
        loc_x = locs_of_boundary.Lon
        loc_y = locs_of_boundary.Lat
        lent = len(locs_of_boundary['Name'])

        HX = [int(np.array(np.where(UK_lon == self.find_nearest(UK_lon, loc_x[i])))[0][0]) for i in range(lent)]
        HY = [int(np.array(np.where(UK_lat == self.find_nearest(UK_lat, loc_y[i])))[0][0]) for i in range(lent)]

        # new_path = os.path.join(tide_dir, 'TPXO9_atlas_v5_nc')
        h_path = glob.glob(tide_dir + r'/h*')
        constants = [self.format_string(h.split('_')[-5]) for h in h_path]

        v_consts = [xr.open_dataset(h) for h in h_path]
        freq_df = pd.read_csv('tide_cons/tidal_conts_&_freqs.csv')
        # self.log(freq_df)
        # self.log(constants) # This is empty as well
        dataframe_list = [np.array(freq_df.loc[freq_df['Names'].str.contains(c, case=False)]) for c in constants]
        
        # self.log(dataframe_list) # The issue is this is empty
        
        df_freq = pd.DataFrame(np.concatenate(dataframe_list), columns=["Name", "freq", "Name_Lower"])
        df_freq['print_name'] = df_freq['Name'].str.replace(" ", "")

        self.log("Loaded and processed tidal constituent data.")

        start = datetime.strptime(start_date, '%Y-%m-%d %H:%M')
        end = datetime.strptime(end_date, '%Y-%m-%d %H:%M')
        total_minutes = int((end - start).total_seconds() / 60)
        timestep_minutes = int(timestep)
        t = [start + timedelta(minutes=i) for i in range(0, total_minutes + 1, timestep_minutes)]

        con = np.char.ljust(np.char.upper(selected_constituents), 4)
        cons = con.reshape(-1, 1)

        amp, pha, fre, consts_names_rearanged = [], [], [], []

        for i, file in enumerate(sorted(glob.glob(output_path + r'/*.csv'))):
            df = pd.read_csv(file)
            amplitudes, phases, frequencies = [], [], []
            for _, row in df.iterrows():
                name = self.format_string(row['TC'])
                if name in cons:
                    amplitude = row['Amplitude_(m)']
                    phase = row['Phase_(Deg)']
                    frequency = row['Freq_(deg_hour)']
                    amplitudes.append(amplitude)
                    phases.append(phase)
                    frequencies.append(frequency)
                    if i == 0:
                        consts_names_rearanged.append(name)
            amp.append(amplitudes)
            pha.append(phases)
            fre.append(frequencies)
        
        self.log(output_path + r'/*.csv')
        con2 = np.char.ljust(np.char.upper(consts_names_rearanged), 4)
        
        cons2 = con2.reshape(-1, 1)
        '''
        The issue is with con2
        Now the usse is conts names rearranged is emmpty.
        Its trying to find a csv file that doesnt exist in the target location. This csv file contains harmonic information
        
        
        
        '''
        for i, file in enumerate(sorted(glob.glob(output_path + r'/*.csv'))):
            
            freq = fre[i]
            FREQ = np.array(freq)

            tidecons = [[amp[i][j], 0, pha[i][j], 0] for j in range(len(cons2))]
            tidecons2 = np.array(tidecons).astype(float)
            eta = tt.t_predic(np.array(t), cons2, FREQ, tidecons2)

            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(t, eta)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.xticks(rotation=45)
            plt.xlim([t[0], t[0] + timedelta(days=3)])
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, '3_days.png'), dpi=200)

            fig2, ax2 = plt.subplots(figsize=(7, 3))
            ax2.plot(t, eta)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, '0all_days.png'), dpi=200)

            df2 = pd.DataFrame(eta, t)
            csv_filename = r'/point' + r'_' + str(i + 1) + r'.csv'
            to_write = os.path.join(output_path, csv_filename)
            df2.to_csv(to_write, header=False)
        
        if output_options.get('Delft 3D Ocean Input Files .bc Outputs', False):
            self.generate_bc_output(output_path, start_date, locs_of_boundary, t, output_path, df_freq, amp, pha, fre)

    def generate_bc_output(self, output_path, start_date, locs_of_boundary, t, new_folder2, df_freq, amp, pha, fre):
        bc_file = os.path.join(output_path, 'WaterLevel.bc')
        with open(bc_file, "w") as f:
            f.write("")
            f.close()

        seconds_since = str(start_date)
        for i, file in enumerate(sorted(glob.glob(new_folder2 + '/*.csv'))):
            surface_height = pd.read_csv(file, names=['time', 'surface_height'])
            self.write_text_block_to_file_H(bc_file, locs_of_boundary.Name[i], str(t[0]))
            data_to_write = surface_height.surface_height
            new_time = [i[:-3] for i in surface_height.time]
            converted_timeseries = [int(i) for i in self.convert_to_seconds_since_date(new_time, seconds_since)]

            with open(bc_file, "a") as f:
                for j, data in enumerate(data_to_write):
                    f.write(f"{converted_timeseries[j]}    {data}\n")
                f.write('\n')

        self.log(f".bc file generated at {bc_file}")

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def format_string(self, input_string):
        capitalized_string = input_string.upper()
        formatted_string = capitalized_string.ljust(4)
        return formatted_string

    def write_text_block_to_file_H(self, file_path, name, date):
        with open(file_path, "a") as f:
            f.write("[forcing]\n")
            f.write(f"Name                            = {name}\n")
            f.write("Function                        = timeseries\n")
            f.write("Time-interpolation              = linear\n")
            f.write("Quantity                        = time\n")
            f.write(f"Unit                            = seconds since {date}\n")
            f.write("Quantity                        = waterlevelbnd\n")
            f.write("Unit                            = m\n")

        self.log(f"Text block with Name = '{name}' written to {file_path} successfully!")

    def convert_to_seconds_since_date(self, timeseries, date_str):
        reference_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        converted_timeseries = []
        for point in timeseries:
            point_date = datetime.strptime(point, "%Y-%m-%d %H:%M")
            time_difference = (point_date - reference_date).total_seconds()
            converted_timeseries.append(time_difference)
        return converted_timeseries

    def log(self, message):
        self.progress_text.config(state='normal')
        self.progress_text.insert(tk.END, str(message) + '\n')
        self.progress_text.see(tk.END)
        self.progress_text.config(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    app = TidalModelApp(root)
    root.mainloop()
