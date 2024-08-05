"""
TDM - Tidal Model Driver

This script generates .pli files when needed, and uses those .pli files to generate boundary condition CSVs and the boundary condition file for use in Delft scripts.
It also generates a tidal height file based on the constituents.

Lines with Aaron are for local testing in Spyder.

Bangor University, School of Ocean Sciences.
Created on Tue May 03 2022
# -*- coding: utf-8 -*- 
@author: Aaron Andrew Furnish
"""

#%% Import necessary libraries
import sys
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
from o_func import opsys

#%% Constants
script_path = os.path.join(opsys(), 'GitHub', 'bc_delft', 't_tide_water_automater', 'TMD.py')
tidal_consts_list = [i.replace(' ', '') for i in [
    '2N2', 'K1', 'K2', 'M2', 'M4', 'MF', 'MM', 'MN4', 'MS4', 'N2', 'O1', 'P1', 'Q1', 'S1', 'S2'
]]
valid_inputs = ['Y', 'y', 'N', 'n']
print('.pli file generation coming soon')

#%% Functions
def file_reader(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 5:
        print("Error: The file does not contain enough lines.")
        sys.exit()
    
    directory1 = lines[0].strip()
    if not os.path.exists(directory1):
        print(f"Directory '{directory1}' does not exist.")
        sys.exit()
    
    string_list = lines[1].strip().split()
    print("Tidal Constituents List:", string_list)
    
    start_date_line = lines[2].strip()
    end_date_line = lines[3].strip()
    timestep_list = lines[4].strip()
    
    return directory1, string_list, start_date_line, end_date_line, timestep_list

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def format_string(input_string):
    return input_string.upper().ljust(4)

def write_text_block_to_file_H(file_path, name, date):
    with open(file_path, "a") as f:
        f.write("[forcing]\n")
        f.write(f"Name                            = {name}\n")
        f.write("Function                        = timeseries\n")
        f.write("Time-interpolation              = linear\n")
        f.write("Quantity                        = time\n")
        f.write(f"Unit                            = seconds since {date}\n")
        f.write("Quantity                        = waterlevelbnd\n")
        f.write("Unit                            = m\n")
    print(f"Text block with Name = '{name}' written to {file_path} successfully!")

def convert_to_seconds_since_date(timeseries, date_str):
    reference_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    return [(datetime.strptime(point, "%Y-%m-%d %H:%M") - reference_date).total_seconds() for point in timeseries]

#%% Set up usage of the example file
try:
    input_path = 'test.txt'
except:
    print("\nError: Insufficient command-line arguments.")
    print("Usage: python3 TMD.py <programming_file.txt>")
    print("\n<programming_file_dir>: is folder containing a.txt file, a .pli file, and 2 sub-dirs")
    print("txt file: Line1: tidal file directory, Line2: output file directory, Line3: Tidal Constituents [M2, S2]")
    print("pli file: should be delft .pli file in delft pli format")
    print("\nCurrent included constituents are " + str(tidal_consts_list).replace('[','').replace(']','').replace("'",""))
    print("\nUsage of: python3 TND.py example\nWill set up an example model output in the example file in results")
    print("\nOutput file directory must contain the boundary condition file with a .pli extension in delft format")
    print("\nPassing just the name of a text file that does not exist \nwill create it and set up the folder system for you")
    sys.exit()

full_output_path = os.path.join(os.path.split(script_path)[0], 'results', input_path[:-4])
files = glob.glob(full_output_path + '/*.txt')

if not files:
    print("Text file does not exist. Would you like to create it? (Y/N)")
    answer = input()
    if answer in valid_inputs:
        if answer in ['Y', 'y']:
            filename = '/' + full_output_path.split('/')[-1]
            with open(full_output_path + filename + '/.txt', 'w') as f:
                f.write("")
            print('Directories and text file created. Please populate it.')
            sys.exit()
        else:
            print('Please set up text file.')
            sys.exit()

tide_dir, consts, start, end, timestep = file_reader(files[0])
print('Tidal Constants Used:', str(consts).replace('[','').replace(']','').replace("'",""))

#%% Setup directories
extension = r'.pli'
files = glob.glob(full_output_path + '/*' + extension)

if files:
    print(f"Pli condition files with extension '{extension}' found:")
    for file in files:
        print(file.split('/')[-1])
else:
    print(f"No files with extension '{extension}' found.")

filename = files[0]
main_folder_path = os.path.join(os.path.split(script_path)[0], 'results', input_path.split('/')[-1].split('.')[0])

sub_dir = os.path.join(main_folder_path, 'tidal_boundary_csvs_per_point')
sub_dir2 = os.path.join(main_folder_path, 'tidal_heights')
os.makedirs(sub_dir, exist_ok=True)
os.makedirs(sub_dir2, exist_ok=True)

# Check tidal constants
for item in consts:
    if item not in tidal_consts_list:
        print(f"Tidal constant '{item}' not found. Exiting.")
        print("Please use a choice of:", str(tidal_consts_list).replace('[','').replace(']','').replace("'",""))
        sys.exit()

print('All inputs entered successfully. Running the rest of the program...\n')

#%% Load data and perform operations
grid_ds = xr.open_dataset(glob.glob(tide_dir + r'/grid*.nc')[0])
print('Printing Grid metadata:')
print(grid_ds)

lat_y = np.array(grid_ds['lat_z'][::50])
lon_x = np.array(grid_ds['lon_z'][::50])
hz = np.array(np.fliplr(np.rot90((grid_ds['hz'][::50, ::50]), axes=(1, 0))))

plt.figure()
plt.pcolor(lon_x, lat_y, hz)
plt.colorbar()
plt.title('Tidal Height')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Process grid data
lat_y = np.array(grid_ds['lat_z'])
lon_x = np.array(grid_ds['lon_z'])
hz = np.array(np.fliplr(np.rot90((grid_ds['hz']), axes=(1, 0))))

neg_lon = (360 - lon_x[int(len(lon_x) / 2):]) * -1
pos_lon = lon_x[:int(len(lon_x) / 2)]
new_lon = np.concatenate((neg_lon, pos_lon), axis=0)

neg_hz = hz[:, int(len(lon_x) / 2):]
pos_hz = hz[:, :int(len(lon_x) / 2)]
new_hz = np.concatenate((neg_hz, pos_hz), axis=1)

# UK boundary coordinates
uk = [-11, 2, 49, 61]
imin = find_nearest(new_lon, uk[0])
imax = find_nearest(new_lon, uk[1])
jmin = find_nearest(lat_y, uk[2])
jmax = find_nearest(lat_y, uk[3])

mx = int(np.where(new_lon == imin)[0])
MX = int(np.where(new_lon == imax)[0])
my = int(np.where(lat_y == jmin)[0])
MY = int(np.where(lat_y == jmax)[0])

UK_hz = new_hz[my:MY, mx:MX]
UK_lon = new_lon[mx:MX]
UK_lat = lat_y[my:MY]

plt.figure()
plt.pcolor(UK_lon, UK_lat, UK_hz, cmap='jet')
plt.colorbar()
plt.title('UK Tidal Height')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Load boundary locations
locs_of_boundary = pd.read_csv(filename, delimiter=' ', header=1, usecols=[0, 2, 3], names=['Lon', 'Lat', 'Name'])
locs_of_boundary['Lat'] = locs_of_boundary['Lat']
loc_x = locs_of_boundary['Lon']
loc_y = locs_of_boundary['Lat']

lent = len(locs_of_boundary)
HX = [int(np.where(UK_lon == find_nearest(UK_lon, x))[0]) for x in loc_x]
HY = [int(np.where(UK_lat == find_nearest(UK_lat, y))[0]) for y in loc_y]

for i in range(len(HX)):
    plt.scatter(UK_lon[HX[i]], UK_lat[HY[i]], label=locs_of_boundary['Name'][i])
plt.legend()
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Boundary Locations')
plt.show()

for i in range(len(HX)):
    x = HX[i]
    y = HY[i]
    bc_csv = pd.DataFrame({
        'DateTime': [],
        'Height': []
    })

    for date in pd.date_range(start=start, end=end, freq=timestep):
        date_str = date.strftime('%Y-%m-%d %H:%M:%S')
        bc_csv = bc_csv.append({
            'DateTime': date_str,
            'Height': UK_hz[y, x]
        }, ignore_index=True)
    
    bc_csv.to_csv(os.path.join(sub_dir, f'boundary_condition_{i+1}.csv'), index=False)
    print(f"Boundary condition CSV {i+1} created.")

# Generate tidal height file
dates = pd.date_range(start=start, end=end, freq=timestep)
heights = [UK_hz[HY[i], HX[i]] for i in range(len(HX))]

height_data = pd.DataFrame({
    'DateTime': dates,
    'Height': heights
})
height_data.to_csv(os.path.join(sub_dir2, 'tidal_heights.csv'), index=False)
print("Tidal heights CSV file created.")

print("Process complete.")
