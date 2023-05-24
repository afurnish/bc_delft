""" TDM - Tidal Model Driver 

This file is set up to generate .pli files when needed

To then use those pli files to generate boundary conditon csvs as well as the 
boundary condition file for use in delft scripts. 

It then generates a tidal height file based on the constituents. 


Lines with Aaron are for local testing in spyder

Bangor University, School of Ocean Sciences. 
Created on Tue May 03 2022
# -*- coding: utf-8 -*- 
@author: Aaron Andrew Furnish
"""

#%% Command line inputs 
import sys
import os
import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib.dates as mdates
from datetime import datetime
import datetime as dtime
import time
from datetime import timedelta
import ttide as tt

from o_functions.start import opsys2; start_path = opsys2()


#%% Constants
script_path = os.path.abspath(sys.argv[0])[:-7]
#script_path = r'/Volumes/PD/GitHub/python-oceanography/Delft 3D FM Suite 2019/8.automated_tides' #Aaron
#tidal_consts_list = ['M2', 'S2']
tidal_consts_list = [i.replace(' ','') for i in ['2N2 ','K1  ','K2  ','M2  ','M4  ','MF  ','MM  ','MN4 ','MS4 ','N2  ','O1  ','P1  ','Q1  ','S1  ','S2  ']]
valid_inputs =  ['Y', 'y', 'N', 'n']

print('.pli file generation coming soon')

#%% Functions
def file_reader(file):  
    with open(file, 'r') as file:
        lines = file.readlines()
    # Extract the directories from the lines
    if len(lines) >= 1:
        directory1 = lines[0].strip()
        # Check if the directories exist
        if os.path.exists(directory1):
            print(f"Directory 1 '{directory1}' exists.")
        else:
            print(f"Directory 1 '{directory1}' does not exist.")
    
    else:
        print("Error: The file does not contain the directory.")
        sys.exit()
        
    if len(lines) >= 2:
        string_list = lines[1].strip().split()
        print("Tidal Constituents List:", string_list)
    else:
        print("Error: The file does not contain any tidal constituents.")
        sys.exit()
        
    if len(lines) >= 3:
        start_date_line = lines[2].strip()  # Assuming the date line is the third line (index 2)
        print("Start Date List:", start_date_line)
    else:
        print("Error: The file does not contain any start_dates.")
        sys.exit()
    
    if len(lines) >= 4:
        end_date_line = lines[3].strip()  # Assuming the date line is the third line (index 2)
        print("End Date List:", end_date_line)
    else:
        print("Error: The file does not contain any end_dates.")
        sys.exit()
        
    if len(lines) >= 5:
        timestep_list = lines[4].strip()  # Assuming the date line is the third line (index 2)
        print("Timestep List:", timestep_list)
    else:
        print("Error: The file does not contain any timeteps.")
        sys.exit()

   
    return directory1, string_list, start_date_line, end_date_line, timestep_list

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def format_string(input_string):
    # Capitalize the string
    capitalized_string = input_string.upper()

    # Add spaces to make the string 4 characters long
    formatted_string = capitalized_string.ljust(4)

    return formatted_string

def write_text_block_to_file_H(file_path, name,date):
    """
    Write a text block to a file with the given file path,
    and a customizable 'Name' value.

    Args:
        file_path (str): The file path to write the text block to.
        name (str): The value for the 'Name' line in the text block.
    """
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
    """
    Convert a timeseries of points to seconds since a specific date.

    Args:
        timeseries (list): The timeseries of points.
        date_str (str): The date to use as the reference, in the format "YYYY-MM-DD HH:MM:SS".

    Returns:
        list: The timeseries of points converted to seconds since the reference date.
    """
    # Convert the date_str to a datetime object
    reference_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    
    # Initialize an empty list to store the converted timeseries
    converted_timeseries = []

    # Loop through each point in the timeseries
    for point in timeseries:
        # Convert the point to a datetime object
        point_date = datetime.strptime(point, "%Y-%m-%d %H:%M")
        
        # Calculate the time difference in seconds between the point and the reference date
        time_difference = (point_date - reference_date).total_seconds()
        
        # Append the converted time difference to the converted timeseries
        converted_timeseries.append(time_difference)
    
    return converted_timeseries

#%%Set up usage of the example file
try:
    input_path = sys.argv[1] 
except:
    print("\nError: Insufficient command-line arguments.")
    print("Usage: python3 TMD.py <programming_file.txt>")
    print("\n<programming_file_dir>: is folder containing  a.txt file, a .pli file and 2 sub-dirs")
    print("txt file: Line1: tidal file directory, Line2: output file directory, Line3: Tidal Constituents [M2, S2]")
    print("pli file: should be delft .pli file in delft pli format")

    print("\nCurrent included constituents are " + str(tidal_consts_list).replace('[','').replace(']','').replace("'",""))
  
    print("\nUsage of: python3 TND.py example\nWill set up an example model output in the example file in results")
    print("\nOutput file directory must contain the boundary condition file with a .pli extension in delft format")
    print("\nPassing just the name of a text file that does not exist \nwill create it and set up the folder system for you")
    #print(len(sys.argv))
    sys.exit()
   
#when running in shell it does not have a / at the end of the script_path but does in spyder
# input_path = '/results/mac_example/example.txt'
#input_path = '/results/example'
#script_path = r'/Volumes/PD/GitHub/python-oceanography/Delft 3D FM Suite 2019/8.automated_tides'
#%%
if input_path == 'example':
    a = 1
else:
    full_output_path = script_path + input_path
    
    if os.path.isdir(full_output_path) == True:
        print('Directory exists, continuing program...')
        
        
        try:
            f = glob.glob(full_output_path + '/*.txt')
            if os.path.exists((f)[0]) == True:
                print('\nText File exists, checking file contents...\n')
                tide_dir, consts, start, end, timestep = file_reader(f[0])
                print(tide_dir)
                print('\nFile checked, continuing to run program...')
        except:
            while True:
                answer = input('Text file does not exist, would you like to create it?')
                if answer in valid_inputs:
                    break
                if answer == 'Y' or 'y':
                    filename = '/' + full_output_path.split('/')[-1]
                    with open(full_output_path + filename +  '/.txt', 'w') as f:
                        f.write("")
                    f.close()
                    print('Directories and text file created, please populate it')
                    sys.exit()
                else:
                    print('Please go back and set up text file')
    else:
        while True:
            answer = input('Directory does not exist, would you like to set up filesystem fully (y/Y or n/N) ?: ')            
            if answer in valid_inputs:
                break
        if answer == 'y' or 'Y': # Makes primary output file if it does not exist
            path = script_path + '/results/'
            new_path = path + input_path.split('/')[-1].split('.')[0]
            os.mkdir(new_path)
            sub_path = new_path + '/tidal_boundary_csvs_per_point'
            os.mkdir(sub_path)
            sub_path2 = new_path + '/tidal_heights'
            os.mkdir(sub_path2)
            file = new_path + '/'+ input_path.split('/')[-1] + '.txt'
            with open(file, 'w') as f:
                f.write("")
            f.close()
            print('Directories and text file created, please populate it')
            sys.exit()
                
        else:
            print('Please go back and check your filesystem setup, exiting program...')
            sys.exit()
    
    
extension = r'.pli'
files = glob.glob(full_output_path + '/*' + extension) # finding boundary conditon file
if files:
    print(f"\Pli condition files with extension '{extension}' found: " )  
    for file in files:
        print(file.split('/')[-1])
        
else:
    print(f"No files with extension '{extension}' found.")

filename = file
    
#Main folder path 
main_folder_path = script_path + '/results/'+ input_path.split('/')[-1].split('.')[0]

#Backup to make file deposit area
sub_dir = script_path + '/results/'+ input_path.split('/')[-1].split('.')[0] + '/tidal_boundary_csvs_per_point'
if os.path.exists(sub_dir) == False:
    os.mkdir(sub_dir)
    
sub_dir2 = script_path + '/results/'+ input_path.split('/')[-1].split('.')[0] + '/tidal_heights'
if os.path.exists(sub_dir2) == False:
    os.mkdir(sub_dir2)


for item in consts:
    if item not in tidal_consts_list:
        print(f"Tidal constant '{item}' not found in the directory of constants. Breaking the code. \nPlease try again")
        print("Please try again using a choice of: " + str(tidal_consts_list).replace('[','').replace(']','').replace("'","") +'\n') 
        sys.exit()
    
print('Tidal Constants Used     = ' + str(consts).replace('[','').replace(']','').replace("'",""))
    
print('All inputs entered sucessfully, running rest of program...\n')


#%% Original Script




# path_to_atlas = tide_dir
# path_new = path_to_atlas

# ### Run this when troubleshooting 
path_new = tide_dir
#path_new = r'/Volumes/PD/Original Data/TPXO9_atlas_v5_nc' #Aaron

#file = r'/Volumes/PD/GitHub/python-oceanography/Delft 3D FM Suite 2019/8.automated_tides/results/mac_example/example.pli' #Aaron

grid_ds = xr.open_dataset(glob.glob(path_new + r'/grid*.nc')[0]) # lazy loading of dataset to view metadata
print('Printing Grid metadata')
print(grid_ds) # print metadata for nc file dataset

#%% load in quick testing constituent (tidal height hz)
lat_y = np.array(grid_ds['lat_z'][::50])
lon_x = np.array(grid_ds['lon_z'][::50])
hz = np.array( np.fliplr(np.rot90((grid_ds['hz'][::50,::50]),axes =(1,0))) )
plt.figure(0)
#plt.pcolor(lon_x, lat_y, hz) #old map with uk at edges
# example of splitting grid and reforming it to be correct
  # length of lon/2
neg_lon = (360 - lon_x[int(len(lon_x)/2):(len(lon_x))] ) *-1
pos_lon = lon_x[0 : int(len(lon_x)/2)]
new_lon = np.concatenate((neg_lon,pos_lon), axis=0)

neg_hz = hz[: , int(len(lon_x)/2):(len(lon_x))]
pos_hz = hz[: , 0 : int(len(lon_x)/2)]
new_hz = np.concatenate((neg_hz,pos_hz), axis=1)
plt.figure(1)
plt.pcolor(new_lon, lat_y, new_hz) #new map with uk in the centre

#%% Load Data into the UK Boundary 
lat_y = np.array(grid_ds['lat_z'])
lon_x = np.array(grid_ds['lon_z'])
hz = np.array( np.fliplr(np.rot90((grid_ds['hz']),axes =(1,0))) )


neg_lon = (360- lon_x[int(len(lon_x)/2):(len(lon_x))] ) *-1
pos_lon = lon_x[0 : int(len(lon_x)/2)]
new_lon = np.concatenate((neg_lon,pos_lon), axis=0)
new_lon[5399] = 0 # some reason it thinks its not 0

neg_hz = hz[: , int(len(lon_x)/2):(len(lon_x))]
pos_hz = hz[: , 0 : int(len(lon_x)/2)]
new_hz = np.concatenate((neg_hz,pos_hz), axis=1)

new_lat = np.array(grid_ds['lat_z'])

#uk lat and lon corners
names = ['location','imin','imax','jmin','jmax']
nm = [0,2,3]
nm_word = ['Lon','Lat','Name']
uk = [-11,2,49,61]
#locs_df = pd.read_csv(start_path + r'modelling_DATA/kent_estuary_project/tidal_boundary/outer_bounds_points_of_grid')
#locs_df.set_index('Location')

#k = 0 # UK row
imin = find_nearest(new_lon,uk[0])
imax = find_nearest(new_lon,uk[1])
jmin = find_nearest(lat_y,uk[2])
jmax = find_nearest(lat_y,uk[3])

mx = int(np.array(np.where(new_lon == imin))) #min x
MX = int(np.array(np.where(new_lon == imax))) #max x
my = int(np.array(np.where(lat_y == jmin))) #min y
MY = int(np.array(np.where(lat_y == jmax))) #max y

UK_hz = new_hz[my:MY,mx:MX] # slice the UK
UK_lon = new_lon[mx:MX]
UK_lat = new_lat[my:MY]

plt.pcolor(UK_lon, UK_lat, UK_hz, cmap = 'jet')


#%% Generate location data
nm = [0,2,3]
nm_word = ['Lon','Lat','Name']
locs_of_boundary = pd.read_csv(filename, delimiter=(' '),header=1, usecols=(nm),names=(nm_word))
#locs_of_boundary.at[0,'Lat'] = ( locs_of_boundary['Lat'][0]+ 0.03)#use if boundary point is not in ocean for morecombe bay
locs_of_boundary.at[0,'Lat'] = ( locs_of_boundary['Lat'][0])
loc_x = locs_of_boundary.Lon
loc_y = locs_of_boundary.Lat
lent = (len(locs_of_boundary['Name']))

HX = []
for i in range(lent):
    fin = find_nearest(UK_lon,loc_x[i])
    HX.append(int(np.array(np.where(UK_lon == fin))))
HY = []
for i in range(lent):
    fin = find_nearest(UK_lat,loc_y[i])
    HY.append(int(np.array(np.where(UK_lat == fin))))

version = len(glob.glob(sub_dir + '/*'))

start_s = str(start).split(' ')[0]
end_e = str(end).split(' ')[0]

new_folder = sub_dir + '/v' + str(version+1).zfill(3) + '_' + locs_of_boundary.Name[0] + '_' + start_s + 'to' + end_e + '_' + '_'.join(consts)
os.mkdir(new_folder)
new_folder2 = sub_dir2 + '/v' + str(version+1).zfill(3) + '_' + locs_of_boundary.Name[0] + '_' + start_s + 'to' + end_e + '_' + '_'.join(consts)
os.mkdir(new_folder2)


#%% Load rest of nc Data
new_path = start_path + r'/modelling_DATA/kent_estuary_project/tidal_boundary/TPXO9_atlas_v5_nc'
h_path = []
for filename in glob.glob(new_path + r'/h*'):
    h_path.append(filename)
g_path = []
for filename in glob.glob(new_path + r'/grid*'):
    g_path.append(filename)
#code to extract names of tidal constituents    
# Muted section assists in generation of tidal constituents variable name
constants = []
for i in range(len(h_path)):
    sample = h_path[i].split('_')[-5]
    #constants.append('v_' + sample[1] + ',') 
    const_match = format_string(sample)
    constants.append(const_match)
# new_const= "" 
# for i in constants:
#     new_const += str(i)
    
v_2n2 = xr.open_dataset(h_path[0])
v_k1 = xr.open_dataset(h_path[1])
v_k2 = xr.open_dataset(h_path[2])
v_m2 = xr.open_dataset(h_path[3])
v_m4 = xr.open_dataset(h_path[4])
v_mf = xr.open_dataset(h_path[4])
v_mm = xr.open_dataset(h_path[6])
v_mn4 = xr.open_dataset(h_path[7])
v_ms4 = xr.open_dataset(h_path[8])
v_n2 = xr.open_dataset(h_path[9])
v_o1 = xr.open_dataset(h_path[10])
v_p1 = xr.open_dataset(h_path[11])
v_q1 = xr.open_dataset(h_path[12])
v_s1 = xr.open_dataset(h_path[13])
v_s2 = xr.open_dataset(h_path[14])

v_consts = v_2n2,v_k1,v_k2,v_m2,v_m4,v_mf,v_mm,v_mn4,v_ms4,v_n2,v_o1,v_p1,v_q1,v_s1,v_s2

#generate frequency table
freq_df = pd.read_csv('tide_cons/tidal_conts_&_freqs.csv')
dataframe_list = []
for i in constants:
    dataframe_list.append(np.array(freq_df.loc[freq_df['Names'].str.contains(i, case=False)]))
df_freq = pd.DataFrame(np.concatenate(dataframe_list),columns = ["Name", "freq", "Name_Lower"])  
df_freq['print_name'] = df_freq['Name'].str.replace(" ","")
df_amp_phase = pd.DataFrame() # make empty dataframe
#Generation of massive loop to run all constituents. 
for i in range(len(v_consts)):
    real = v_consts[i]['hRe']
    imag = v_consts[i]['hIm']
    #real
    real_shift = np.array( np.fliplr(np.rot90(real,axes =(1,0))))
    r_Neg = real_shift[: , int(len(lon_x)/2):(len(lon_x))] # negative lon
    r_Pos = real_shift[:, 0 : int(len(lon_x)/2)] # positive lon
    r_comb = np.concatenate((r_Neg,r_Pos), axis=1) # combine back together
    r_location_data = r_comb[my:MY,mx:MX] # select the uk
    #imag
    imag_shift = np.array( np.fliplr(np.rot90(imag,axes =(1,0))))
    i_Neg = imag_shift[: , int(len(lon_x)/2):(len(lon_x))] # negative lon
    i_Pos = imag_shift[:, 0 : int(len(lon_x)/2)] # positive lon
    i_comb = np.concatenate((i_Neg,i_Pos), axis=1) # combine back together
    i_location_data = i_comb[my:MY,mx:MX] # select the uk
    
    
#5%% Generation of loop to find location
    print_name = df_freq['print_name'][i]
    frequency = str(df_freq['freq'][i])
    for j in range (lent):
        amp = (abs( float((r_location_data[HY[j] , HX[j]]))+1j*float((i_location_data[HY[j] , HX[j]])) )/1000)
        pha = (math.atan2(-(i_location_data[HY[j] , HX[j]]),r_location_data[HY[j] , HX[j]])/np.pi*180)
        
        csv_filename = r'/point' + r'_' + str(j + 1) + r'.csv'
        f = open(new_folder + '/' + csv_filename, 'a')
        if i == 0:
            f.write("TC,Amplitude_(m),Phase_(Deg),Freq_(deg_hour)" + '\n')    
        f.write(print_name + ',' + str(amp) + "," + str(pha) + "," + frequency + '\n')
        f.close()
    
print('nPrinting Harmonics Table')
print(df_freq)    
#%% Plotting    
# fig5, ax = plt.subplots()
# fig5.set_figheight(15)
# fig5.set_figwidth(15)
# plt.pcolor(UK_lon, UK_lat, r_location_data)
# plt.scatter(loc_x,loc_y,c = 'r')
# for i in range(lent):
#     plt.text(x = loc_x[i],y = loc_y[i], s = str([i]), fontdict=dict(color='red', alpha=1, size=15))
   
    
# fig6, ax = plt.subplots()
# fig6.set_figheight(15)
# fig6.set_figwidth(15)
# plt.pcolor(UK_lon, UK_lat, r_location_data)
# plt.scatter(loc_x,loc_y,c = 'r')
# for i in range(lent):
#     plt.text(x = loc_x[i],y = loc_y[i], s = str([i]), fontdict=dict(color='red', alpha=1, size=15))
# ax.set_xlim(-4.25,-3.25)
# ax.set_ylim(53.25,54.75)

#%% Generating the tidal heights files and making it straight into a boundary condition file. 

#%% Generate naming system and extract values

start = datetime.strptime(start, '%Y-%m-%d %H:%M')
end = datetime.strptime(end, '%Y-%m-%d %H:%M')
total_minutes = int((end - start).total_seconds() / 60)
timestep_minutes = int(timestep)
t = [start + timedelta(minutes=i) for i in range(0, total_minutes + 1, timestep_minutes)]


#start = dtime.datetime(2013, 1, 1, 0, 0, 0)

#t = [start + timedelta(hours = i) for i in range(32*24)]



#making blank adjustable tide maker
con = np.char.ljust(np.char.upper(consts), 4)
cons = con.reshape(-1, 1) # this reshapes them according to original formula 
#%%
amp = []
pha = []
fre = []
consts_names_rearanged = []

for i,file in enumerate(sorted(glob.glob(new_folder + r'/*.csv'))):
    df = pd.read_csv(file)
    print(i)
    amplitudes = []
    phases = []
    frequencies = []
    for index, row in df.iterrows():
        name = format_string(row['TC'])

        if name in cons:
            print(name)
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
print(consts_names_rearanged)

con2 = np.char.ljust(np.char.upper(consts_names_rearanged), 4)
cons2 = con2.reshape(-1, 1)

for i,file in enumerate(sorted(glob.glob(new_folder + r'/*.csv'))):
    freq = fre[i]
    FREQ = np.array(freq)
    
    tidecons = []
    new_amp = amp[i]
    new_pha =pha[i]
    for j,file in enumerate(cons2):
        tidecons.append(([new_amp[j],0, new_pha[j],0]))
        
    tidecons2 = np.array(tidecons).astype(float)
    eta = tt.t_predic(np.array(t), cons2, FREQ, tidecons2)
    
    fig, ax = plt.subplots(figsize = (7,3))
    ax.plot(t, eta)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.xticks(rotation=45)
    plt.xlim([t[0], t[0] + timedelta(days=3)])
    plt.xticks()
    plt.tight_layout()
    plt.savefig(new_folder2 + '/3_days.png', dpi = 200)
    
    fig2, ax2 = plt.subplots(figsize = (7,3))
    ax2.plot(t, eta)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    #plt.xlim([t[0], t[25]])
    plt.xticks()
    plt.tight_layout()
    plt.savefig(new_folder2 + '/0all_days.png', dpi = 200)

    df2 = pd.DataFrame(eta,t)

    
    csv_filename = r'/point' + r'_' + str(i+1) + r'.csv'
    to_write = new_folder2 + '/' + csv_filename
    df2.to_csv(to_write, header = False)

#%%MAKE THE BOUNDARY FILE
bc_file = main_folder_path + '/WaterLevel.bc'
with open(bc_file, "w") as f:
    f.write("")
    f.close()
#Load data and write
seconds_since = str(start)

for i,file in enumerate(sorted(glob.glob(new_folder2+'/*.csv'))):
    #print(file)
    surface_height = pd.read_csv(file, names = ['time', 'surface_height'])
    write_text_block_to_file_H(bc_file,locs_of_boundary.Name[i], str(t[0]))
    data_to_write = surface_height.surface_height
    new_time = [i[:-3] for i in surface_height.time] #reformat data, remove seconds off the end
    converted_timeseries = [int(i) for i in (convert_to_seconds_since_date(new_time, seconds_since))] 

    with open(bc_file, "a") as f:
        for j,file in enumerate(data_to_write):
            #print(file)
            f.write(str(converted_timeseries[j]) + '    ' + str(file) + '\n')             
        f.write('\n')


    
