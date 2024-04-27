# um_emc2_toolkit.py

# This file contains all the functions required to process and save UM model output in preparation for use with EMC2.
# It also contains the UM model class for bridging processed UM model output with EMC2 functionality.

# Load required packages
import os # for interacting with the operating system
import re # for regular expression pattern matching
import glob # for pathname pattern matching
import xarray as xr # for general dataset file handling
import pandas as pd # for additional data handling
import numpy as np # for scientific computing
import netCDF4 as nc # for NetCDF file handling
import iris # for .pp file handling
import emc2 # for creating simulated data
from emc2.core import Model # for importing model class presets
from emc2.core import Instrument # for importing instrument class presets
from emc2.core.instruments import HSRL # for using the HSRL (lidar) instrument preset - specific to the MARCUS example
from emc2.core.instruments import WACR # for using the WACR (radar) instrument preset - specific to the MARCUS example
from emc2.core.instrument import ureg # for using units compatible with emc2
from datetime import datetime, timedelta # for manipulating dates and times
import matplotlib.pyplot as plt # for providing a primary, MATLAB-like plotting interface
import matplotlib as mpl # for providing core functionality to create figures and axes in Matplotlib plots
import matplotlib.dates as mdates # for providing functionality to work with dates and times in Matplotlib plots
from matplotlib.colors import LogNorm # for providing funtionality to work with color scales in Matplotlib plots

# um_emc2_main function
##############################
# The purpose of this function is to process and re-save UM regional model output so that it is ready to use with the Earth Model Column Collaboratory (EMC2) instrument simulator.
# This function processes one day of data at daily or higher resolution. To process multiple days, it is recommended to construct a wrapper for this function.
# This function ultimately saves a single .nc file to the output path at native model time and height resolution and with one horizontal grid cell of data. The horizontal grid cell may be either a fixed point or a moving point.
# This function calls um_emc2_input_validator, um_emc2_dictionary_sorter, um_emc2_data_processor, and um_emc2_final_saver to check, process, and re-save the data.
##############################
# Inputs:
# a date for a single day as either a string ('YYYYMMDD') or datetime object;
# a parent folder containing all data to read (preferably in .nc format, though .pp files can also be processed);
# an output path; and
# an xarray dataset containing latitude and longitude variables as a function of a datetime coordinate.
##############################

# Define the um_emc2_main function
def um_emc2_main(date, parent_folder, output_folder, coordinates):
    
    # Validate the input, including the date, parent and output folders, coordinates, and types of files to process, and return the lists of unsorted .nc and .pp files to process
    date, date_str, coordinates, files_list_nc, files_list_pp = um_emc2_input_validator(date = date, parent_folder = parent_folder, output_folder = output_folder, coordinates = coordinates)
    print('um_emc2_input_validator executed successfully')
    print()
    
    # Pass file lists to the sorting function and return the sorted dictionary
    filenames_dict = um_emc2_dictionary_sorter(files_list_nc = files_list_nc, files_list_pp = files_list_pp)
    print('um_emc2_dictionary_sorter executed successfully')
    print()

    # Process the data - extract only relevant variables, limit time and spatial domains, and save partial files in the output folder
    um_emc2_data_processor(date = date, date_str = date_str, parent_folder = parent_folder, output_folder = output_folder, filenames_dict = filenames_dict, coordinates = coordinates)
    print('um_emc2_data_processor executed successfully')
    print()

    # Save the final data file for the selected day
    um_emc2_final_saver(date_str, output_folder)
    print('um_emc2_final_saver executed successfully')
    print()

    print('um_emc2_main executed successfully')

# um_emc2_input_validator function
##############################
# The purpose of this function is check whether the inputs to um_emc2_main are valid and to list all valid files to process.
# This function returns the date as a datetime object as well as a date_str, a coordinates dataset modified to be compatible with following functions, and lists of valid .nc and .pp files.
##############################
# Inputs:
# a date for a single day as either a string ('YYYYMMDD') or datetime object;
# a parent folder containing all data to read (preferably in .nc format, though .pp files can also be processed);
# an output path; and
# an xarray dataset containing latitude and longitude variables as a function of a datetime coordinate.
##############################

# Define the um_emc2_input_validator function
def um_emc2_input_validator(date, parent_folder, output_folder, coordinates):

    # Date check
    # If the date is a string, check whether the format is valid and convert it to datetime object
    if isinstance(date, str):
        try:
            date = datetime.strptime(date, '%Y%m%d')
        except ValueError:
            raise ValueError('Invalid date format. Please provide a date in "YYYYMMDD" format')
        date_str = date.strftime('%Y%m%d')
    # If the date is a datetime, create a string version
    elif isinstance(date, datetime):
        date_str = date.strftime('%Y%m%d')
    else:
        raise ValueError('Input date must be a string or a datetime object')
    
    # Parent folder check
    if not os.path.isdir(parent_folder):
        raise ValueError(f'parent_folder "{parent_folder}" is not a valid directory.')
    
    # Output path check
    if not os.path.isdir(output_folder):
        raise ValueError(f'output_folder "{output_folder}" is not a valid directory.')
    
    # Coordinates check
    # Check for latitude and longitude variables
    # Find variables with appropriate strings
    lat_var = [var for var in coordinates.data_vars if 'lat' in var]
    lon_var = [var for var in coordinates.data_vars if 'lon' in var]
    # Raise an exception if a latitude or longitude variable is not found or if more than one such variable is found
    if not lat_var or not lon_var:
        raise ValueError('Latitude or longitude variable not found')
    if len(lat_var) > 1 or len(lon_var) > 1:
        raise ValueError('More than one latitude or longitude variable found')
    # Rename the latitude and longitude variables
    coordinates = coordinates.rename({lat_var[0]: 'latitude', lon_var[0]: 'longitude'})
    # Check for datetime variable
    # Find variable of type datetime
    datetime_var = [var for var in coordinates.coords if coordinates[var].dtype == 'datetime64[ns]']
    # Raise an exception if no datetime variable is found or if more than one such variable is found
    if not datetime_var:
        raise ValueError('Datetime type variable not found')
    if len(datetime_var) > 1:
        raise ValueError('More than one datetime variable found')
    # Rename datetime variable to 'coordinate_datetime'
    coordinates = coordinates.rename({datetime_var[0]: 'coordinate_datetime'})
    
    # File type check
    # Create the list of all files in the parent folder
    # Check what types of files need to be processed
    files_list_nc = []
    files_list_pp = []
    for root, dirs, filenames in os.walk(parent_folder):
        for filename in filenames:
            if filename.endswith('.nc'):
                files_list_nc.append(os.path.join(root, filename))
            elif not os.path.splitext(filename)[1]:
                files_list_pp.append(os.path.join(root, filename))
            else:
                print(f'{filename} is an unrecognized filetype and will not be processed')
    # Print numbers and types of files found
    if files_list_nc:
        print(f'{len(files_list_nc)} .nc files found')
    if files_list_pp:
        print(f'{len(files_list_pp)} .pp files found')
    if not files_list_nc and not files_list_pp:
        raise ValueError('No files of either .nc or other type found')
    if files_list_nc and files_list_pp:
        print('Warning: Multiple file types detected; code will proceeed, but we recommend ensuring uniformity of file types for simplicity of error diagnosis')

    print()
    return(date, date_str, coordinates, files_list_nc, files_list_pp)

# um_emc2_dictionary_sorter function
##############################
# The purpose of this function is to process the lists of .nc and/or .pp files generated by um_emc2_input_validator.
# This function returns filenames_dict, a single dictionary of all .nc and .pp filenames sorted into numeric chunks. This is to handle cases where UM model output contains multiple parts per day.
##############################
# Inputs:
# a list of .nc files and
# a list of .pp files.
##############################

# Define the um_emc2_dictionary_sorter function
def um_emc2_dictionary_sorter(files_list_nc, files_list_pp):

    # Create the dictionary of files sorted by numerical code (presumably time) in the base filename
    # Create a unified files list
    files_list_all = files_list_nc + files_list_pp
    filenames_list = [os.path.basename(filename) for filename in files_list_all]
    # Initialize the dictionary
    filenames_dict = {}
    for filename in filenames_list:
        # If the file type .nc, strip the extension first
        if filename.endswith('.nc'):
            filename_base = os.path.splitext(filename)[0]
        # If the file type is absent, the file is presumed .pp type
        else:
            filename_base = filename
        # If the numeric block is already a key in the dictionary, append the filename including the extension
        # Otherwise, create a new list with the filename as the value for that key
        numeric_blocks = re.findall(r'\d+', filename_base)
        for numeric_block in numeric_blocks:
            if numeric_block in filenames_dict:
                filenames_dict[numeric_block].append(filename)
            else:
                filenames_dict[numeric_block] = [filename]
    # Sort filenames within each numeric block
    for numeric_block, filenames in filenames_dict.items():
        filenames_dict[numeric_block] = sorted(filenames)
    # Sort blocks/keys by number
    filenames_dict = dict(sorted(filenames_dict.items(), key=lambda item: int(item[0])))
    # Print the keys and files
    for key, items in filenames_dict.items():
        print(key)
        for item in items:
            print(item)

    print()
    return(filenames_dict)

# um_emc2_data_processor function
##############################
# The purpose of this function is to process the files listed in filenames_dict, extracting the variables required by EMC2, retaining only data within the current date, and subsetting the data to the coordinates provided.
# This function saves a number of intermediate files equal to the number of chunks the day is divided into (defined by keys in filenames_dict).
##############################
# Inputs:
# a date for a single day as a datetime object;
# a date for a single day as a string ('YYYYMMDD');
# a parent folder containing all data to read (preferably in .nc format, though .pp files can also be processed);
# an output path;
# the dictionary of filenames returned by the um_emc2_dictionary_sorter function; and
# an xarray dataset containing latitude and longitude variables as a function of a datetime coordinate.
##############################

# Define the um_emc2_data_processor function
def um_emc2_data_processor(date, date_str, parent_folder, output_folder, filenames_dict, coordinates):

    # Create the list of variables to look for
    required_variables = ['mass_fraction_of_cloud_liquid_water_in_air','mass_fraction_of_cloud_ice_in_air','mass_fraction_of_rain_in_air','mass_fraction_of_cloud_ice_crystals_in_air','mass_fraction_of_graupel_in_air',
    'number_of_cloud_droplets_per_kg_of_air','number_of_ice_particles_per_kg_of_air','number_of_rain_drops_per_kg_of_air','number_of_snow_aggregates_per_kg_of_air','number_of_graupel_particles_per_kg_of_air',
    'cloud_area_fraction_in_atmosphere_layer','air_pressure','air_temperature','specific_humidity']
    required_coordinates = ['level_height','time','model_level_number','grid_latitude','grid_longitude']
    variables_to_keep = required_variables + required_coordinates
    # Loop across all file groups defined by dictionary keys
    for key in list(filenames_dict.keys()):
        # Create the list of current_files
        current_items = filenames_dict[key]
        # Identify the current key
        print(f'Now searching {key} files...')
        # Reset the key_dataset
        key_dataset = xr.Dataset()
        # Reset the breaker
        break_current = False
        # Reset the list of variables found
        variables_found = []
        # Loop across all current files
        for file in current_items:
            # Check the file type and handle accordingly
            # If the filetype is .nc...
            if file.endswith('.nc'):
                # Load the current dataset
                dataset = xr.open_dataset(f'{parent_folder}/{file}')
                print(f'Now searching {file}...')
                # Check whether the data contains a time coordinate
                if 'time' not in dataset:
                    print('No time coordinate found - skipping current file...')
                    continue
                # Remove the time component from the date
                date_only = np.datetime64(date, 'D')
                # Select time values within the current date
                time_within_date = dataset.time.where((dataset.time.dt.floor('D') == date_only), drop=True)
                # Check whether dataset uses time as an index
                try:
                    dataset = dataset.sel(time=time_within_date)
                except KeyError:
                    print('No index found for coordinate time - skipping current file...')
                    continue
                # Check whether any data remains
                if len(dataset.time) == 0:
                    print(f'{file} in {key} block has no data inside current date - skipping current key...')
                    break_current = True
                    break
                # Check whether the dataset contains any of the required variables
                if any(var in dataset.variables for var in required_variables):
                    # Drop variables not in required_variables or required_coordinates
                    dataset = dataset.drop_vars([var for var in dataset.variables if var not in variables_to_keep])
                    # Loop across the variables in the dataset
                    for var in dataset:
                        # Is the variable required?
                        if var in required_variables:
                            # Has the variable already been found in the current key?
                            if var in variables_found:
                                # Drop the variable from the dataset
                                dataset = dataset.drop_vars(var)
                            else:
                                print(f'{var} found in {file}')
                                # Add the variable to the list of found variables
                                variables_found += [var]
                    # Check whether the dataset still contains any of the required variables
                    if any(var in dataset.variables for var in required_variables):
                        # Subset the dataset to the coordinates location
                        # Create an empty list to store the dataset subsets
                        subset_dataset_list = []
                        # Create empty lists to store latitudes and longitudes
                        lat_values, lon_values = [], []
                        # Iterate over each time value in the dataset
                        for time_value in dataset['time']:
                            # Create a dataset containing only values at 'time_value'
                            dataset_current_time = dataset.sel(time=time_value)
                            # Find the closest coordinate_datetime in the coordinates
                            closest_coordinate_datetime = coordinates['coordinate_datetime'].sel(coordinate_datetime = time_value, method = 'nearest').values
                            # Find the corresponding latitude and longitude values
                            lat_value = coordinates['latitude'].sel(coordinate_datetime=closest_coordinate_datetime).values
                            lon_value = coordinates['longitude'].sel(coordinate_datetime=closest_coordinate_datetime).values
                            # Find the grid_latitude and grid_longitude values in the dataset closest to lat_value and lon_value
                            closest_lat_idx = np.argmin(np.abs(dataset_current_time['grid_latitude'].values - lat_value))
                            closest_lon_idx = np.argmin(np.abs(dataset_current_time['grid_longitude'].values - lon_value))
                            # Subset the dataset based on the closest grid_latitude and grid_longitude values
                            subset = dataset_current_time.isel(grid_latitude=closest_lat_idx, grid_longitude=closest_lon_idx)
                            # Save the current grid_latitude and grid_longitude values
                            lat_values.append(lat_value)
                            lon_values.append(lon_value)
                            # lat_values.append(subset['grid_latitude'].values)
                            # lon_values.append(subset['grid_longitude'].values)
                            # Drop latitude and longitude from the subset
                            subset = subset.drop_vars('grid_latitude')
                            subset = subset.drop_vars('grid_longitude')
                            # Append the subset to the list
                            subset_dataset_list.append(subset)
                        # Concatenate the subsets along a new dimension
                        subset_dataset = xr.concat(subset_dataset_list, dim='time')
                        # Add latitudes and longitudes as variables
                        lat_values = xr.DataArray(data=lat_values, dims='dim_name')
                        lon_values = xr.DataArray(data=lon_values, dims='dim_name')
                        subset_dataset['latitude'] = ('time', lat_values.data)
                        subset_dataset['longitude'] = ('time', lon_values.data)
                        # Merge the subset_dataset with the key_dataset
                        key_dataset = key_dataset.merge(subset_dataset)
                        print(f'File {file} merged')
                    else:
                        # No variables found?
                        print(f'{file} contains no required variables')
                else:
                    # No variables found?
                    print(f'{file} contains no required variables')
            # If the filetype is not .nc...
            else:
                # Load the current dataset using iris
                iris_cubes = iris.load(f'{parent_folder}/{file}')
                # Initialize an empty dataset
                dataset = xr.Dataset()
                print(f'Now searching {file}...')
                # Iterate over the variables in iris_cubes to create the dataset
                for i, f in enumerate(iris_cubes):
                    # Create the temporary dataset
                    temp_dataset = xr.DataArray.from_iris(iris_cubes[i])                
                    # Check whether the temp_dataset current variable is required
                    if temp_dataset.name in required_variables:
                        # Check whether the variable has already been found
                        if temp_dataset.name not in variables_found:
                            try:
                                dataset = dataset.merge(temp_dataset)
                            except xr.MergeError as e:
                                if 'conflicting values for variable \'level_height\'' in str(e):
                                    print(f'Warning: Conflicing values detected for variable "level_height" in {temp_dataset.name}; compat="override" used')
                                    dataset = dataset.merge(temp_dataset, compat='override')
                                elif 'conflicting values for variable \'forecast_period\'' in str(e):
                                    print(f'Warning: Conflicing values detected for variable "forecast_period" in {temp_dataset.name}; compat="override" used')
                                    dataset = dataset.merge(temp_dataset, compat='override')
                                elif 'conflicting values for variable \'forecast_reference_time\'' in str(e):
                                    print(f'Warning: Conflicing values detected for variable "forecast_reference_time" in {temp_dataset.name}; compat="override" used')
                                    dataset = dataset.merge(temp_dataset, compat='override')
                                elif 'conflicting values for variable \'height\'' in str(e):
                                    print(f'Warning: Conflicing values detected for variable "height" in {temp_dataset.name}; compat="override" used')
                                    dataset = dataset.merge(temp_dataset, compat='override')
                                elif 'conflicting values for variable \'level_height\'' in str(e):
                                    print(f'Warning: Conflicing values detected for variable "level_height" in {temp_dataset.name}; compat="override" used')
                                    dataset = dataset.merge(temp_dataset, compat='override')
                                else:
                                    raise e
                            except AttributeError as e:
                                print(f"Skipping variable {temp_dataset.name}: {e}...")
                                continue  # Skip this iteration and continue with the next one
                            except ValueError as e:
                                if "Unpacking PP fields with LBPACK of 1 requires mo_pack to be installed" in str(e):
                                    print("Warning: mo_pack is not installed; skipping the current loop iteration...")
                                    continue  # Skip the current iteration and move to the next one
                                else:
                                    raise # Re-raise the error if it's not the expected one
                        print(f'{temp_dataset.name} found in {file}')
                        # Add the variable to the list of found variables
                        variables_found += [temp_dataset.name]
                        # Identify the variable found
                # Check whether the data contains a time coordinate
                if 'time' not in dataset.variables:
                    print('No time coordinate found - skipping current file...')
                    continue
                # Remove the time component from the date
                date_only = np.datetime64(date, 'D')
                # Select time values within the current day
                time_within_date = dataset.time.where((dataset.time.dt.floor('D') == date_only), drop=True)
                # Check whether dataset uses time as an index
                try:
                    dataset = dataset.sel(time=time_within_date)
                except KeyError:
                    print('No index found for coordinate time - skipping current file...')
                    continue
                # Check whether any data remains
                if len(dataset.time) == 0:
                    print(f'{file} in {key} block has no data inside current date - skipping current key...')
                    break_current = True
                    break
                # Drop variables not in required_variables or required_coordinates
                dataset = dataset.drop_vars([var for var in dataset.variables if var not in variables_to_keep])
                # Check whether the dataset still contains any of the required variables
                if any(var in dataset.variables for var in required_variables):
                    # Subset the dataset to the coordinates location
                    # Create an empty list to store the dataset subsets
                    subset_dataset_list = []
                    # Create empty lists to store latitudes and longitudes
                    lat_values, lon_values = [], []
                    # Iterate over each time value in the dataset
                    for time_value in dataset['time']:
                        # Create a dataset containing only values at 'time_value'
                        dataset_current_time = dataset.sel(time=time_value)
                        # Find the closest coordinate_datetime in the coordinates
                        closest_coordinate_datetime = coordinates['coordinate_datetime'].sel(coordinate_datetime = time_value, method = 'nearest').values
                        # Find the corresponding latitude and longitude values
                        lat_value = coordinates['latitude'].sel(coordinate_datetime=closest_coordinate_datetime).values
                        lon_value = coordinates['longitude'].sel(coordinate_datetime=closest_coordinate_datetime).values
                        # Find the grid_latitude and grid_longitude values in the dataset closest to lat_value and lon_value
                        closest_lat_idx = np.argmin(np.abs(dataset_current_time['grid_latitude'].values - lat_value))
                        closest_lon_idx = np.argmin(np.abs(dataset_current_time['grid_longitude'].values - lon_value))
                        # Subset the dataset based on the closest grid_latitude and grid_longitude values
                        subset = dataset_current_time.isel(grid_latitude=closest_lat_idx, grid_longitude=closest_lon_idx)
                        # Save the current grid_latitude and grid_longitude values
                        lat_values.append(lat_value)
                        lon_values.append(lon_value)
                        # lat_values.append(subset['grid_latitude'].values)
                        # lon_values.append(subset['grid_longitude'].values)
                        # Drop latitude and longitude from the subset
                        subset = subset.drop_vars('grid_latitude')
                        subset = subset.drop_vars('grid_longitude')
                        # Append the subset to the list
                        subset_dataset_list.append(subset)
                    # Concatenate the subsets along a new dimension
                    subset_dataset = xr.concat(subset_dataset_list, dim='time')
                    # Add latitudes and longitudes as variables
                    lat_values = xr.DataArray(data=lat_values, dims='dim_name')
                    lon_values = xr.DataArray(data=lon_values, dims='dim_name')
                    subset_dataset['latitude'] = ('time', lat_values.data)
                    subset_dataset['longitude'] = ('time', lon_values.data)
                    # Merge the subset_dataset with the key_dataset
                    key_dataset = key_dataset.merge(subset_dataset)
                    print(f'File {file} merged')           
                else:
                    # No variables found?
                    print(f'{file} contains no required variables')
        # If there are no data in the current date, skip the current key
        if not break_current:
            # Set the save path and save the data
            print(f'Saving the {key} data...')
            save_path = os.path.join(output_folder, f'um_emc2_chunk_{date_str}_{key}.nc')
            key_dataset.to_netcdf(save_path)
            print(f'File saved: {save_path}')
    
        print()

# um_emc2_final_saver function
##############################
# The purpose of this function is to process the intermediate files produced by the um_emc2_data_processor function, making final changes such as unit and file attribute inclusions before deleting each intermediate file.
# This function saves the final UM model data file, processed and ready for use with EMC2.
##############################
# Inputs:
# a date for a single day as a string ('YYYYMMDD');
# an output path where the intermediate files have been saved and where the final file will also be saved;
##############################

# Define the um_emc2_final_saver function
def um_emc2_final_saver(date_str, output_folder):

    # Define the pattern of partial files to search for
    pattern = os.path.join(output_folder, f'*{date_str}*')
    # List all partial files for the current date
    saved_partial_files = glob.glob(pattern)
    # Initialize the final dataset
    final_dataset = xr.Dataset()
    # Load each partial file and merge into the final dataset
    for file in saved_partial_files:
        temp_dataset = xr.open_dataset(file)
        final_dataset = final_dataset.merge(temp_dataset)
    
    # Create a variable filled with zeros and add to the dataset
    zeros_data = xr.DataArray(
        data = np.zeros((len(final_dataset['time']), len(final_dataset['level_height']))),
        dims = ('time', 'model_level_number'),
        coords = {'time': final_dataset['time'], 'level_height': final_dataset['level_height']}
    )
    final_dataset['zeros_var'] = zeros_data
    
    # Add a 2-dimensional z_values variable to the dataset
    level_height = final_dataset['level_height'].values
    height_data = level_height[:, np.newaxis] * np.ones(len(final_dataset['time']))
    final_dataset['height_var'] = (('level_height', 'time'), height_data)
    final_dataset['height_var'].attrs['units'] = 'meter'
    final_dataset['height_var'] = final_dataset['height_var'].transpose('time', 'level_height')
    
    # Add units to all fields
    # Mass mixing ratio
    final_dataset['mass_fraction_of_cloud_liquid_water_in_air'] = final_dataset['mass_fraction_of_cloud_liquid_water_in_air'].assign_attrs(units=ureg.kg/ureg.kg)
    final_dataset['mass_fraction_of_cloud_liquid_water_in_air'].attrs['units'] = str(final_dataset['mass_fraction_of_cloud_liquid_water_in_air'].attrs['units'])
    final_dataset['mass_fraction_of_cloud_ice_crystals_in_air'] = final_dataset['mass_fraction_of_cloud_ice_crystals_in_air'].assign_attrs(units=ureg.kg/ureg.kg)
    final_dataset['mass_fraction_of_cloud_ice_crystals_in_air'].attrs['units'] = str(final_dataset['mass_fraction_of_cloud_ice_crystals_in_air'].attrs['units'])
    final_dataset['mass_fraction_of_rain_in_air'] = final_dataset['mass_fraction_of_rain_in_air'].assign_attrs(units=ureg.kg/ureg.kg)
    final_dataset['mass_fraction_of_rain_in_air'].attrs['units'] = str(final_dataset['mass_fraction_of_rain_in_air'].attrs['units'])
    final_dataset['mass_fraction_of_cloud_ice_in_air'] = final_dataset['mass_fraction_of_cloud_ice_in_air'].assign_attrs(units=ureg.kg/ureg.kg)
    final_dataset['mass_fraction_of_cloud_ice_in_air'].attrs['units'] = str(final_dataset['mass_fraction_of_cloud_ice_in_air'].attrs['units'])
    final_dataset['mass_fraction_of_graupel_in_air'] = final_dataset['mass_fraction_of_graupel_in_air'].assign_attrs(units=ureg.kg/ureg.kg)
    final_dataset['mass_fraction_of_graupel_in_air'].attrs['units'] = str(final_dataset['mass_fraction_of_graupel_in_air'].attrs['units'])
    # Number concentration
    # Add number concentration per kilogram of air to final_dataset
    final_dataset['number_of_cloud_droplets_per_kg_of_air'] = final_dataset['number_of_cloud_droplets_per_kg_of_air']
    final_dataset['number_of_ice_particles_per_kg_of_air'] = final_dataset['number_of_ice_particles_per_kg_of_air']
    final_dataset['number_of_rain_drops_per_kg_of_air'] = final_dataset['number_of_rain_drops_per_kg_of_air']
    final_dataset['number_of_snow_aggregates_per_kg_of_air'] = final_dataset['number_of_snow_aggregates_per_kg_of_air']
    final_dataset['number_of_graupel_particles_per_kg_of_air'] = final_dataset['number_of_graupel_particles_per_kg_of_air']
    # Add units
    final_dataset['number_of_cloud_droplets_per_kg_of_air'] = final_dataset['number_of_cloud_droplets_per_kg_of_air'].assign_attrs(units=ureg.kg**(-1))
    final_dataset['number_of_cloud_droplets_per_kg_of_air'].attrs['units'] = str(final_dataset['number_of_cloud_droplets_per_kg_of_air'].attrs['units'])
    final_dataset['number_of_ice_particles_per_kg_of_air'] = final_dataset['number_of_ice_particles_per_kg_of_air'].assign_attrs(units=ureg.kg**(-1))
    final_dataset['number_of_ice_particles_per_kg_of_air'].attrs['units'] = str(final_dataset['number_of_ice_particles_per_kg_of_air'].attrs['units'])
    final_dataset['number_of_rain_drops_per_kg_of_air'] = final_dataset['number_of_rain_drops_per_kg_of_air'].assign_attrs(units=ureg.kg**(-1))
    final_dataset['number_of_rain_drops_per_kg_of_air'].attrs['units'] = str(final_dataset['number_of_rain_drops_per_kg_of_air'].attrs['units'])
    final_dataset['number_of_snow_aggregates_per_kg_of_air'] = final_dataset['number_of_snow_aggregates_per_kg_of_air'].assign_attrs(units=ureg.kg**(-1))
    final_dataset['number_of_snow_aggregates_per_kg_of_air'].attrs['units'] = str(final_dataset['number_of_snow_aggregates_per_kg_of_air'].attrs['units'])
    final_dataset['number_of_graupel_particles_per_kg_of_air'] = final_dataset['number_of_graupel_particles_per_kg_of_air'].assign_attrs(units=ureg.kg**(-1))
    final_dataset['number_of_graupel_particles_per_kg_of_air'].attrs['units'] = str(final_dataset['number_of_graupel_particles_per_kg_of_air'].attrs['units'])
    # Stratiform fraction
    final_dataset['cloud_area_fraction_in_atmosphere_layer'] = final_dataset['cloud_area_fraction_in_atmosphere_layer'].assign_attrs(units=ureg.meter/ureg.meter)
    final_dataset['cloud_area_fraction_in_atmosphere_layer'].attrs['units'] = str(final_dataset['cloud_area_fraction_in_atmosphere_layer'].attrs['units'])
    # Pressure
    final_dataset['air_pressure'] = final_dataset['air_pressure'].assign_attrs(units=ureg.pascal)
    final_dataset['air_pressure'].attrs['units'] = str(final_dataset['air_pressure'].attrs['units'])
    # Temperature
    final_dataset['air_temperature'] = final_dataset['air_temperature'].assign_attrs(units=ureg.kelvin)
    final_dataset['air_temperature'].attrs['units'] = str(final_dataset['air_temperature'].attrs['units'])
    # Specific humidity
    final_dataset['specific_humidity'] = final_dataset['specific_humidity']
    final_dataset['specific_humidity'] = final_dataset['specific_humidity'].assign_attrs(units=ureg.kg/ureg.kg)
    final_dataset['specific_humidity'].attrs['units'] = str(final_dataset['specific_humidity'].attrs['units'])
    
    # Change primary height dimension to use level_height values
    level_heights = final_dataset['level_height']
    final_dataset['model_level_number'] = level_heights
    # Remove/rename coordinates
    # final_dataset = final_dataset.drop_vars(['level_height', 'sigma', 'forecast_reference_time'])
    final_dataset = final_dataset.drop_vars(['level_height'])
    final_dataset = final_dataset.rename({'model_level_number': 'level_height'})
    
    # Add general attributes to the dataset
    # final_dataset.attrs['description'] = 'UM regional model data subsetted to the MARCUS RSV location'
    # final_dataset.attrs['authors'] = 'Calum L. Knight & Sonya L. Fiddes'
    final_dataset.attrs['creation_date'] = datetime.now().strftime('%Y-%m-%d')
    
    # Set the final filename
    final_filename = f'um_emc2_{date_str}.nc'
    # Save the final dataset
    final_dataset.to_netcdf(os.path.join(output_folder, final_filename))
    
    # Delete partial files
    for file in saved_partial_files:
        os.remove(file)

# UM model class
##############################
# The purpose of this class is to allow EMC2 to create simulated instrument data from UM model output.
# It is used as follows: UM_instance = UM(file_path), where file_path points to UM model data previously processed and saved by um_emc2_main.
##############################

class UM(Model):
    def __init__(self, file_path, time_range=None, load_processed=False, time_dim="time", appended_str=False, all_appended_in_lat=False):
        """
        This loads a UM simulation with all of the necessary parameters for EMC2 to run.
        Parameters
        ----------
        file_path: str
            Path to a UM simulation.
        time_range: tuple, list, or array, typically in datetime64 format
            Two-element array with starting and ending of time range.
        load_processed: bool
            If True, treating the 'file_path' variable as an EMC2-processed dataset; thus skipping
            appended string removal and dimension stacking, which are typically part of pre-processing.
        time_dim: str
            Name of the time dimension. Typically "time" or "ncol".
        appended_str: bool
            If True, removing appended strings added to fieldnames and coordinates during
            post-processing (e.g., in cropped regions from global simualtions).
        all_appended_in_lat: bool
            If True using only the appended str portion to the lat_dim. Otherwise, combining
            the appended str from both the lat and lon dims (relevant if appended_str is True).
        """
        super().__init__()
        self.Rho_hyd = {'cl': 1000. * ureg.kg / (ureg.m**3),
                        'ci': 500. * ureg.kg / (ureg.m**3),
                        'pl': 1000. * ureg.kg / (ureg.m**3),
                        'pi': 250. * ureg.kg / (ureg.m**3),
                        'gr': 500. * ureg.kg / (ureg.m**3)}
        # Graupel densities can range from 50 to 890 kg/m^3 (Pruppacher and Klett, 1978) so use 500 k/gm^3
        self.fluffy = {'ci': 0.5 * ureg.dimensionless,
                       'pi': 0.5 * ureg.dimensionless,
                       'gr': 0.5 * ureg.dimensionless}
        # Possible fluffiness parameter from Field et al. (2023), Table A1, is 0.422; however, for now this will remain the same as for the other ice species
        self.lidar_ratio = {'cl': 18. * ureg.dimensionless,
                            'ci': 24. * ureg.dimensionless,
                            'pl': 5.5 * ureg.dimensionless,
                            'pi': 24. * ureg.dimensionless,
                            'gr': 24. * ureg.dimensionless}
        # Graupel lidar ratio assumed to be the same as for other ice species
        self.LDR_per_hyd = {'cl': 0.03 * 1 / (ureg.kg / (ureg.m**3)),
                            'ci': 0.35 * 1 / (ureg.kg / (ureg.m**3)),
                            'pl': 0.10 * 1 / (ureg.kg / (ureg.m**3)),
                            'pi': 0.40 * 1 / (ureg.kg / (ureg.m**3)),
                            'gr': 0.40 * 1 / (ureg.kg / (ureg.m**3))}
        # LDR per hydrometeor assumed to be the same as for precipitating ice
        self.vel_param_a = {'cl': 3e-7,
                            'ci': 700.,
                            'pl': 841.997,
                            'pi': 11.72,
                            'gr': 253.}
        # Fall velocity parameter a from Field et al. (2023), Table A1
        self.vel_param_b = {'cl': 2. * ureg.dimensionless,
                            'ci': 1. * ureg.dimensionless,
                            'pl': 0.8 * ureg.dimensionless,
                            'pi': 0.41 * ureg.dimensionless,
                            'gr': 0.734 * ureg.dimensionless}
        # Fall velocity parameter b from Field et al. (2023), Table A1
        super()._add_vel_units()
        # Names of mixing ratio of species
        self.q_names = {'cl': 'mass_fraction_of_cloud_liquid_water_in_air', 'ci': 'mass_fraction_of_cloud_ice_crystals_in_air', 'pl': 'mass_fraction_of_rain_in_air', 'pi': 'mass_fraction_of_cloud_ice_in_air', 'gr': 'mass_fraction_of_graupel_in_air'}
        # Number concentration of each species
        self.N_field = {'cl': 'number_of_cloud_droplets_per_kg_of_air', 'ci': 'number_of_ice_particles_per_kg_of_air', 'pl': 'number_of_rain_drops_per_kg_of_air', 'pi': 'number_of_snow_aggregates_per_kg_of_air', 'gr': 'number_of_graupel_particles_per_kg_of_air'}
        # Stratiform fraction
        # UM provides a generalized (identical for all hydrometeor types) cloud fraction.
        self.strat_frac_names = {'cl': 'cloud_area_fraction_in_atmosphere_layer', 'ci': 'cloud_area_fraction_in_atmosphere_layer', 'pl': 'cloud_area_fraction_in_atmosphere_layer', 'pi': 'cloud_area_fraction_in_atmosphere_layer', 'gr': 'cloud_area_fraction_in_atmosphere_layer'}
        self.strat_frac_names_for_rad = {'cl': 'cloud_area_fraction_in_atmosphere_layer', 'ci': 'cloud_area_fraction_in_atmosphere_layer', 'pl': 'cloud_area_fraction_in_atmosphere_layer', 'pi': 'cloud_area_fraction_in_atmosphere_layer', 'gr': 'cloud_area_fraction_in_atmosphere_layer'}
        # Convective fraction
        self.conv_frac_names = {'cl': 'zeros_var', 'ci': 'zeros_var', 'pl': 'zeros_var', 'pi': 'zeros_var', 'gr': 'zeros_var'}
        self.conv_frac_names_for_rad = {'cl': 'zeros_var', 'ci': 'zeros_var', 'pl': 'zeros_var', 'pi': 'zeros_var', 'gr': 'zeros_var'}
        # strat_re_fields is a dictionary mapping hydrometeor classes to effective radius fields
        # The effective radius fields here are left empty for use when calc_re = True
        self.strat_re_fields = {'cl': 'effective_radius_cl', 'ci': 'effective_radius_ci', 'pl': 'effective_radius_pl', 'pi': 'effective_radius_pi', 'gr': 'effective_radius_gr'}
        # Stratiform mixing ratio
        self.q_names_stratiform = {'cl': 'mass_fraction_of_cloud_liquid_water_in_air', 'ci': 'mass_fraction_of_cloud_ice_crystals_in_air', 'pl': 'mass_fraction_of_rain_in_air', 'pi': 'mass_fraction_of_cloud_ice_in_air', 'gr': 'mass_fraction_of_graupel_in_air'}
        # Convective mixing ratio
        self.q_names_convective = {'cl': 'zeros_var', 'ci': 'zeros_var', 'pl': 'zeros_var', 'pi': 'zeros_var', 'gr': 'zeros_var'}
        # Water vapor mixing ratio
        self.q_field = "zeros_var"
        # Pressure
        self.p_field = "air_pressure"
        # Height
        self.z_field = "height_var"
        # Temperature
        self.T_field = "air_temperature"
        # Name of height dimension
        self.height_dim = "level_height"
        # Name of time dimension
        self.time_dim = "time"
        self.hyd_types = ["cl", "ci", "pl", "pi", "gr"]
        self.process_conv = False

        # Load processed or unprocessed data
        if load_processed:
            self.ds = xr.Dataset()
            self.load_subcolumns_from_netcdf(file_path)
        else:
            self.ds = xr.open_dataset(file_path)
            if appended_str:
                if np.logical_and(not np.any(['ncol' in x for x in self.ds.coords]), all_appended_in_lat):
                    for x in self.ds.dims:
                        if 'ncol' in x:  # ncol in dims but for some reason not in the coords
                            self.ds = self.ds.assign_coords({'ncol': self.ds[x]})
                            self.ds = self.ds.swap_dims({x: "ncol"})
                            break
                super().remove_appended_str(all_appended_in_lat)
                if all_appended_in_lat:
                    self.lat_dim = "ncol"  # here 'ncol' is the spatial dim (acknowledging cube-sphere coords)
            # Add additional time coordinates
            if time_dim == "ncol":
                time_datetime64 = np.array([x.strftime('%Y-%m-%dT%H:%M') for x in self.ds["time"].values],
                                           dtype='datetime64')
                self.ds = self.ds.assign_coords(time=('ncol', time_datetime64))
            # Crop specific model output time range (if requested)
            if time_range is not None:
                if np.issubdtype(time_range.dtype, np.datetime64):
                    if time_dim == "ncol":
                        super()._crop_time_range(time_range, alter_coord="time")
                    else:
                        super()._crop_time_range(time_range)
                else:
                    raise RuntimeError("Input time range is not in the required datetime64 data type")
            # Stack dimensions in the case of a regional output or squeeze lat/lon dims if exist and len==1
            super().check_and_stack_time_lat_lon(file_path=file_path, order_dim=False)
            # Convert pressure units from Pa to hPa
            if self.ds[self.p_field].units == "pascal":
                # Convert pressure field from Pa to hPa
                self.ds[self.p_field] = self.ds[self.p_field] / 100.0
                self.ds[self.p_field].attrs["units"] = "hPa"
            # Calculate air density
            self.ds["rho_a"] = self.ds[self.p_field] * 1e2 / (self.consts["R_d"] * self.ds[self.T_field])
            self.ds["rho_a"].attrs["units"] = "kg / m ** 3"
            # Convert number concentration from 1/kg to 1/cm^3
            for hyd in ["cl", "ci", "pl", "pi", 'gr']:
                if self.ds[self.N_field[hyd]].units == '1 / kilogram':
                    self.ds[self.N_field[hyd]].values *= self.ds["rho_a"].values / 1e6
                    self.ds[self.N_field[hyd]].attrs["units"] = "1 / cm^3"
            # Ensure consistent dimension order (time x height)
            self.permute_dims_for_processing()
            
        # Name the model
        self.model_name = "UM_reg"