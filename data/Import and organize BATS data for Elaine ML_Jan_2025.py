# Loads data files from BATS and matches data from different files based on date and depth
# This aims to use the files directly from BATS without any manual processing
# WHW, 2021-01-25

import pandas as pd

# Load primary production data txt file
# note: renamed the column 'dep1' to 'Depth' to match other tables 
column_names_pp = [
    "Id", "yymmdd_in", "yymmdd_out", "decy_in", "decy_out", "hhmm_in", "hhmm_out", "Lat_in", "Lat_out", 
    "Long_in", "Long_out", "QF", "Depth", "pres", "temp", "salt", "lt1", "lt2", "lt3", "dark", "t0", "pp"
]
df_pp = pd.read_csv('C:\Users\elain\OneDrive\Documents\Research - BATS\data\bats_primary_production_v003.txt', sep='\s+', comment='/', names=column_names_pp, skiprows=39)
df_pp = df_pp.replace(-999, None)
print("Column headers for df_pp:"), print(df_pp.columns.tolist())
print("Primary Production Data (DF PP TXT):"), print(df_pp.head())

# Load chlorophyll data txt file
column_names_chl = [
    "Id", "yyyymmdd", "decy", "time", "latN", "lonW", "QF", "Depth", "p1", "p2", "p3", "p4", "p5", "p6", 
    "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14", "p15", "Chl", "Phae", "p18", "p19", "p20", "p21"
]
df_chl = pd.read_csv('bats_pigments.txt', sep='\s+', comment='/', names=column_names_chl, skiprows=51)
print("Column headers for df_chl:"), print(df_chl.columns.tolist())
print("Pigments Data (DF Chl TXT):"), print(df_chl.head())

# Load bottle data txt file
# note: renamed the column 'Sal1' to 'Sal' 
# note: renamed the column 'O2(1)' to 'O2' 
# note: renamed the column 'NO31' to 'NO3'
# note: renamed the column 'PO41' to 'PO4'
column_names_bottle = [
    "Id", "yyyymmdd", "decy", "time", "latN", "lonW", "QF", "Depth", "Temp", "CTD_S", "Sal", "Sig-th", 
    "O2", "OxFixT", "Anom1", "CO2", "Alk", "NO3", "NO21", "PO4", "Si1", "POC", "PON", "TOC", "TN", 
    "Bact", "POP", "TDP", "SRP", "BSi", "LSi", "Pro", "Syn", "Piceu", "Naneu"
]
df_bottle = pd.read_csv('bats_bottle.txt', sep='\s+', comment='/', names=column_names_bottle, skiprows=60)
print("Column headers for df_bottle:"), print(df_bottle.columns.tolist())
print("Bottle Data (DF Bottle):"), print(df_bottle.head())

# ### Previous Version ### Load primary porduction data 
# df_pp = pd.read_excel('bats_primary_production.xlsx', sheet_name='data', usecols="R:T", skiprows=0)
# df_pp.columns = ["Date", "Depth", "PP"]  # Add custom column names
# print("Primary Production Data (date frame PP):"), print(df_pp.head())

# ### Previous Version ### Load chlorophyll date 
# df_chl = pd.read_excel('bats_pigments.xlsx', sheet_name='data', usecols="AH:AJ", skiprows=0)
# df_chl.columns = ["Date", "Depth", "Chl"]  # Add custom column names 
# print("Pigments Data (data frame Chl):"), print(df_chl.head())

# ### Perevious version ### Load bottle data 
# df_b = pd.read_excel('bats_bottle_w.xlsx', sheet_name='data', usecols="BC:BP", skiprows=0)
# df_b.columns = ["Date", "Depth", "Temp", "Sal", "O2", "NO3", "PO4", "POC", "PON", "POP", "TOC", "TON", "TOP", "BAC"]  # Add custom column names for matrix_b
# print('Currently ignoring PHY data')
# print("Bottle Data (DF B):"), print(df_b.head())



# Generate a list of unique values in the 'Date' column
unique_dates = df_pp['yymmdd_in'].unique()

# Uncomment for testing a smaller subset of the data 
# unique_dates = unique_dates[0:1]   

# Display the number of unique dates
print(f"Number of unique dates in data frame PP: {len(unique_dates)}")

# Add an empty column 'Chl' to matrix_pp
df_pp['Chl'] = None


# Loop through each unique date
print("Unique date loop value")
for i in range(0, len(unique_dates)):
    date = unique_dates[i]
    # print(f"Processing Date: {date}")
    
    # Find the index of all rows in df_pp where 'Date' matches unique_dates[i]
    pp_indices = df_pp.index[df_pp['yymmdd_in'] == date].tolist()
    # print(f"PP Indices for Date {date}: {pp_indices}")

    # Loop through the rows of df_pp that match the current date
    for ii in pp_indices:
        depth_pp = df_pp.loc[ii, 'Depth']  # Get the depth value from df_pp
        
        #  - - - - - - - - - - - - - - - - - - Chl data matching - - - - - - - - - - - - - - - - - - - - - - - - 
        # Find rows in df_chl where 'Date' is within ±1 of the given date
        filtered_chl = df_chl[(df_chl['yyyymmdd'] >= date - 1) & (df_chl['yyyymmdd'] <= date + 1)]

    
        if not filtered_chl.empty:
            # Find the closest depth in df_chl that matches the depth from df_pp
            closest_chl = filtered_chl[filtered_chl['Chl'].notna()].iloc[(filtered_chl[filtered_chl['Chl'].notna()]['Depth'] - depth_pp).abs().argmin()]
            # Assign the 'Chl' value from the closest row in df_chl to df_pp
            df_pp.at[ii, 'Chl'] = closest_chl['Chl']
        #     print(f"Assigned Chl value {closest_chl['Chl']} to PP index {ii} for Depth {depth_pp}")
        # else:
        #     print(f"No matching chl data found for Date {date} within ±1 days.")

        #  - - - - - - - - - - - - - - - - - - Bottle data matching - - - - - - - - - - - - - - - - - - - - - - - - 
        # Find rows in df_b where 'yyyymmdd' is within ±1 of the given date
        # TODO: current method would not work with first day of one month and last day of previous month
        filtered_b = df_bottle[(df_bottle['yyyymmdd'] >= date - 1) & (df_bottle['yyyymmdd'] <= date + 1)]

            # Assign the bottle data (Temp, Sal, O2, etc.) to df_pp
        filtered_temp = filtered_b[filtered_b['Temp']!=-999]
        # Process 'Temp'
        filtered_temp = filtered_b[filtered_b['Temp']!=-999]
        if not filtered_temp.empty:
            closest_b = filtered_temp.iloc[(filtered_temp['Depth'] - depth_pp).abs().argmin()]
            df_pp.at[ii, 'Temp'] = closest_b['Temp']
        else:
            df_pp.at[ii, 'Temp'] = None  # Optional: Set None if no match is found

        # Process 'Sal'
        filtered_sal = filtered_b[filtered_b['Sal']!=-999]
        if not filtered_sal.empty:
            closest_b = filtered_sal.iloc[(filtered_sal['Depth'] - depth_pp).abs().argmin()]
            df_pp.at[ii, 'Sal'] = closest_b['Sal']
        else:
            df_pp.at[ii, 'Sal'] = None

        # Process 'O2'
        filtered_o2 = filtered_b[filtered_b['O2']!=-999]
        if not filtered_o2.empty:
            closest_b = filtered_o2.iloc[(filtered_o2['Depth'] - depth_pp).abs().argmin()]
            df_pp.at[ii, 'O2'] = closest_b['O2']
        else:
            df_pp.at[ii, 'O2'] = None

        # Process 'NO3'
        filtered_no3 = filtered_b[filtered_b['NO3']!=-999]
        if not filtered_no3.empty:
            closest_b = filtered_no3.iloc[(filtered_no3['Depth'] - depth_pp).abs().argmin()]
            df_pp.at[ii, 'NO3'] = closest_b['NO3']
        else:
            df_pp.at[ii, 'NO3'] = None

        # Process 'PO4'
        filtered_po4 = filtered_b[filtered_b['PO4']!=-999]
        if not filtered_po4.empty:
            closest_b = filtered_po4.iloc[(filtered_po4['Depth'] - depth_pp).abs().argmin()]
            df_pp.at[ii, 'PO4'] = closest_b['PO4']
        else:
            df_pp.at[ii, 'PO4'] = None

        # Process 'POC'
        filtered_poc = filtered_b[filtered_b['POC']!=-999]
        if not filtered_poc.empty:
            closest_b = filtered_poc.iloc[(filtered_poc['Depth'] - depth_pp).abs().argmin()]
            df_pp.at[ii, 'POC'] = closest_b['POC']
        else:
            df_pp.at[ii, 'POC'] = None

        # Process 'PON'
        filtered_pon = filtered_b[filtered_b['PON']!=-999]
        if not filtered_pon.empty:
            closest_b = filtered_pon.iloc[(filtered_pon['Depth'] - depth_pp).abs().argmin()]
            df_pp.at[ii, 'PON'] = closest_b['PON']
        else:
            df_pp.at[ii, 'PON'] = None

        # Process 'POP'
        filtered_pop = filtered_b[filtered_b['POP']!=-999]
        if not filtered_pop.empty:
            closest_b = filtered_pop.iloc[(filtered_pop['Depth'] - depth_pp).abs().argmin()]
            df_pp.at[ii, 'POP'] = closest_b['POP']
        else:
            df_pp.at[ii, 'POP'] = None

        # Process 'TOC'
        filtered_toc = filtered_b[filtered_b['TOC']!=-999]
        if not filtered_toc.empty:
            closest_b = filtered_toc.iloc[(filtered_toc['Depth'] - depth_pp).abs().argmin()]
            df_pp.at[ii, 'TOC'] = closest_b['TOC']
        else:
            df_pp.at[ii, 'TOC'] = None

        # Process 'TN'
        filtered_ton = filtered_b[filtered_b['TN']!=-999]
        if not filtered_ton.empty:
            closest_b = filtered_ton.iloc[(filtered_ton['Depth'] - depth_pp).abs().argmin()]
            df_pp.at[ii, 'TN'] = closest_b['TN']
        else:
            df_pp.at[ii, 'TON'] = None

        # Process 'TDP'
        filtered_tdp = filtered_b[filtered_b['TDP']!=-999]
        if not filtered_tdp.empty:
            closest_b = filtered_tdp.iloc[(filtered_tdp['Depth'] - depth_pp).abs().argmin()]
            df_pp.at[ii, 'TDP'] = closest_b['TDP']
        else:
            df_pp.at[ii, 'TDP'] = None

        # Process 'BAC'
        filtered_bac = filtered_b[filtered_b['Bact']!=-999]
        if not filtered_bac.empty:
            closest_b = filtered_bac.iloc[(filtered_bac['Depth'] - depth_pp).abs().argmin()]
            df_pp.at[ii, 'Bact'] = closest_b['Bact']
        else:
            df_pp.at[ii, 'Bact'] = None

        #     print(f"Assigned Bottle data to PP index {ii} for Depth {depth_pp}")
        # # else:
        # #     print(f"No matching bottle data found for Date {date} within ±1 days.")

# Display the updated df_pp
print("Updated Primary Production Data (df_pp) with Chl and bottle values:")
print(df_pp.head())


# Save the updated dataframe to an Excel file
df_pp.to_excel('matched_data_from_BATS.xlsx', index=False, na_rep='#N/A')


print('Finsihsed')