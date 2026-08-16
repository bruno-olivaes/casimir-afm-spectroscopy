import pandas as pd
import glob
import os

h5_files = glob.glob(r"C:\Users\bruno\casimir-afm-spectroscopy\data\zenodo_raw\**\*.h5", recursive=True)
for f in h5_files:
    print(f"--- File: {os.path.basename(f)} ---")
    try:
        df = pd.read_hdf(f)
        print("Data columns:")
        print(df.head(5).to_csv(index=False))
        # df.to_csv(f.replace('.h5', '.csv'), index=False)
    except Exception as e:
        print("Could not read:", e)
    print("\n")
