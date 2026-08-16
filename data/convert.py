import pandas as pd
df = pd.read_hdf("zenodo_raw/Figure ED1/Casimir_pressure_dist_v4.h5")
df.to_csv("Casimir_pressure_dist_v4.csv", index=False)
print("Saved Casimir_pressure_dist_v4.csv")
