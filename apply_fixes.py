import os

# Fix 1: bootstrap_analysis.py
f1 = "src/bootstrap_analysis.py"
with open(f1, "r", encoding="utf-8") as file:
    content = file.read()

target1 = """    residuals = P_exp - P_model
    
    n_boot = 1000
    chi2_boot = np.zeros(n_boot)
    
    for i in range(n_boot):
        # Resample residuals with replacement
        res_boot = np.random.choice(residuals, size=len(residuals), replace=True)
        P_boot = P_model + res_boot
        chi2_boot[i] = np.sum(((P_boot - P_model) / sigma_noise)**2)"""

replace1 = """    std_residuals = (P_exp - P_model) / sigma_noise

    n_boot = 1000
    chi2_boot = np.zeros(n_boot)

    for i in range(n_boot):
        res_boot = np.random.choice(std_residuals, size=len(std_residuals), replace=True)
        chi2_boot[i] = np.sum(res_boot**2)"""

if target1 in content:
    content = content.replace(target1, replace1)
    with open(f1, "w", encoding="utf-8") as file:
        file.write(content)
    print("Fixed bootstrap_analysis.py")
else:
    print("Could not find target1 in bootstrap_analysis.py")


# Fix 2: casimir_fft_pipeline.py
f2 = "src/casimir_fft_pipeline.py"
with open(f2, "r", encoding="utf-8") as file:
    content = file.read()

target2 = """    noise_floor = np.median(psd_pos)
    sigma_floor = np.std(psd_pos[psd_pos < 3*noise_floor])
    threshold = noise_floor + 3 * sigma_floor"""

replace2 = """    noise_floor = np.median(psd_pos)
    sigma_floor = np.std(psd_pos[psd_pos < 3*noise_floor])
    threshold = noise_floor + 4 * sigma_floor"""

if target2 in content:
    content = content.replace(target2, replace2)
    with open(f2, "w", encoding="utf-8") as file:
        file.write(content)
    print("Fixed casimir_fft_pipeline.py")
else:
    print("Could not find target2 in casimir_fft_pipeline.py")

# Fix 3: sensitivity_table.py
f3 = "src/sensitivity_table.py"
with open(f3, "r", encoding="utf-8") as file:
    content = file.read()

target3 = """                        if max_psd > 0:
                            amp_base = np.sqrt(max_psd) * 4.0 / s_dict['sum_window']
                            amp_sh = np.sqrt(max_psd_sh) * 4.0 / s_dict['sum_window']
                            scalloping_loss_pct = (1.0 - (amp_sh / amp_base)) * 100"""

replace3 = """                        if max_psd > 0 and max_psd_sh > 0:
                            amp_base = np.sqrt(max_psd) * 4.0 / s_dict['sum_window']
                            amp_sh = np.sqrt(max_psd_sh) * 4.0 / s_dict['sum_window']
                            scalloping_loss_pct = (1.0 - (amp_sh / amp_base)) * 100
                        else:
                            scalloping_loss_pct = np.nan"""

if target3 in content:
    content = content.replace(target3, replace3)
    with open(f3, "w", encoding="utf-8") as file:
        file.write(content)
    print("Fixed sensitivity_table.py")
else:
    print("Could not find target3 in sensitivity_table.py")

