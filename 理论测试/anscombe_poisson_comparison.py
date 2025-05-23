import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sewar.full_ref import vifp # Import VIF
import time
import matplotlib.pyplot as plt

def generate_checkerboard(size, val_black, val_white):
    img = np.zeros((size, size), dtype=np.float64)
    for r in range(size):
        for c in range(size):
            if (r + c) % 2 == 0: 
                img[r, c] = val_black
            else:
                img[r, c] = val_white
    return img

def anscombe_transform(image):
    # Ensure non-negative values before sqrt for safety, though Poisson samples should be non-negative
    return 2.0 * np.sqrt(np.maximum(0, image) + 3.0/8.0)

# --- Original Metric Calculation Functions (VIF on un-normalized Anscombe where applicable) ---
def calculate_pairwise_metrics(image_set, norm_factor_psnr_ssim, data_range_psnr_ssim, img_size_for_ssim):
    psnr_values = []
    ssim_values = []
    vif_values = [] 
    num_images = len(image_set)
    if num_images < 2:
        return [], [], [] 

    for i in range(num_images):
        for j in range(i + 1, num_images):
            img1_for_psnr_ssim = image_set[i] / norm_factor_psnr_ssim
            img2_for_psnr_ssim = image_set[j] / norm_factor_psnr_ssim
            current_psnr = psnr(img1_for_psnr_ssim, img2_for_psnr_ssim, data_range=data_range_psnr_ssim)
            current_ssim = ssim(img1_for_psnr_ssim, img2_for_psnr_ssim, data_range=data_range_psnr_ssim, 
                                multichannel=False, win_size=min(7, img_size_for_ssim),
                                gaussian_weights=True, use_sample_covariance=False)
            psnr_values.append(current_psnr)
            ssim_values.append(current_ssim)
            img1_for_vif = image_set[i]
            img2_for_vif = image_set[j]
            try: current_vif = vifp(img1_for_vif, img2_for_vif)
            except Exception: current_vif = np.nan
            vif_values.append(current_vif)
    return psnr_values, ssim_values, vif_values

def calculate_metrics_vs_ideal(image_set, ideal_image, norm_factor_psnr_ssim, data_range_psnr_ssim, img_size_for_ssim):
    psnr_values, ssim_values, vif_values = [], [], []
    ideal_image_for_psnr_ssim = ideal_image / norm_factor_psnr_ssim
    ideal_image_for_vif = ideal_image
    for noisy_img in image_set:
        noisy_img_for_psnr_ssim = noisy_img / norm_factor_psnr_ssim
        current_psnr = psnr(ideal_image_for_psnr_ssim, noisy_img_for_psnr_ssim, data_range=data_range_psnr_ssim)
        current_ssim = ssim(ideal_image_for_psnr_ssim, noisy_img_for_psnr_ssim, data_range=data_range_psnr_ssim, 
                            multichannel=False, win_size=min(7, img_size_for_ssim),
                            gaussian_weights=True, use_sample_covariance=False)
        psnr_values.append(current_psnr)
        ssim_values.append(current_ssim)
        noisy_img_for_vif = noisy_img
        try: current_vif = vifp(ideal_image_for_vif, noisy_img_for_vif)
        except Exception: current_vif = np.nan
        vif_values.append(current_vif)
    return psnr_values, ssim_values, vif_values

# --- New VIF Calculation Functions for Experiment 3 (VIF on specifically normalized and CLIPPED inputs) ---
def calculate_pairwise_vif_on_normalized_images(image_set_to_normalize, vif_norm_factor, img_size_for_ssim_win_unused):
    vif_values = []
    num_images = len(image_set_to_normalize)
    if num_images < 2: return []
    
    # Normalize and then clip to [0, 1] before VIF calculation
    normalized_image_set = [np.clip(img / vif_norm_factor, 0.0, 1.0) for img in image_set_to_normalize]

    for i in range(num_images):
        for j in range(i + 1, num_images):
            img1_norm_clipped_for_vif = normalized_image_set[i]
            img2_norm_clipped_for_vif = normalized_image_set[j]
            try: current_vif = vifp(img1_norm_clipped_for_vif, img2_norm_clipped_for_vif)
            except Exception as e:
                # print(f"VIF calc error (pairwise-normed-clipped): {e}. Appending NaN.")
                current_vif = np.nan
            vif_values.append(current_vif)
    return vif_values

def calculate_cross_vif_on_normalized_images(set1_to_normalize, set2_to_normalize, vif_norm_factor_set1, vif_norm_factor_set2, img_size_for_ssim_win_unused):
    vif_values = []
    if not set1_to_normalize or not set2_to_normalize: return []

    # Normalize each set with its factor, then clip to [0, 1] before VIF calculation
    normalized_clipped_set1 = [np.clip(img / vif_norm_factor_set1, 0.0, 1.0) for img in set1_to_normalize]
    normalized_clipped_set2 = [np.clip(img / vif_norm_factor_set2, 0.0, 1.0) for img in set2_to_normalize]

    for img1_norm_clipped_for_vif in normalized_clipped_set1:
        for img2_norm_clipped_for_vif in normalized_clipped_set2:
            try: current_vif = vifp(img1_norm_clipped_for_vif, img2_norm_clipped_for_vif)
            except Exception as e:
                # print(f"VIF calc error (cross-normed-clipped): {e}. Appending NaN.")
                current_vif = np.nan
            vif_values.append(current_vif)
    return vif_values

# --- NEW VIF Calculation Function for VIF vs Ideal with specific normalizations and CLIPPING for VIF input ---
def calculate_vif_vs_ideal_on_normalized_images(noisy_anscombe_set, ideal_anscombe_image, 
                                                vif_norm_factor_noisy_set, vif_norm_factor_ideal_image):
    vif_values = []
    if not noisy_anscombe_set: return []

    # Normalize and clip the ideal image
    normalized_clipped_ideal_image = np.clip(ideal_anscombe_image / vif_norm_factor_ideal_image, 0.0, 1.0)

    for noisy_ansc_img in noisy_anscombe_set:
        # Normalize and clip the current noisy image
        normalized_clipped_noisy_img = np.clip(noisy_ansc_img / vif_norm_factor_noisy_set, 0.0, 1.0)
        try:
            current_vif = vifp(normalized_clipped_ideal_image, normalized_clipped_noisy_img)
        except Exception as e:
            # print(f"VIF calc error (vs ideal, normed-clipped): {e}. Appending NaN.")
            current_vif = np.nan
        vif_values.append(current_vif)
    return vif_values

# --- Statistics Functions ---
def get_stats_dict(metric_values):
    valid_metric_values = [x for x in metric_values if not np.isnan(x)]
    if not valid_metric_values:
        return {'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan, 'count': 0, 'nan_count': len(metric_values)}
    return {
        'mean': np.mean(valid_metric_values), 'median': np.median(valid_metric_values),
        'std': np.std(valid_metric_values), 'min': np.min(valid_metric_values),
        'max': np.max(valid_metric_values), 'count': len(valid_metric_values),
        'nan_count': len(metric_values) - len(valid_metric_values)
    }

def print_stats_from_dict(stats_dict, name, unit=""):
    if stats_dict['count'] == 0:
        print(f"  {name}: No valid values to calculate statistics (all {stats_dict['nan_count']} were NaN)." if stats_dict['nan_count'] > 0 else f"  {name}: No values to calculate statistics.")
        return
    print(f"  {name} (NaNs: {stats_dict['nan_count' ]}/{stats_dict['nan_count']+stats_dict['count']}):")
    print(f"    Mean:   {stats_dict['mean']:.4f} {unit}")
    print(f"    Median: {stats_dict['median']:.4f} {unit}")
    print(f"    StdDev: {stats_dict['std']:.4f} {unit}")
    print(f"    Min:    {stats_dict['min']:.4f} {unit}")
    print(f"    Max:    {stats_dict['max']:.4f} {unit}")

def main():
    img_size = 16
    cb_black_val_orig, cb_white_val_orig = 200.0, 100.0 # User changed values for original CB
    num_noisy_samples_analysis = 100

    print(f"Starting analysis with {num_noisy_samples_analysis} samples for a {img_size}x{img_size} checkerboard.")
    print(f"Original Checkerboard (CB1) values: Black={cb_black_val_orig}, White={cb_white_val_orig}\n")

    ideal_checkerboard_original = generate_checkerboard(img_size, cb_black_val_orig, cb_white_val_orig)
    noisy_samples_original = [np.random.poisson(ideal_checkerboard_original).astype(np.float64) for _ in range(num_noisy_samples_analysis)]
    noisy_samples_anscombe_set1 = [anscombe_transform(img) for img in noisy_samples_original]

    # --- Experiment 1 & 2: Original and Anscombe Domain (Standard VIF handling) ---
    # This section can be re-enabled if full Exp1 & Exp2 outputs are needed.
    # For now, we summarize data generation and proceed to Exp3 which uses noisy_samples_anscombe_set1.
    print("--- Experiments 1 & 2 Data Generation (Summary for Exp3/Exp4 context) ---")
    # For Exp1 & 2, VIF is calculated on images *before* PSNR/SSIM specific normalization factors are applied.
    # For Exp1: images are noisy_samples_original, ideal_checkerboard_original
    # For Exp2: images are noisy_samples_anscombe_set1, anscombe_transform(ideal_checkerboard_original)
    # The stats from these would be used for the plots if uncommented later.
    print(f"Generated {len(noisy_samples_original)} original domain noisy samples (for Exp1 context). ")
    print(f"Generated {len(noisy_samples_anscombe_set1)} Anscombe transformed samples (Set 1) from CB1 (for Exp2, Exp3, Exp4 context).\\n")
    
    # --- Experiment 3: VIF comparison with specific normalizations and clipping for VIF input (MODIFIED) ---
    print("--- Experiment 3 (Modified): VIF Comparison with Specific Normalizations and Clipping ---")

    # CB3 base is CB1/4
    cb_black_val_set3_base = cb_black_val_orig / 4.0
    cb_white_val_set3_base = cb_white_val_orig / 4.0
    ideal_checkerboard_set3_orig_domain_base = generate_checkerboard(img_size, cb_black_val_set3_base, cb_white_val_set3_base)
    print(f"Checkerboard 3 (CB3) base definition for sampling: Black={cb_black_val_set3_base}, White={cb_white_val_set3_base}")

    # Poisson samples from CB3 base, THEN multiply by 4.0, THEN Anscombe transform
    noisy_samples_set3_orig_domain_base = [np.random.poisson(ideal_checkerboard_set3_orig_domain_base).astype(np.float64) for _ in range(num_noisy_samples_analysis)]
    noisy_samples_set3_multiplied = [img * 4.0 for img in noisy_samples_set3_orig_domain_base]
    noisy_samples_anscombe_set3 = [anscombe_transform(img) for img in noisy_samples_set3_multiplied]
    print(f"Generated {len(noisy_samples_anscombe_set3)} Anscombe transformed samples (Set 3).")
    print("(Set 3 generation: Poisson samples from CB1/4, then pixel values * 4.0, then Anscombe transform)")

    # VIF Normalization Factors for Experiment 3
    norm_param_for_vif_set1 = 230.0  # For Set 1
    vif_norm_factor_set1_ansc = anscombe_transform(norm_param_for_vif_set1)
    
    norm_param_for_vif_set3_exp3 = 230.0  # For Set 3, as per new requirement (same as Set 1)
    vif_norm_factor_set3_ansc_exp3 = anscombe_transform(norm_param_for_vif_set3_exp3)

    print(f"VIF Input Norm Factor for Anscombe Set 1 (Exp3): A({norm_param_for_vif_set1}) = {vif_norm_factor_set1_ansc:.4f}")
    print(f"VIF Input Norm Factor for Anscombe Set 3 (Exp3): A({norm_param_for_vif_set3_exp3}) = {vif_norm_factor_set3_ansc_exp3:.4f}\\n")
    print("Note: For Experiment 3 VIF, inputs are normalized by these factors AND THEN CLIPPED to [0,1].")

    print("Calculating: VIF Internal for (Anscombe Set 1 / A(230.0) then clipped)")
    start_time = time.time()
    vif_internal_set1_ansc_normA230_vals = calculate_pairwise_vif_on_normalized_images(
        noisy_samples_anscombe_set1, vif_norm_factor_set1_ansc, img_size)
    stats_vif_internal_set1_ansc_normA230 = get_stats_dict(vif_internal_set1_ansc_normA230_vals)
    print(f"Calculation time: {time.time() - start_time:.2f}s")
    print_stats_from_dict(stats_vif_internal_set1_ansc_normA230, "VIF Internal (Anscombe Set 1 / A(230.0) clipped)")

    print(f"\\nCalculating: VIF Cross between (Anscombe Set 1 / A({norm_param_for_vif_set1}) clipped) and (Anscombe Set 3 / A({norm_param_for_vif_set3_exp3}) clipped)")
    start_time = time.time()
    vif_cross_set1A230_set3A230_vals = calculate_cross_vif_on_normalized_images( # Renamed variable for clarity
        noisy_samples_anscombe_set1, noisy_samples_anscombe_set3, 
        vif_norm_factor_set1_ansc, vif_norm_factor_set3_ansc_exp3, img_size)
    stats_vif_cross_set1A230_set3A230 = get_stats_dict(vif_cross_set1A230_set3A230_vals) # Renamed variable
    print(f"Calculation time: {time.time() - start_time:.2f}s")
    print_stats_from_dict(stats_vif_cross_set1A230_set3A230, f"VIF Cross (Set1_Ansc/A({norm_param_for_vif_set1}) clip vs Set3_Ansc/A({norm_param_for_vif_set3_exp3}) clip)")
    
    print("\\nExperiment 3 (Modified) complete.\\n")

    # --- Experiment 4: New VIF comparison ---
    print("--- Experiment 4: VIF Comparison with CB4 (CB1/8 based samples) ---")

    # CB4 base is CB1/8
    cb_black_val_set4_base = cb_black_val_orig / 4.0
    cb_white_val_set4_base = cb_white_val_orig / 4.0
    ideal_checkerboard_set4_orig_domain_base = generate_checkerboard(img_size, cb_black_val_set4_base, cb_white_val_set4_base)
    print(f"Checkerboard 4 (CB4) base definition for sampling: Black={cb_black_val_set4_base}, White={cb_white_val_set4_base}")

    # Poisson samples from CB4 base, THEN Anscombe transform (NO intermediate multiplication for Set 4)
    noisy_samples_set4_orig_domain_base = [np.random.poisson(ideal_checkerboard_set4_orig_domain_base).astype(np.float64) for _ in range(num_noisy_samples_analysis)]
    noisy_samples_anscombe_set4 = [anscombe_transform(img) for img in noisy_samples_set4_orig_domain_base]
    print(f"Generated {len(noisy_samples_anscombe_set4)} Anscombe transformed samples (Set 4) from CB4 base.")
    print("(Set 4 generation: Poisson samples from CB1/8, then Anscombe transform)")

    # VIF Normalization Factors for Experiment 4
    # Set 1 still uses A(230.0) -> vif_norm_factor_set1_ansc (already computed)
    norm_param_for_vif_set4_exp4 = 230.0 / 8.0
    vif_norm_factor_set4_ansc_exp4 = anscombe_transform(norm_param_for_vif_set4_exp4)

    print(f"VIF Input Norm Factor for Anscombe Set 1 (Exp4 context): A({norm_param_for_vif_set1}) = {vif_norm_factor_set1_ansc:.4f}")
    print(f"VIF Input Norm Factor for Anscombe Set 4 (Exp4): A({norm_param_for_vif_set4_exp4:.3f}) = {vif_norm_factor_set4_ansc_exp4:.4f}\\n") # Using .3f for norm_param for print consistency
    print("Note: For Experiment 4 VIF, inputs are normalized by these factors AND THEN CLIPPED to [0,1].")

    print(f"Calculating: VIF Cross between (Anscombe Set 1 / A({norm_param_for_vif_set1}) clipped) and (Anscombe Set 4 / A({norm_param_for_vif_set4_exp4:.3f}) clipped)")
    start_time = time.time()
    vif_cross_set1_vs_set4_exp4_vals = calculate_cross_vif_on_normalized_images(
        noisy_samples_anscombe_set1, noisy_samples_anscombe_set4, 
        vif_norm_factor_set1_ansc, vif_norm_factor_set4_ansc_exp4, img_size)
    stats_vif_cross_set1_vs_set4_exp4 = get_stats_dict(vif_cross_set1_vs_set4_exp4_vals)
    print(f"Calculation time: {time.time() - start_time:.2f}s")
    print_stats_from_dict(stats_vif_cross_set1_vs_set4_exp4, f"VIF Cross (Set1_Ansc/A({norm_param_for_vif_set1}) clip vs Set4_Ansc/A({norm_param_for_vif_set4_exp4:.3f}) clip)")

    print(f"\\nCalculating: VIF Internal for (Anscombe Set 4 / A({norm_param_for_vif_set4_exp4:.3f}) then clipped)")
    start_time = time.time()
    vif_internal_set4_ansc_norm_vals = calculate_pairwise_vif_on_normalized_images( # Renamed variable
        noisy_samples_anscombe_set4, vif_norm_factor_set4_ansc_exp4, img_size)
    stats_vif_internal_set4_ansc_norm = get_stats_dict(vif_internal_set4_ansc_norm_vals) # Renamed variable
    print(f"Calculation time: {time.time() - start_time:.2f}s")
    print_stats_from_dict(stats_vif_internal_set4_ansc_norm, f"VIF Internal (Anscombe Set 4 / A({norm_param_for_vif_set4_exp4:.3f}) clipped)")
    
    print("\\nExperiment 4 complete.")
    
    # --- Extended VIF Analysis vs Ideal CB1 (Normalized and Clipped) ---
    print("--- Extended VIF Analysis vs Ideal CB1 (Normalized and Clipped by A(230.0)) ---")

    ideal_anscombe_set1 = anscombe_transform(ideal_checkerboard_original) # CB1 Anscombe transformed
    vif_norm_factor_A230 = anscombe_transform(230.0) # Common normalization factor A(230.0)
    
    print(f"Common VIF Input Norm Factor for this section: A(230.0) = {vif_norm_factor_A230:.4f}")
    print("Inputs (noisy sets and ideal CB1) are normalized by this factor AND THEN CLIPPED to [0,1] before VIF.")

    # 1. VIF (Set1_Ansc vs Ideal1_Ansc, both normalized by A(230.0) and clipped)
    print(f"\\nCalculating: VIF between (Anscombe Set 1 / A(230.0) clipped) and (Ideal CB1 Anscombe / A(230.0) clipped)")
    start_time = time.time()
    vif_set1_vs_ideal1_A230_vals = calculate_vif_vs_ideal_on_normalized_images(
        noisy_samples_anscombe_set1, 
        ideal_anscombe_set1,
        vif_norm_factor_A230,  # norm factor for noisy_samples_anscombe_set1
        vif_norm_factor_A230   # norm factor for ideal_anscombe_set1
    )
    stats_vif_set1_vs_ideal1_A230 = get_stats_dict(vif_set1_vs_ideal1_A230_vals)
    print(f"Calculation time: {time.time() - start_time:.2f}s")
    print_stats_from_dict(stats_vif_set1_vs_ideal1_A230, "VIF (Set1 Ansc vs Ideal1 Ansc, norm A(230) clipped)")

    # 2. VIF (Set3_Ansc vs Ideal1_Ansc, both normalized by A(230.0) and clipped)
    # noisy_samples_anscombe_set3 is already defined and generated in Experiment 3
    # ideal_anscombe_set1 is already defined
    # vif_norm_factor_A230 is the common normalization factor
    print(f"\\nCalculating: VIF between (Anscombe Set 3 / A(230.0) clipped) and (Ideal CB1 Anscombe / A(230.0) clipped)")
    start_time = time.time()
    vif_set3_vs_ideal1_A230_vals = calculate_vif_vs_ideal_on_normalized_images(
        noisy_samples_anscombe_set3, # from Exp3 (CB1/2 base, samples*2, then Ansc)
        ideal_anscombe_set1,
        vif_norm_factor_A230,  # norm factor for noisy_samples_anscombe_set3
        vif_norm_factor_A230   # norm factor for ideal_anscombe_set1
    )
    stats_vif_set3_vs_ideal1_A230 = get_stats_dict(vif_set3_vs_ideal1_A230_vals)
    print(f"Calculation time: {time.time() - start_time:.2f}s")
    print_stats_from_dict(stats_vif_set3_vs_ideal1_A230, "VIF (Set3 Ansc vs Ideal1 Ansc, norm A(230) clipped)")

    print("\\\\nExtended VIF Analysis complete.")

    # Note: Plots from previous version for Exp1 & Exp2 are currently commented out / not run.
    # To re-enable them, the full stats from Exp1 & Exp2 would need to be calculated and passed.
    # Example (requires stats_psnr_orig_internal etc. to be populated by running full Exp1&2):
    # fig_psnr, axs_psnr = plt.subplots(2, 1, figsize=(10, 8))
    # ... (rest of plotting code for PSNR, SSIM, VIF gaps from Exp1&2)
    # plt.show()

if __name__ == '__main__':
    main()

