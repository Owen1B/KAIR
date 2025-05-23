# 该脚本用于测试基于条件二项分布模型的剂量比（kappa）估计方法，以及使用轮廓似然（profile likelihood）估计未知缩放因子的方法。
# 脚本模拟不同真实剂量比下的两幅SPECT图像，然后尝试从这些图像中恢复剂量比或相关参数。
# 主要输出:
# 1. 'profile_loglik_d2factor_*.png': 对于每个测试的真实剂量比因子，生成轮廓对数似然函数随候选缩放因子变化的曲线图。
# 2. 控制台输出详细的统计信息，包括：
#    - 直接从总计数估计的kappa值、其置信区间和显著性检验结果。
#    - 通过轮廓似然估计的缩放因子 (alpha_hat) 和由此间接得到的kappa估计值。
#    - 真实kappa值与估计值的比较。
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gammaln # For log factorial, if needed
import os
from skimage.transform import resize # If resizing SPECT needed for speed

# Setup Matplotlib for Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the SPECT file, assuming a project structure like KAIR/trainsets and KAIR/scripts
DEFAULT_SPECT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'trainsets', 'spectH_raw', 'B001_pure.dat'))
# Fallback if the script is run from a different relative location
ALT_DEFAULT_SPECT_FILE = os.path.abspath(os.path.join('trainsets', 'spectH_raw', 'B001_pure.dat'))


# --- Helper Functions ---
def load_spect_anterior_data(file_path_param=DEFAULT_SPECT_FILE):
    """Loads the SPECT .dat file and returns the anterior view."""
    file_path = file_path_param
    if not os.path.exists(file_path):
        print(f"Warning: SPECT data file not found at primary path: {file_path}")
        # Try alternative common path if script is e.g. in KAIR/
        alt_path_from_kair_root = ALT_DEFAULT_SPECT_FILE
        if os.path.exists(alt_path_from_kair_root):
            print(f"Found SPECT data at alternative path: {alt_path_from_kair_root}")
            file_path = alt_path_from_kair_root
        else:
            # Attempt path construction assuming script is in a subdir of project root
            # e.g., project_root/scripts/this_script.py
            # project_root/trainsets/spectH_raw/B001_pure.dat
            path_strat1 = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'trainsets', 'spectH_raw', os.path.basename(file_path_param)))
            # e.g., project_root/some_dir/this_script.py
            # project_root/trainsets/spectH_raw/B001_pure.dat
            path_strat2 = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'trainsets', 'spectH_raw', os.path.basename(file_path_param)))
            
            paths_to_try = [path_strat1, path_strat2]
            found_path = None
            for p_try in paths_to_try:
                if os.path.exists(p_try):
                    file_path = p_try
                    found_path = p_try
                    print(f"Found SPECT data at: {file_path}")
                    break
            if not found_path:
                print(f"Error: SPECT data file not found at primary, alternative, or constructed paths. Please check path: {file_path_param}")
                return None
                
    print(f"Loading SPECT file from: {file_path}")
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        # Reshape based on known dimensions for B001_pure.dat: (views, height, width)
        reshaped_data = data.reshape(2, 1024, 256) 
        return reshaped_data[0].astype(np.float32)  # Return anterior view
    except Exception as e:
        print(f"Error loading SPECT data ({file_path}): {e}")
        return None

def poisson_sample(ideal_image):
    """Performs Poisson sampling on an ideal image."""
    return np.random.poisson(np.maximum(ideal_image, 0)).astype(np.float32)

# --- Core Statistical Functions ---

def calculate_kappa_from_sums(S1, S2, S1_label="S1", S2_label="S2"):
    """
    Estimates kappa (dose_ratio = d2/d1) from total counts S1 and S2.
    Calculates CI and performs a z-test for H0: kappa = 1.
    """
    print(f"\n--- Estimating Kappa from {S1_label}={S1}, {S2_label}={S2} ---")
    if S1 < 0 or S2 < 0:
        print("Error: S1 and S2 must be non-negative.")
        return np.nan, (np.nan, np.nan), np.nan, np.nan

    if S1 == 0 and S2 == 0:
        print("Warning: Both S1 and S2 are 0. Kappa is indeterminate.")
        return np.nan, (np.nan, np.nan), np.nan, np.nan
    
    kappa_hat = np.nan
    if S1 > 0:
        kappa_hat = S2 / S1
    elif S2 > 0: # S1 is 0, S2 > 0
        kappa_hat = np.inf
    # else S1=0, S2=0, kappa_hat remains np.nan

    print(f"Kappa_hat ({S2_label}/{S1_label}): {kappa_hat:.4f}")

    # For CI and z-test, use continuity correction for SE
    S1_eff = S1 + 0.5
    S2_eff = S2 + 0.5
    
    log_kappa_hat = np.nan
    if kappa_hat > 0 and np.isfinite(kappa_hat): # kappa_hat is positive finite
        log_kappa_hat = np.log(kappa_hat)
    elif kappa_hat == 0: # S2=0, S1>0
        log_kappa_hat = -np.inf
    elif np.isinf(kappa_hat): # S1=0, S2>0
        log_kappa_hat = np.inf

    se_log_kappa = np.sqrt(1/S1_eff + 1/S2_eff)
    
    ci_low, ci_high = np.nan, np.nan
    z_stat, p_value = np.nan, np.nan

    if np.isfinite(log_kappa_hat) and np.isfinite(se_log_kappa) and se_log_kappa > 0:
        ci_low = kappa_hat * np.exp(-1.96 * se_log_kappa)
        ci_high = kappa_hat * np.exp(1.96 * se_log_kappa)
        print(f"95% CI for Kappa: [{ci_low:.4f}, {ci_high:.4f}]")

        z_stat = log_kappa_hat / se_log_kappa # Test log_kappa = 0
        p_value = 2 * stats.norm.sf(np.abs(z_stat)) # Two-sided p-value
        print(f"Z-statistic for log(kappa_hat)=0: {z_stat:.4f}, P-value: {p_value:.4e}")
        if p_value < 0.05:
            print(f"  Conclusion: Reject H0: kappa=1. Statistically significant difference in dose (kappa != 1).")
        else:
            print(f"  Conclusion: Cannot reject H0: kappa=1. No statistically significant difference in dose.")

    elif log_kappa_hat == -np.inf : # kappa_hat = 0
         print(f"Kappa_hat is 0. Strong evidence kappa < 1 (implies d2 significantly less than d1, or d2=0).")
         z_stat = -np.inf; p_value = 0.0 
         ci_low, ci_high = 0.0, 0.0 # CI is [0,0] effectively
         print(f"  Z-statistic: -inf, P-value: {p_value:.4e}. Conclusion: Reject H0: kappa=1.")
    elif np.isinf(log_kappa_hat): # kappa_hat = Inf
         print(f"Kappa_hat is Inf. Strong evidence kappa > 1 (implies d1 significantly less than d2, or d1=0).")
         z_stat = np.inf; p_value = 0.0
         # CI lower bound could be estimated based on S2 if S1 was e.g. 1, upper is inf.
         # For simplicity, report as [some_large_num, inf] or just state highly significant.
         ci_low, ci_high = kappa_hat, np.inf # Effectively [large, inf]
         print(f"  Z-statistic: +inf, P-value: {p_value:.4e}. Conclusion: Reject H0: kappa=1.")
    
    return kappa_hat, (ci_low, ci_high), z_stat, p_value

def profile_log_likelihood(alpha, Y1_counts, Z2_normalized_0_1, min_Y2_original_counts):
    """Calculates the profile log-likelihood for a given alpha (candidate c2 scaling factor)."""
    Y2_candidate_counts = (Z2_normalized_0_1 * alpha + min_Y2_original_counts).round().astype(int)
    Y2_candidate_counts = np.maximum(Y2_candidate_counts, 0) # Ensure non-negative counts

    S1 = np.sum(Y1_counts)
    S2_candidate = np.sum(Y2_candidate_counts)
    N_total = S1 + S2_candidate

    if N_total == 0:
        return -np.inf # Or a very small number if S1 and S2_candidate are both zero
    
    # Formula: S1*log(p_hat) + S2*log(1-p_hat) where p_hat = S1/N_total
    # This is maximized log-likelihood for Binomial(N_total, p_hat)
    log_L = 0
    # Handle terms where S1 or S2_candidate is 0 to avoid log(0) issues.
    # x*log(x) -> 0 as x -> 0. So if S_i = 0, its term is 0.
    # If S_i > 0 but S_i/N_total = 0 (can't happen if S_i > 0), or S_i/N_total = 1.
    if S1 > 0:
        log_L += S1 * np.log(S1 / N_total)
    if S2_candidate > 0:
        log_L += S2_candidate * np.log(S2_candidate / N_total)
        
    return log_L

# --- Main Simulation Logic ---
def main():
    print("--- Loading Ideal SPECT Image ---")
    I_orig = load_spect_anterior_data()
    if I_orig is None:
        print("Failed to load SPECT image. Exiting.")
        return
    
    # Ensure ideal image has some counts to avoid issues with Poisson sampling log(0) or mean 0
    I_orig = np.maximum(I_orig, 1e-6) # Add a tiny baseline if it has true zeros
    print(f"Loaded I_orig. Shape: {I_orig.shape}, Min: {I_orig.min():.2f}, Max: {I_orig.max():.2f}")

    d1_true = 1.0 # Dose for Image 1 (reference)
    
    # Factors to multiply d1_true to get d2_true for Image 2
    d2_true_factors = [1.0, 0.7, 0.5, 1.5, 0.1] 

    for d2_factor_idx, d2_true_factor in enumerate(d2_true_factors):
        d2_true = d1_true * d2_true_factor
        print(f"\n======================================================================")
        print(f"Test {d2_factor_idx+1}/{len(d2_true_factors)}: d1_true = {d1_true:.2f}, d2_true = {d2_true:.2f} (Factor = {d2_true_factor:.2f})")
        print(f"======================================================================")

        # Generate count images based on true doses and ideal image
        Y1_actual_counts = poisson_sample(d1_true * I_orig).round().astype(int)
        Y2_actual_counts = poisson_sample(d2_true * I_orig).round().astype(int)
        
        print(f"Y1_actual_counts: Min={Y1_actual_counts.min()}, Max={Y1_actual_counts.max()}, Sum={np.sum(Y1_actual_counts)}")
        print(f"Y2_actual_counts: Min={Y2_actual_counts.min()}, Max={Y2_actual_counts.max()}, Sum={np.sum(Y2_actual_counts)}")

        # --- Part A: Assume Y1_counts and Y2_counts are known (c1, c2 implicitly known) ---
        print("\n--- Part A: Direct Kappa Estimation (Y1, Y2 counts known) ---")
        S1_A = np.sum(Y1_actual_counts)
        S2_A = np.sum(Y2_actual_counts)
        kappa_hat_A, ci_A, z_A, pval_A = calculate_kappa_from_sums(S1_A, S2_A)
        print(f"True kappa (d2/d1): {d2_true/d1_true:.4f}")

        # --- Part B: Y1_counts known, Y2 is 0-1 normalized (c2 unknown), estimate c2 (alpha) and kappa ---
        print("\n--- Part B: Profile Likelihood for c2 (alpha) and Kappa ---")
        Y1_input_B = Y1_actual_counts.copy()

        min_Y2_actual = np.min(Y2_actual_counts)
        max_Y2_actual = np.max(Y2_actual_counts)
        scale_Y2_actual = max_Y2_actual - min_Y2_actual

        if scale_Y2_actual <= 1e-6 : # Effectively zero or constant image
            print("Warning: Y2_actual_counts have no significant range (scale_Y2_actual is near zero). Skipping profile likelihood.")
            Z2_norm_input_B = np.zeros_like(Y2_actual_counts, dtype=float)
            alpha_hat_B = 0
            kappa_hat_B = 0 # Or handle as per calculate_kappa_from_sums if S1_B > 0
            if np.sum(Y1_input_B) > 0 :
                 kappa_hat_B_val, _, _, _ = calculate_kappa_from_sums(np.sum(Y1_input_B), 0) # S2 for Y2 will be 0 if alpha_hat_B is 0
                 kappa_hat_B = kappa_hat_B_val if np.isfinite(kappa_hat_B_val) else 0
            else: # Both S1 and S2 are effectively zero.
                 kappa_hat_B = np.nan
        else:
            Z2_norm_input_B = (Y2_actual_counts - min_Y2_actual) / scale_Y2_actual
            
            # Define alpha search space (candidate values for scale_Y2_actual)
            # Adapt range based on expected scale_Y2_actual, or use a broad fixed range.
            # Max of SPECT I_orig is ~300. If d2_true=1, scale_Y2_actual also ~300.
            # If d2_true is small, scale_Y2_actual will be small.
            alpha_low_bound = max(1.0, scale_Y2_actual * 0.2 if scale_Y2_actual > 0 else 1.0)
            alpha_high_bound = max(10.0, scale_Y2_actual * 1.8 if scale_Y2_actual > 0 else 10.0)
            # Ensure high bound is meaningfully larger than low bound
            if alpha_high_bound <= alpha_low_bound: alpha_high_bound = alpha_low_bound + 10
            
            alpha_search_space = np.linspace(alpha_low_bound, alpha_high_bound, 100) # 100 points in search
            if scale_Y2_actual > 0 and not (alpha_low_bound <= scale_Y2_actual <= alpha_high_bound): # Ensure true value is in search space for viz
                 alpha_search_space = np.sort(np.append(alpha_search_space, scale_Y2_actual))

            print(f"Searching for alpha (est. c2 scale) in range [{alpha_search_space.min():.2f}, {alpha_search_space.max():.2f}]. True scale_Y2_actual: {scale_Y2_actual:.2f}")

            log_likelihood_values = []
            for alpha_candidate in alpha_search_space:
                ll_val = profile_log_likelihood(alpha_candidate, Y1_input_B, Z2_norm_input_B, min_Y2_actual)
                log_likelihood_values.append(ll_val)
            
            if not log_likelihood_values or np.all(np.isneginf(log_likelihood_values)):
                print("Error: All log-likelihood values are -infinity. Cannot find alpha_hat.")
                alpha_hat_B = np.nan
            else:
                alpha_hat_B = alpha_search_space[np.nanargmax(log_likelihood_values)] # Use nanargmax

            # Reconstruct Y2 counts with estimated alpha_hat_B
            Y2_counts_estimated_B = (Z2_norm_input_B * alpha_hat_B + min_Y2_actual).round().astype(int)
            Y2_counts_estimated_B = np.maximum(Y2_counts_estimated_B, 0)
            
            S1_B = np.sum(Y1_input_B)
            S2_B_estimated = np.sum(Y2_counts_estimated_B)
            
            kappa_hat_B_val, _, _, _ = calculate_kappa_from_sums(S1_B, S2_B_estimated, S2_label="S2_est_B")
            kappa_hat_B = kappa_hat_B_val
            
            print(f"Estimated alpha_hat_B (c2 scale factor): {alpha_hat_B:.4f} (True scale_Y2_actual: {scale_Y2_actual:.4f})")
            print(f"Resulting Kappa_hat_B (from Y1 and Y2_est_B): {kappa_hat_B:.4f} (True kappa: {d2_true/d1_true:.4f})")

            # Plot profile log-likelihood
            plt.figure(figsize=(10, 6))
            plt.plot(alpha_search_space, log_likelihood_values, 'o-')
            plt.axvline(scale_Y2_actual, color='r', linestyle='--', label=f'True scale_Y2_actual = {scale_Y2_actual:.2f}')
            if np.isfinite(alpha_hat_B):
                plt.axvline(alpha_hat_B, color='g', linestyle=':', label=f'Estimated alpha_hat_B = {alpha_hat_B:.2f}')
            plt.xlabel("Alpha (Candidate c2 Scale Factor)")
            plt.ylabel("Profile Log-Likelihood")
            plt.title(f"Profile Log-Likelihood for alpha (True d2/d1 = {d2_true/d1_true:.2f})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_filename = f"profile_loglik_d2factor_{d2_true_factor:.2f}.png"
            plt.savefig(plot_filename)
            print(f"Profile likelihood plot saved to: {plot_filename}")
            plt.close()

if __name__ == '__main__':
    main() 