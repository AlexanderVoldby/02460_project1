import numpy as np

# Given sample values
samples_Gaussian = [94.08, 92.30, 92.80, 94.22, 94.43, 92.03, 93.29, 92.03, 94.55, 94.40]
samples_MoG = [91.75, 91.57, 91.22, 90.81, 90.96, 91.86, 90.58, 91.52, 91.78, 91.09]
samples_Vamp = [90.74, 90.03, 91.07, 90.42, 90.29, 90.84, 90.90, 91.85, 90.81, 90.68]

# Function to compute mean and standard deviation
def compute_stats(samples, name):
    mean_value = np.mean(samples)
    std_dev = np.std(samples, ddof=0)  # ddof=0 for population std, ddof=1 for sample std
    print(f"For the {name} prior, the mean and standard deviation of the test error is:")
    print(f"Mean: {mean_value:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}\n")

# Compute stats for each prior
compute_stats(samples_Gaussian, "Gaussian")
compute_stats(samples_MoG, "MoG")
compute_stats(samples_Vamp, "Vamp")
