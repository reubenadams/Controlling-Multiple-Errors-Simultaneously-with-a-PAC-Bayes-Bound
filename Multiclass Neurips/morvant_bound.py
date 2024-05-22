# TODO: Fix this Morvant implementation which is clearly wrong.

import torch as t
from volume_comparison_large_m import unif_simplex_sample, confidence_interval


def get_spectral_norm(matrix):
    singular_values = t.linalg.svdvals(matrix)
    return singular_values.max()


def get_batch_spectral_norm(matrices):
    singular_values = t.linalg.svdvals(matrices)
    return singular_values.max(dim=1)[0]


def get_morvant_bound(emp_morvant_C, num_classes, m_minus, KL, delta):
    spectral_norm = get_spectral_norm(emp_morvant_C)
    print("Spectral norm:", spectral_norm)
    fraction = 8 * num_classes / (m_minus - 8 * num_classes)
    print("Fraction:", fraction)
    bracket = KL + t.log(m_minus / (4 * delta))
    print("Bracket:", bracket)
    return spectral_norm + t.sqrt(fraction * bracket)


def get_morvant_C(our_C):
    """Takes in a confusion matrix that sums to one, normalises the rows and zeros the diagonal"""
    normed_C = our_C / t.sum(our_C, dim=1, keepdim=True)
    t.diagonal(normed_C).zero_()
    return normed_C


def get_batch_morvant_C(our_Cs):
    normed_Cs = our_Cs / t.sum(our_Cs, dim=2, keepdim=True)
    t.diagonal(normed_Cs, dim1=1, dim2=2).zero_()
    return normed_Cs


def get_m_minus(our_C, m):
    num_each_class = t.sum(our_C, dim=1) * m  # TODO: I think this should be dim=1!
    return t.min(num_each_class)


def get_batch_m_minus(our_Cs, ms):
    batch_size = our_Cs.shape[0]
    num_each_class = t.sum(our_Cs, dim=2) * ms.reshape(batch_size, -1)
    return t.min(num_each_class, dim=1)[0]


def get_sample_our_Cs(num_classes, num_mc_samples):
    M = num_classes ** 2
    flattened_Cs = unif_simplex_sample(M, num_mc_samples)
    return flattened_Cs.reshape(num_mc_samples, num_classes, num_classes)


def get_num_in_morvant_vol(empirical_our_C, m_minus, num_classes, delta_B, KL, sample_true_our_Cs):
    empirical_morvant_C = get_morvant_C(empirical_our_C)
    morvant_bound = get_morvant_bound(empirical_morvant_C, num_classes, m_minus, KL, delta_B)
    print("m_minus:", m_minus, "Morvant bound:", morvant_bound)
    sample_true_morvant_Cs = get_batch_morvant_C(sample_true_our_Cs)
    sample_spectral_norms = get_batch_spectral_norm(sample_true_morvant_Cs)
    return (sample_spectral_norms < morvant_bound).int().sum()


def get_vol_morvant_estimate(empirical_our_C, m, num_classes, delta_B, KL, num_mc_samples):
    sample_true_our_Cs = get_sample_our_Cs(num_classes, num_mc_samples)
    num_in_morvant_vol = get_num_in_morvant_vol(empirical_our_C, m, num_classes, delta_B, KL, sample_true_our_Cs)
    return num_in_morvant_vol / num_mc_samples


def get_batch_vol_morvant_estimate(empirical_our_C, m, num_classes, delta_B, delta_CI, KL, batch_size, num_batches):
    total_sample_size = batch_size * num_batches
    total_in_vol_morvant = 0
    m_minus = get_m_minus(empirical_our_C, m)
    for batch_num in range(num_batches):
        if batch_num % 100 == 0:
            print(f"{batch_num + 1} out of {num_batches}")
        sample_true_our_Cs = get_sample_our_Cs(num_classes, batch_size)
        num_in_vol_morvant = get_num_in_morvant_vol(empirical_our_C, m_minus, num_classes, delta_B, KL, sample_true_our_Cs)
        total_in_vol_morvant += num_in_vol_morvant
    vol_morvant_CI = confidence_interval(num_positive=total_in_vol_morvant, num_trials=total_sample_size, delta=delta_CI)
    print(f"Num in morvant vol: {total_in_vol_morvant}/{total_sample_size}")
    return vol_morvant_CI


#-1. Pick num_classes, empirical_C [[q11, q12], [q21, q22]], M, KL, delta
# 0. Set M = num_classes ** 2, calculate m_minus, and calculate the Morvant bound

# 1. Sample in [[p11, p12], [p21, p22]] 
# 2. Convert to Morvant confusion matrices
# 3. Calculate their spectral norms
# 4. Check if these are less than the morvant bound

# TODO: It looks like m has to be horrendously big before the volume is appreciably less than 1? This is great news!
# m = 1000
# m_vals = [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000]
# m_vals = [100, 300, 1000]
m_vals = [8000]
# num_classes = 10
num_classes_vals = [2, 5, 10]
# num_classes_vals = [2]

delta_B = t.tensor(0.05)
delta_CI = t.tensor(0.05)
KL = t.tensor(0.)
# num_mc_samples = 1000
batch_size = 100000
num_batches = 1

# print(get_vol_morvant_estimate(empirical_our_C, m, num_classes, delta_B, KL, num_mc_samples))

for num_classes in num_classes_vals:
    for m in m_vals:
        print("\t\t\tnum_classes:", num_classes, "M:", num_classes ** 2, "m:", m)
        empirical_our_C = t.ones((num_classes, num_classes)) / num_classes ** 2
        print("\t\t\t", get_batch_vol_morvant_estimate(empirical_our_C, m, num_classes, delta_B, delta_CI, KL, batch_size, num_batches))
        print("\t\t\tnum_classes:", num_classes, "M:", num_classes ** 2, "m:", m)
        print()


