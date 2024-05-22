import torch as t
from scipy.special import binom, gamma
from torch.distributions.dirichlet import Dirichlet
from scipy.stats import binomtest


def get_kl_component(q_i, p_i):
    assert 0 <= q_i <= 1 and 0 <= p_i <= 1
    if q_i == 0:
        return 0
    if p_i == 0:
        return t.inf
    return q_i * t.log(q_i / p_i)


def get_batch_kl_component(batch_q_i, batch_p_i):
    # [[q11, q12],  [[p11, p12],  ->  [[q11*log(q11/p11), q12*log(q12/p12)],
    #  [q21, q22]],  [p21, p22]]       [q21*log(q21/p21), q22*log(q22/p22)]]
    return batch_q_i * t.log(batch_q_i / batch_p_i)


def get_kl_scalars(q, p):
    return get_kl_component(q, p) + get_kl_component(1 - q, 1 - p)


def get_batch_kl_scalars(batch_q, batch_p):
    # [[q11, q12],  [[p11, p12],  ->  [[kl_scalars(q11, p11), kl_scalars(q12, p12)],
    #  [q21, q22]],  [p21, p22]]       [kl_scalars(q21, p21), kl_scalars(q22, p22]]
    return get_batch_kl_component(batch_q, batch_p) + get_batch_kl_component(1 - batch_q, 1 - batch_p)


def get_kl_vectors(q_vec, p_vec):
    return sum([get_kl_component(q_i, p_i) for q_i, p_i in zip(q_vec, p_vec)])


def get_batch_kl_vectors(batch_q_vec, batch_p_vec):
    # [q_vec_1,   [p_vec_1,   ->   [kl_vectors(q_vec_1, p_vec_1,
    #  q_vec_2],   p_vec_2]         kl_vectors(q_vec_2, p_vec_2)]
    return get_batch_kl_component(batch_q_vec, batch_p_vec).sum(dim=1)


def get_individual_kls(q_vec, p_vec):
    return t.tensor([get_kl_scalars(q_i, p_i) for q_i, p_i in zip(q_vec, p_vec)])


def get_batch_individual_kls(batch_q_vec, batch_p_vec):
    # [[q11, q12],  [[p11, p12],  ->  [[kl_scalars(q11, p11), kl_scalars(q12, p12)],
    #  [q21, q22]],  [p21, p22]]       [kl_scalars(q21, p21), kl_scalars(q22, p22]]
    return get_batch_kl_scalars(batch_q_vec, batch_p_vec)


def get_naive_union_bound_on_scalar_kl(m, M, delta, KL):
    return (KL + t.log(2 * t.sqrt(m) / (delta / M))) / m


def get_our_bound_on_vector_kl(m, M, delta, KL):
    summands = [binom(M, z) * (2 / m) ** (z/2) / gamma((M-z) / 2) for z in range(M)]
    log_term = - t.log(delta)
    log_term += 0.5 * t.log(t.tensor(t.pi))
    log_term += 1 / (12 * m)
    log_term += ((M - 1) / 2) * t.log(t.tensor(m / 2))
    log_term += t.log(t.tensor(sum(summands)))
    return (KL + log_term) / m


def volume_estimates(emp_risk, m, M, delta, KL, sample):
    # discrete_points = unif_simplex_sample(M, mc_samples)
    discrete_points = sample
    num_in_intersection = 0
    num_in_ours_only = 0
    num_in_theirs_only = 0
    num_in_neither = 0
    union_B = get_naive_union_bound_on_scalar_kl(m, M, delta, KL)
    our_B = get_our_bound_on_vector_kl(m, M, delta, KL)
    # print("union_B", union_B)
    # print("our_B", our_B)

    for p in discrete_points:
        union_flag = (get_individual_kls(emp_risk, p) < union_B).all()
        our_flag = get_kl_vectors(emp_risk, p) < our_B
        if union_flag and our_flag:
            num_in_intersection += 1
        elif union_flag and not our_flag:
            num_in_theirs_only += 1
        elif not union_flag and our_flag:
            num_in_ours_only += 1
        elif not union_flag and not our_flag:
            num_in_neither += 1
        else:
            print("Something has gone wrong")
    print(num_in_intersection, num_in_ours_only, num_in_theirs_only, num_in_neither)
    volume_intersection = num_in_intersection / mc_samples
    volume_ours_only = num_in_ours_only / mc_samples
    volume_theirs_only = num_in_theirs_only / mc_samples
    volume_neither = num_in_neither / mc_samples
    return volume_intersection, volume_ours_only, volume_theirs_only, volume_neither


# TODO: Check that the bounds are finite!
def get_batch_volume_estimates(emp_risk, m, M, delta_B, delta_CI, KL, sample):
    naive_B_scalar_kl = get_naive_union_bound_on_scalar_kl(m, M, delta_B, KL)
    print("Naive bound:", naive_B_scalar_kl)
    loose_naive_B_scalar_kl = get_naive_union_bound_on_scalar_kl(m, M, delta_B / 2, KL)
    our_B_vector_kl = get_our_bound_on_vector_kl(m, M, delta_B, KL)
    print("Our bound:", our_B_vector_kl)
    loose_our_B_vector_kl = get_our_bound_on_vector_kl(m, M, delta_B / 2, KL)

    individual_kls = get_batch_individual_kls(emp_risk.expand(mc_samples, -1), sample)
    kl_vectors = get_batch_kl_vectors(emp_risk.expand(mc_samples, -1), sample)

    naive_flags = (individual_kls < naive_B_scalar_kl).all(dim=1)
    ours_flags = kl_vectors < our_B_vector_kl
    loose_naive_flags = (individual_kls < loose_naive_B_scalar_kl).all(dim=1)
    loose_ours_flags = kl_vectors < loose_our_B_vector_kl
    loose_intersection_flags = t.logical_and(loose_naive_flags, loose_ours_flags)

    naive_is_subset_of_ours = (t.logical_or(t.logical_not(naive_flags), ours_flags)).all()
    loose_intersection_is_subset_of_naive = (t.logical_or(t.logical_not(loose_intersection_flags), naive_flags)).all()
    vol_naive_CI = confidence_interval(num_positive=naive_flags.int().sum(), num_trials=mc_samples, delta=delta_CI)
    vol_ours_CI = confidence_interval(num_positive=ours_flags.int().sum(), num_trials=mc_samples, delta=delta_CI)
    vol_loose_intersection_CI = confidence_interval(num_positive=loose_intersection_flags.int().sum(), num_trials=mc_samples, delta=delta_CI)

    return naive_is_subset_of_ours, loose_intersection_is_subset_of_naive, vol_naive_CI, vol_ours_CI, vol_loose_intersection_CI


def compare_volumes(emp_risk, m, M, delta_B, delta_CI, KL, sample):
    n_intersection, n_ours_only, n_naive_only, n_neither = batch_volume_estimates(emp_risk, m, M, delta_B, KL, sample)
    n_intersection_of_loosened_regions, _, _, _ = batch_volume_estimates(emp_risk, m, M, delta_B / 2, KL, sample)
    our_vol_CI = confidence_interval(
        num_positive=n_intersection + n_ours_only,
        num_trials=mc_samples,
        delta=delta_CI,
    )
    naive_vol_CI = confidence_interval(
        num_positive=n_intersection + n_naive_only,
        num_trials=mc_samples,
        delta=delta_CI,
    )
    hybrid_vol_CI = confidence_interval(
        num_positive=n_intersection_of_loosened_regions,
        num_trials=mc_samples,
        delta=delta_CI
    )


def unif_simplex_sample(M, num_samples):
    dist = Dirichlet(t.tensor([1.] * M))
    return dist.sample((num_samples,))


def confidence_interval(num_positive, num_trials, delta):
    result = binomtest(k=num_positive, n=num_trials, p=delta)
    low = result.proportion_ci().low
    high = result.proportion_ci().high
    return low, high


if __name__ == "__main__":
    # mc_samples = 10000000
    # sample = unif_simplex_sample(4, mc_samples)
    # 
    # volume_intersection, volume_ours_only, volume_theirs_only, volume_neither = batch_volume_estimates(
    #     t.tensor([0.25, 0.25, 0.25, 0.25]),
    #     m=t.tensor(100.),
    #     M=4, delta=t.tensor(0.05),
    #     KL=t.tensor(0.),
    #     sample=sample,
    # )
    
    # volume_ours = volume_intersection + volume_ours_only
    # volume_theirs = volume_intersection + volume_theirs_only

    delta_B = t.tensor(0.05)
    delta_CI = t.tensor(0.05)
    M_vals = t.tensor([3, 10, 100])
    m_vals = t.tensor([100, 300, 1000])
    mc_samples = 10000000

    # for delta in delta_vals:
    #     print("delta:", delta)
    #     for M in M_vals:
    #         print("\tM:", M)
    #         sample = unif_simplex_sample(M, mc_samples)
    #         emp_risk = t.tensor([1/M for _ in range(M)])
    #         for m in m_vals:
    #             if m < M:
    #                 continue
    #             print("\t\tm:", m)
    #             num_in_intersection, num_in_ours_only, num_in_theirs_only, num_in_neither = batch_volume_estimates(
    #                 emp_risk=emp_risk,  # TODO: Generalise this
    #                 m=m,
    #                 M=M,
    #                 delta=delta,
    #                 KL=t.tensor(0.),
    #                 sample=sample,
    #             )
    #             num_in_intersection, num_in_ours_only, num_in_theirs_only, num_in_neither = batch_volume_estimates(
    #                 emp_risk=emp_risk,  # TODO: Generalise this
    #                 m=m,
    #                 M=M,
    #                 delta=delta / 2,  # We ensure *both* bounds hold simultaneously WPAL 1 - delta
    #                 KL=t.tensor(0.),
    #                 sample=sample,
    #             )
    #             our_vol_CI = confidence_interval(
    #                 num_positive=num_in_intersection + num_in_ours_only,
    #                 num_trials=mc_samples,
    #                 delta=delta,
    #             )
    #             their_vol_CI = confidence_interval(
    #                 num_positive=num_in_intersection + num_in_theirs_only,
    #                 num_trials=mc_samples,
    #                 delta=delta,
    #             )
    #             print("\t\t\tOur CI:", our_vol_CI)
    #             print("\t\t\tTheir CI:", their_vol_CI)
    #             if our_vol_CI[1] < their_vol_CI[0]:
    #                 print("\t\t\tOurs is better")
    #             elif their_vol_CI[1] < our_vol_CI[0]:
    #                 print("\t\t\tTheirs is better")
    #             else:
    #                 print("\t\t\tEither could be better")
    
    for M in M_vals:
        sample = unif_simplex_sample(M, mc_samples)
        emp_risk = t.tensor([1/M for _ in range(M)])
        for m in m_vals:
            if m < M:
                continue
            print("M:", M)
            print("m:", m)
            naive_is_subset_of_ours, loose_intersection_is_subset_of_naive, vol_naive_CI, vol_ours_CI, vol_loose_intersection_CI = get_batch_volume_estimates(
                emp_risk=emp_risk,  # TODO: Generalise this
                m=m,
                M=M,
                delta_B=delta_B,
                delta_CI=delta_CI,
                KL=t.tensor(0.),
                sample=sample,
            )
            print("\tNaive⊆Ours:", naive_is_subset_of_ours)
            print("\tIntersection⊆Naive:", loose_intersection_is_subset_of_naive)
            print("\t", vol_naive_CI, "Naive vol")
            print("\t", vol_ours_CI, "Our vol")
            print("\t", vol_loose_intersection_CI, "Int vol")
            print()
