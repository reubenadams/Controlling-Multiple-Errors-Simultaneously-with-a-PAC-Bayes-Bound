import torch as t


def empirical_error_counts(pred_label_probs, true_labels, error_matrix, M):
    # print("pred_label_probs")
    # print(pred_label_probs)
    # print("true_labels")
    # print(true_labels)
    # print("error_matrix")
    # print(error_matrix)
    e_counts = t.empty(M)
    error_types = error_matrix[true_labels]
    # print("error_types")
    # print(error_types)
    for j in range(M):
        mask = error_types == j
        # print("mask")
        # print(mask)
        e_counts[j] = (pred_label_probs[mask]).sum(0)
    return e_counts


def construct_error_matrix(labels, error_types):
    assert labels == list(range(len(labels)))
    n = len(labels)
    error_type_matrix = t.empty((n, n))
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            for k, error_type_k in enumerate(error_types):
                if (pred_label, true_label) in error_type_k:
                    error_type_matrix[i, j] = k
                    break
            else:
                print("Cannot find error type")
    return error_type_matrix


labels = [0, 1]
E0 = {(0, 0), (1, 1)}
E1 = {(0, 1)}
E2 = {(1, 0)}
error_types = [E0, E1, E2]
loss_vec = [0, 1, 3]
M = len(error_types)
error_matrix = construct_error_matrix(labels, error_types)


if __name__ == "__main__":
    pred_label_probs = t.tensor([[0.1, 0.9], [0.4, 0.6], [0.5, 0.5]])
    true_labels = t.tensor([1, 0, 1])
    e_counts = empirical_error_counts(pred_label_probs, true_labels, error_matrix, M=3)
    print(e_counts)
