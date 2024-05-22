from morvant_bound import *


def test__get_spectral_norm__2x2():
    A = t.arange(4).float().reshape(2, 2)
    result = get_spectral_norm(A)
    answer = t.sqrt(3 * t.sqrt(t.tensor(5.)) + 7)
    assert t.isclose(result, answer)


def test__get_spectral_norm__3x3():
    A = t.arange(9).float().reshape(3, 3)
    result = get_spectral_norm(A)
    answer = t.sqrt(12 * t.sqrt(t.tensor(70.)) + 102)
    assert t.isclose(result, answer)


def test__get_batch_spectral_norm__3x2x2():
    As = t.arange(4).float().reshape(2, 2).expand(3, 2, 2)
    result = get_batch_spectral_norm(As)
    answer = (t.sqrt(3 * t.sqrt(t.tensor(5.)) + 7)).expand(3)
    assert t.allclose(result, answer)


def test__get_batch_spectral_norm__4x3x3():
    As = t.arange(9).float().reshape(3, 3).expand(4, 3, 3)
    result = get_batch_spectral_norm(As)
    answer = t.sqrt(12 * t.sqrt(t.tensor(70.)) + 102).expand(4)
    assert t.allclose(result, answer)


def test__get_morvant_C__2x2():
    our_C = t.tensor([[0.2, 0.3], [0.4, 0.1]])
    result = get_morvant_C(our_C)
    answer = t.tensor([[0., 0.6], [0.8, 0.0]])
    assert t.allclose(result, answer)


def test__get_morvant_C__3x3():
    our_C = t.tensor([[0.1, 0.05, 0.05], [0.25, 0.1, 0.15], [0.01, 0.02, 0.27]])
    result = get_morvant_C(our_C)
    answer = t.tensor([[0., 0.25, 0.25], [0.5, 0., 0.3], [1/30, 1/15, 0.]])
    assert t.allclose(result, answer)


def test__get_batch_morvant_C__3x2x2():
    our_Cs = t.tensor([[0.2, 0.3], [0.4, 0.1]]).expand(3, 2, 2)
    result = get_batch_morvant_C(our_Cs)
    answer = t.tensor([[0., 0.6], [0.8, 0.0]]).expand(3, 2, 2)
    assert t.allclose(result, answer)


def test__get_batch_morvant_C__5x3x3():
    our_Cs = t.tensor([[0.1, 0.05, 0.05], [0.25, 0.1, 0.15], [0.01, 0.02, 0.27]]).expand(5, 3, 3)
    result = get_batch_morvant_C(our_Cs)
    answer = t.tensor([[0., 0.25, 0.25], [0.5, 0., 0.3], [1/30, 1/15, 0.]]).expand(5, 3, 3)
    assert t.allclose(result, answer)


def test__get_m_minus__4x4():
    empirical_error_counts = t.arange(16).reshape((4, 4))
    m = empirical_error_counts.sum()
    answer = t.tensor(6.)
    our_C = empirical_error_counts / m
    result = get_m_minus(our_C, m)
    assert t.allclose(result, answer)


def test__get_batch_m_minus__10x4x4():
    empirical_error_counts = t.arange(16).reshape((4, 4)).expand(10, 4, 4)
    ms = empirical_error_counts.sum(-1).sum(-1)
    answer = t.tensor(6.).expand(10)
    our_Cs = empirical_error_counts / ms.reshape(10, 1, 1)
    result = get_batch_m_minus(our_Cs, ms)
    assert t.allclose(result, answer)


def test__get_sample_our_Cs():
    num_classes = 4
    num_mc_samples = 100
    result = get_sample_our_Cs(num_classes, num_mc_samples)
    assert result.shape == (num_mc_samples, num_classes, num_classes)
    assert t.allclose(result.sum(-1).sum(-1), t.tensor(1., device="cuda"))


