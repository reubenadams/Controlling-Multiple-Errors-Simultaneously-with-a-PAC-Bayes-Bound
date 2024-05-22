import nltk.corpus.reader

import wandb
from pac_bayes.configs import *
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import math
from pac_bayes.kl_utils import *
from simple_stoch_models import StochMLP, var
from empirical_risk_vectors import *
from scipy.special import binom, gamma
from inverse_kl_derivatives import get_data


def det_evaluate(model, train_loader, test_loader, loss_fn, device):
    train_loss = test(model, train_loader, loss_fn, device, stoch=False)
    train_error = test(model, train_loader, num_incorrect, device, stoch=False)
    test_loss = test(model, test_loader, loss_fn, device, stoch=False)
    test_error = test(model, test_loader, num_incorrect, device, stoch=False)
    return train_loss, train_error, test_loss, test_error


# Rename this to prior?
def det_train_log(model, train_loader, val_loader, loss_fn, epoch, device):
    train_loss, train_error, val_loss, val_error = det_evaluate(
        model, train_loader, val_loader, loss_fn, device
    )
    wandb.log(
        {
            "Det_train_loss": train_loss,
            "Det_train_error": train_error,
            "Det_val_loss": val_loss,
            "Det_val_error": val_error,
            "Epoch": epoch,
        }
    )


# TODO: Add an early stopping criterion for when the loss on the test_loader starts increasing
def det_train(model, train_loader, val_loader, loss_fn, config: DetTrainConfig, device):
    # model.all_det()
    optimizer = t.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    det_train_log(model, train_loader, val_loader, loss_fn, epoch=0, device=device)

    for epoch in range(1, config.epochs + 1):
        print(f"Epoch: {epoch}/{str(config.epochs)}")
        for batch, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            wandb.log({"Det_train_batch_loss": loss, "Batch": batch})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        det_train_log(model, train_loader, val_loader, loss_fn, epoch=epoch, device=device)


# TODO: Shouldn't this take the cert_loader, too?
def stoch_train(
    post_model,
    prior_model,
    prior_scale,
    post_loader,
    loss_fn,
    config: StochTrainConfig,
    n_cert,
    bound_mc_samples,
    delta_prime,
    device,
):
    """Trains the posterior and the scale of the prior to minimize the PAC-Bayes bound"""
    post_model.train()
    prior_model.eval()
    # learnable_parameters = list(post_model.parameters())
    learnable_parameters = list(post_model.parameters()) + [prior_scale]
    # optimizer = t.optim.SGD(
    #     learnable_parameters, lr=config.lr, weight_decay=config.weight_decay
    # )
    optimizer = t.optim.SGD(learnable_parameters, lr=config.lr)
    for epoch in range(1, config.epochs + 1):
        print(f"Epoch: {epoch}/{str(config.epochs)}")
        post_model.all_stoch()  # Necessary because the test function keeps setting it to all det?

        for batch, (x, y) in enumerate(post_loader):
            x = x.to(device)
            y = y.to(device)
            stoch_loss = post_model.mc_loss(x, y, loss_fn, config.num_grad_samples, reset_noise=config.reset_noise) / x.size(0)
            kl = post_model.kl_divergence(prior_model, prior_scale)
            kl_bound = pb_kl_bound(
                kl,
                var(prior_scale),
                n_cert,
                config.b,
                config.c,
                config.delta,
            )
            sqrt_term = t.sqrt(kl_bound / 2)
            # training_obj = pb_training_obj(stoch_loss, kl_bound)
            # training_obj = stoch_loss
            # training_obj = kl
            # training_obj = kl_bound
            training_obj = stoch_loss + sqrt_term
            wandb.log(
                {
                    "Stoch_Loss": stoch_loss,
                    "KL": kl,
                    "PB_kl_bound": kl_bound,
                    "Sqrt_term": sqrt_term,
                    "PB_objective": training_obj,
                    "Prior_scale": prior_scale.data,
                    "Post_scale": post_model.layer1.weight_scale.mean(),
                    "Batch": batch,
                }
            )

            optimizer.zero_grad()
            training_obj.backward()
            t.nn.utils.clip_grad_norm_(learnable_parameters, max_norm=0.0001)
            optimizer.step()

        kl = post_model.kl_divergence(prior_model, prior_scale)
        # TODO: A problem here occurs if the variance has become infinite!
        kl_bound = pb_kl_bound(
            kl, var(prior_scale), n_cert, config.b, config.c, config.delta
        )
        final_bound = pb_final_bound(
            kl_bound,
            post_model,
            post_loader.dataset,  # TODO: Shouldn't we be passing the cert_loader, too?
            num_incorrect,
            bound_mc_samples,
            delta_prime,
            device
        )
        wandb.log(
            {
                "KL": kl,
                "Full_PB_kl_bound": kl_bound,
                "Full_PB_error_bound": final_bound,
                "Epoch": epoch,
            }
        )







# TODO: Add final q_hat bound
def stoch_train_multiclass(
    post_model,
    prior_model,
    prior_scale,
    post_loader,
    cert_loader,
    config: StochTrainConfig,
    error_matrix,
    M,
    loss_vec,
    n_cert,
    bound_mc_samples,
    delta_prime,
    device,
):
    """Trains the posterior and the scale of the prior to minimize the PAC-Bayes bound"""
    post_model.train()
    prior_model.eval()
    learnable_parameters = list(post_model.parameters()) + [prior_scale]
    optimizer = t.optim.SGD(learnable_parameters, lr=config.lr)
    for epoch in range(1, config.epochs + 1):
        print(f"Epoch: {epoch}/{str(config.epochs)}")
        post_model.all_stoch()  # Necessary because the test function keeps setting it to all det?

        for batch, (x, y) in enumerate(post_loader):
            x = x.to(device)
            y = y.to(device)

            pred_label_probs = post_model(x)
            empirical_risk = empirical_error_counts(pred_label_probs, y, error_matrix, M) / x.size(0)

            kl = post_model.kl_divergence(prior_model, prior_scale)
            kl_bound = pb_kl_bound_multiclass(kl, var(prior_scale), n_cert, config.b, config.c, config.delta, M)  # TODO: M should probably be in the config?

            with t.no_grad():
                _, u_derivs, c_deriv = get_data(loss_vec=loss_vec, u_vec=empirical_risk.clone().detach(), c=kl_bound.clone().detach())

            # print("empirical risk:", empirical_risk)
            # print("u_derivs:", u_derivs)
            # print("kl_bound:", kl_bound)
            # print("kl_bound_deriv:", c_deriv)
            training_obj = (empirical_risk * t.tensor(u_derivs)).sum() + kl_bound * t.tensor(c_deriv)
            # print("training_obj:", training_obj)

            wandb.log(
                {
                    "empirical_risk_0": empirical_risk[0],
                    "empirical_risk_1": empirical_risk[1],
                    "empirical_risk_2": empirical_risk[2],
                    "KL": kl,
                    "PB_kl_bound": kl_bound,
                    "u_deriv_0": u_derivs[0],
                    "u_deriv_1": u_derivs[1],
                    "u_deriv_2": u_derivs[2],
                    "B_deriv": c_deriv,
                    "PB_objective": training_obj,
                    "Prior_scale": prior_scale.data,
                    "Post_scale": post_model.layer1.weight_scale.mean(),
                    "Batch": batch,
                }
            )

            optimizer.zero_grad()
            training_obj.backward()
            # t.nn.utils.clip_grad_norm_(learnable_parameters, max_norm=0.0001)
            optimizer.step()

        with t.no_grad():
            kl = post_model.kl_divergence(prior_model, prior_scale)
            kl_bound = pb_kl_bound_multiclass(kl, var(prior_scale), n_cert, config.b, config.c, config.delta, M)

            empirical_risk = t.zeros(M)
            for x, y in cert_loader:
                x = x.to(device)
                y = y.to(device)
                pred_label_probs = post_model(x)
                empirical_risk += empirical_error_counts(pred_label_probs, y, error_matrix, M)
            empirical_risk = empirical_risk / len(cert_loader.dataset)

            v_star, _, _ = get_data(loss_vec=loss_vec, u_vec=empirical_risk, c=kl_bound)
            # total_risk_bound = kl_vectors(empirical_risk, v_star)
            total_risk_bound = (t.tensor(loss_vec) * t.tensor(v_star)).sum()
        wandb.log(
            {
                "$\\text{KL}(Q||P)$": kl,
                "$\\text{kl}(R_S(Q)||R_D(Q))$ bound": kl_bound,
                "$R_S(Q)_0$": empirical_risk[0],
                "$R_S(Q)_1$": empirical_risk[1],
                "$R_S(Q)_2$": empirical_risk[2],
                "$R_D^TF(Q)$ bound": total_risk_bound,
                "Epoch": epoch
            }
        )


def test(model, dataloader, loss_fn, device, stoch=False, mean=True):
    if isinstance(model, StochMLP):
        if stoch:
            model.all_stoch()
        else:
            model.all_det()
    total_loss = t.tensor(0.0, device=device)
    with t.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            total_loss += loss_fn(y_hat, y)  # TODO: I think you need to scale this up by the batch size y.size(0)?
    if mean:
        total_loss = total_loss / len(dataloader.dataset)
    return total_loss.detach().cpu()


def logistic_loss(y_hat, y):
    """Returns the *sum* of the logistic losses of each prediction"""
    y_hat = t.reshape(y_hat, y.shape)
    return F.softplus(-y_hat * y).sum() / t.log(t.tensor(2.0))


# First return is for when y_hat is a 1D model output for binary classification according to dz et al.
# Second return is for >1D output
def num_incorrect(y_hat, y):
    # return (t.reshape(t.sign(y_hat), y.shape) != y).sum()
    _, predictions = t.max(y_hat, -1)
    return (predictions != y).sum()


def monte_carlo_test(model, dataset, loss_fn, num_samples, device):
    sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
    # batch_size has to be 1 so that new weights are drawn for every sample
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)
    return test(model, dataloader, loss_fn, device=device, stoch=True, mean=False) / num_samples


# If prior_var is large, then t.log(c / prior_var) can be very small, meaning its log is negative.
def pb_kl_bound(kl, prior_var, n_cert, b, c, delta):
    log_term = t.log(t.tensor(t.pi) ** 2 * n_cert / (6 * delta))
    log_log_term = 2 * t.log(b * t.log(c / prior_var))
    if kl < 0 or log_log_term < 0:
        print("kl:", kl)
        print("log term:", t.log(b * t.log(c / prior_var)))
    numerator = kl + log_log_term + log_term
    return numerator / (n_cert - 1)


def pb_kl_bound_multiclass(kl, prior_var, n_cert, b, c, delta, M):
    delta_lambda = 6 * delta / (t.pi * b * t.log(c / prior_var))**2
    summands = [binom(M, z) * (2 / n_cert) ** (z/2) / gamma((M-z) / 2) for z in range(M)]
    log_term = - t.log(delta_lambda)
    log_term += 0.5 * t.log(t.tensor(t.pi))
    log_term += 1 / (12 * n_cert)
    log_term += ((M - 1) / 2) * t.log(t.tensor(n_cert / 2))
    log_term += t.log(t.tensor(sum(summands)))
    return (kl + log_term) / n_cert


def pb_training_obj(batch_loss, kl_bound):
    return batch_loss + t.sqrt(kl_bound / 2)


def rounded_lamb(lamb, c, b):
    j = -t.log(lamb / c) * b
    return math.floor(j), math.ceil(j)


def pb_final_bound(kl_bound, model, dataset, loss_fn, num_mc_samples, delta_prime, device):
    with t.no_grad():
        mc_empirical_loss = monte_carlo_test(model, dataset, loss_fn, num_mc_samples, device)
        empirical_loss_bound = kl_scalars_inverse(
            mc_empirical_loss.item(), math.log(2 / delta_prime) / num_mc_samples
        )
        true_loss_upper_bound = kl_scalars_inverse(
            empirical_loss_bound, kl_bound.item()
        )
        return true_loss_upper_bound


# def pb_final_bound(kl_bound, mc_empirical_loss, num_mc_samples, delta_prime):
#     empirical_loss_bound = kl_scalars_inverse(mc_empirical_loss, t.log(2 / delta_prime) / num_mc_samples)
#     true_loss_upper_bound = kl_scalars_inverse(empirical_loss_bound, kl_bound)
#     return true_loss_upper_bound
