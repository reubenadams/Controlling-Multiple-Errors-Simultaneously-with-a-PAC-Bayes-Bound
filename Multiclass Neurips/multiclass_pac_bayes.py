# TODO: When loading the weights, get the batch size from the master config
# TODO: Freeze the prior before stoch training
# TODO: Make sure the test function respects the stoch argument, and also puts the stoch flag back!
# TODO: Sort out the idea of having multiple lambdas, one for each layer
# TODO: Add learning rate scheduler as in Dziugaite et al.
# TODO: Need model.train() and model.eval() somewhere?

from pac_bayes.data_utils import *
from model_utils import *
from test_train_utils import *
import wandb
from dataclasses import asdict
import torch.nn as nn
from empirical_risk_vectors import *


if __name__ == "__main__":
    toy = True
    wandb.login()
    wandb.init(
        project="Dz Reproduction",
        name="Multiclass, S=3000, l2=3, lr=0.01, GSamps=100, ytol=eps, scale=-3, 1000stoch_epochs, proper_tot_risk",
    )

    with t.autograd.set_detect_anomaly(True):
        t.manual_seed(0)
        device = "cuda" if t.cuda.is_available() else "cpu"
        config = MasterConfig()
        config.stoch_mlp_config.d_out = 2  # TODO: Generalise to number of labels
        config.stoch_mlp_config.stoch_lin_configs[-1].d_out = 2
        config.stoch_train_config.lr = 0.01

        if toy:
            config.data_config.subset_size = 3000
            config.det_train_config.epochs = 50
            config.stoch_train_config.epochs = 1000
            config.stoch_train_config.num_grad_samples = 100

            for lin_config in config.stoch_mlp_config.stoch_lin_configs:
                lin_config.scale = -3.0

            config.stoch_train_config.prior_scale = -3.0
            config.stoch_train_config.lr = 0.0001
            config.stoch_train_config.reset_noise = False

        train_data, test_data = load_data(label_map_zero_one)
        datasets = split_data(
            train_data,
            config.data_config.prior_prop,
            config.data_config.val_prop,
            config.data_config.cert_prop,
        )

        if config.data_config.subset_size:
            datasets = take_subsets(datasets, config.data_config.subset_size)

        prior_loader, val_loader, cert_loader, post_loader = make_dataloaders(
            datasets, config.data_config.batch_size
        )

        print("Prior samples:", len(prior_loader.dataset))
        print("Val samples:", len(val_loader.dataset))
        print("Cert samples:", len(cert_loader.dataset))
        print("Post samples:", len(post_loader.dataset))

        n_cert = len(cert_loader.dataset)
        wandb.config = asdict(config)

        model_P, model_Q = create_models(config.stoch_mlp_config, device)
        model_P.softmax_in_forward = True
        model_Q.softmax_in_forward = True
        prior_scale = nn.Parameter(t.tensor(config.stoch_train_config.prior_scale))

        loss_fn = nn.CrossEntropyLoss()
        det_train(
            model=model_P,
            train_loader=prior_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            config=config.det_train_config,
            device=device,
        )
        t.save(model_P.state_dict(), "model_P_multiclass")

        model_P.load_state_dict(t.load("model_P_multiclass"))
        model_Q.load_mean(model_P)
        print("KL after initialisation:", model_Q.kl_divergence(model_P, prior_scale))
        # model_Q.set_scale_from_mean()
        print("Stoch Train")
        stoch_train_multiclass(
            post_model=model_Q,
            prior_model=model_P,
            prior_scale=prior_scale,
            post_loader=post_loader,
            cert_loader=cert_loader,
            config=config.stoch_train_config,
            error_matrix=error_matrix,
            M=M,
            loss_vec=loss_vec,
            n_cert=n_cert,
            bound_mc_samples=config.final_bound_config.bound_mc_samples,
            delta_prime=config.final_bound_config.delta_prime,
            device=device,
        )
    wandb.finish()
