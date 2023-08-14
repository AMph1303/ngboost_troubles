from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
import torch
import numpy as np
import matplotlib.pyplot as plt
from botorch.acquisition import ExpectedImprovement, PosteriorMean, ProbabilityOfImprovement
from botorch.optim import optimize_acqf
from ngboost import NGBRegressor

from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model

def func(X):
    return -np.sin(3*X) - X**2 + 0.7*X


class CustomModelGP(ExactGP, GPyTorchModel):
    num_outputs = 1
    def __init__(self, train_x, train_y, likelihood = GaussianLikelihood()):
        super(CustomModelGP, self).__init__(train_x, train_y.flatten(), likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        mvn = MultivariateNormal(mean_x, covar_x)
        print("mean shape:", mvn.mean.shape, "\ncovar shape:", mvn.covariance_matrix.shape)
        print("mean:", mvn.mean, "\ncovar:", mvn.covariance_matrix)
        return mvn

class CustomModel(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y, n_estimators=70):
        # squeeze output dim before passing train_Y to ExactGP
        super().__init__(train_X, train_Y, GaussianLikelihood())
        self.ngboost = NGBRegressor(n_estimators=n_estimators).fit(train_X, train_Y.flatten())
        self.to(train_X)  # make sure we're on the right device/dtype

    def predict_ngboost(self, x):
        if x.requires_grad:
            x = x.detach()
        Y = self.ngboost.predict(x)
        Y_dists = self.ngboost.pred_dist(x)
        mean_y = torch.Tensor(Y_dists[:].params['loc'])
        var_y = Y_dists[:].params['scale']
        covar_y = covariation_matrix(Y, mean_y)
        np.fill_diagonal(covar_y, var_y)
        covar_y = torch.Tensor(covar_y)
        return mean_y, covar_y

    def forward(self, x):
        if x.ndim < 3:
            mean_y, covar_y = self.predict_ngboost(x)
            return MultivariateNormal(mean_y.requires_grad_(), covar_y.requires_grad_())
        else:
            means = []
            covars = []
            for i, x_i in enumerate(x):
                mean_y, covar_y = self.predict_ngboost(x_i)
                means.append(mean_y[np.newaxis, :])
                covars.append(covar_y[np.newaxis, :, :])
            means = torch.cat(means, dim=0).requires_grad_()
            covars = torch.cat(covars, dim=0).requires_grad_()
            mvn = MultivariateNormal(means, covars)
            print(mvn.mean.shape, mvn.covariance_matrix.shape)
            print(mvn.mean.grad_fn, mvn.covariance_matrix.grad_fn,)
            print(mvn.mean, mvn.covariance_matrix)
            return mvn


def covariation_matrix(X, loc_X):
    """
    returns calculations of cov(x,y) = 1/(n-1) sum_i (x_i−mean(x))(y_i−mean(y))
    """
    cov_matrix = np.zeros((len(X), len(X)))
    for i, x in enumerate(X):
        for j in range(i + 1, len(X)):
            cov_matrix[j, i] = cov_matrix[i, j] = (x - loc_X[i]) * (X[j] - loc_X[j])

    return cov_matrix


def plot_suggessions(acquisition, train_X, n_iter=10):
    plt.figure(figsize=(12, n_iter * 3))
    plt.subplots_adjust(hspace=0.4)

    X = torch.arange(-2.0, 2.0, 0.01).reshape(-1, 1)
    Y = func(X)
    label_name = '*'

    assert acquisition in [ProbabilityOfImprovement, PosteriorMean,
                           ExpectedImprovement], 'unexpected acquisition functon'

    if acquisition == ProbabilityOfImprovement:
        label_name = 'Probability of Improvement'
    elif acquisition == PosteriorMean:
        label_name = 'Posterior Mean'
    else:
        label_name = 'Expected Improvement'

    for i in range(n_iter):
        # Getting values of target function in train points
        train_Y = func(train_X)
        # Defining the model
        model = CustomModel(train_X, train_Y)
        #
        # Defining acquisition function
        if acquisition == PosteriorMean:
            aq = acquisition(model)
        else:
            aq = acquisition(model, best_f=train_Y.max())

        # Domain of the target function.
        data_dim = train_X.shape[-1]
        bounds = torch.stack([torch.ones(data_dim) * (-3), torch.ones(data_dim) * (3)])
        # Botorch suggestion
        candidate, acq_value = optimize_acqf(
            aq, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
        )
        train_X = torch.cat((train_X, candidate), 0)
        # Plot everything
        plt.subplot(n_iter, 2, 2 * i + 1)
        plt.plot(X.squeeze(), Y.squeeze(), 'y--', lw=1, label='Real func')
        plt.scatter(train_X.squeeze(), func(train_X).squeeze(), label='Samples', alpha=0.5)
        plt.title(f'Iteration {i + 1}')
        plt.ylim(-6, 2)

        y_mean, y_std = model.posterior(X).mean.flatten().detach().numpy(), model.posterior(
            X).variance.flatten().detach().numpy()
        plt.plot(X.squeeze(), y_mean, color="black", label="Mean")
        plt.fill_between(
            X.squeeze(),
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.1,
            color="black",
            label=r"$\pm$ 1 std. dev.",
        )
        if i == 0:
            plt.legend()

        plt.subplot(n_iter, 2, 2 * i + 2)
        plt.plot(X.squeeze(), aq(X.flatten()[:, None, None]).detach().numpy(),
                 'r-', lw=1, label=f'Acquisition function\n{label_name}')
        plt.title(f'Iteration {i + 1}')
        if i == 0:
            plt.legend()
    plt.show()


def main():
    train_x = torch.Tensor([1.4, -0.95]).reshape(-1, 1)
    train_y = func(train_x).flatten()

    # --- Uncomment to run ngboost model ---
    model = CustomModel(train_x, train_y)
    # --- end ---

    # --- Uncomment to run gp model ---
    #model = CustomModelGP(train_x, train_y)
    #mll = ExactMarginalLogLikelihood(model.likelihood, model)
    #fit_gpytorch_model(mll)
    # --- end ---

    # Defining acquisition function
    aq = ExpectedImprovement(model, best_f=train_y.max())

    # Domain of the target function.
    data_dim = train_x.shape[-1]
    bounds = torch.stack([torch.zeros(data_dim), torch.ones(data_dim)])
    # Botorch suggestion
    candidate, acq_value = optimize_acqf(
        aq, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
    )
    train_x = torch.cat((train_x, candidate), 0)

if __name__ == '__main__':
    main()

