# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from flow_prior import FlowPrior

PI = torch.from_numpy(np.asarray(np.pi))

def reparameterizeMoG(mu, stds, weights):
    batch_size, K, M = mu.shape

    component = td.Categorical(probs=weights).sample()  # Shape: (batch_size,)

    # Gather the corresponding means and stds
    mu_selected = torch.gather(mu, 1, component[:, None, None].expand(-1, 1, M)).squeeze(1)
    stds_selected = torch.gather(stds, 1, component[:, None, None].expand(-1, 1, M)).squeeze(1)

    # Reparameterization trick
    eps = torch.randn((batch_size, M))
    z = mu_selected + stds_selected * eps  # Shape: (batch_size, M)

    return z

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        
        # Standard Gaussian Prior
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.logvar = nn.Parameter(torch.zeros(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.
        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=torch.sqrt(torch.exp(self.logvar))), 1)

class MoGPrior(nn.Module):
    def __init__(self, M, K):
        """
        Define a mixture of Gaussians prior distribution.
        Parameters:
        M: [int] 
           Dimension of the latent space.
        K: [int]
           Number of components in the mixture.
        """
        super(MoGPrior, self).__init__()
        self.M = M # dim of latent called L in book
        self.K = K
        self.means = nn.Parameter(torch.randn(self.K, self.M), requires_grad=True)
        self.logvars = nn.Parameter(torch.zeros(self.K, self.M), requires_grad=True)
        self.weights = nn.Parameter((torch.ones(self.K)/ self.K), requires_grad=True)

    def forward(self):
        weights = F.softmax(self.weights, dim=-1)
        means = torch.clamp(self.means, -4, 4)
        stds = torch.clamp(torch.exp(self.logvars*0.5), -4, 4)

        return td.MixtureSameFamily(td.Categorical(probs=weights),
                                td.Independent(td.Normal(loc=means, scale=stds), 1))

class VampPrior(nn.Module):
    def __init__(self, M, n_pseudo_p, encoder_net):
        """
        Define a VampPrior distribution.
        Parameters:
        M: [int] 
           Dimension of the latent space.
        K: [int]
           Number of pseudo inputs.
        encoder_net: [torch.nn.Module]
              The encoder network that takes as a tensor of dim `B, D' and output a tensor 2M values for the mean and log variance.
        """
        super(VampPrior, self).__init__()
        self.M = M
        self.n_pseudo_points = n_pseudo_p
        self.encoder_net = encoder_net
        
        # Make parameters
        u = torch.randn(self.n_pseudo_points, 784)
        self.u = nn.Parameter(u, requires_grad=True)

        # Initialize the weights
        self.weights = nn.Parameter((torch.ones(self.n_pseudo_points)/self.n_pseudo_points), requires_grad=True)

    def get_params(self):
        vamp_mean, vamp_log_var = torch.chunk(self.encoder_net(self.u), 2, dim=-1)
        return vamp_mean, vamp_log_var

    def forward(self):
        vamp_mean, vamp_log_var = torch.chunk(self.encoder_net(self.u), 2, dim=-1)
        w = F.softmax(self.weights+0.001, dim=-1)
        return td.MixtureSameFamily(td.Categorical(probs=w), td.Independent(td.Normal(loc=vamp_mean, scale=torch.exp(vamp_log_var*0.5)), 1))
 

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.
        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, logvar = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.sqrt(torch.exp(logvar))), 1)

class MoGEncoder(nn.Module):
    def __init__(self, encoder_net, M, K):
        """
        Define a Gaussian encoder distribution based on a given encoder network.
        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(MoGEncoder, self).__init__()
        self.M = M
        self.K = K
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        means, logvars, weights = torch.split(self.encoder_net(x), [self.M * self.K, self.M * self.K, self.K], dim=-1)
        means = torch.clamp(means.view(-1, self.K, self.M), -5, 5)
        logvars = torch.clamp(logvars.view(-1, self.K, self.M), -5, 5)
        weights = F.softmax(weights+0.001, dim=-1)
        return means, logvars, weights 

class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.logvars = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2) #td.Independent return a distribution from where one can sample.


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, decoder, encoder, prior_name):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder
        self.prior_name = prior_name

    def elbo(self, x, beta=1.0):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x) # q is a distribution, from the td.Independent class, where one can sample from the distribution.
        
        if self.prior_name == "MoG":
            # clamp the values of the logvars
            mu, log_var, w = q[0], q[1], q[2]
            z = reparameterizeMoG(mu, torch.exp(log_var*0.5), w)
            elbo_re = torch.mean(self.decoder(z).log_prob(x), dim=0)
            # Define mixture components
            base_distribution = td.Independent(td.Normal(loc=mu, scale=torch.exp(log_var*0.5)), 1)
            mixture = td.Categorical(probs=w)
            # Define mixture model
            mog = td.MixtureSameFamily(mixture, base_distribution)
            ln_q_z_x = mog.log_prob(z)
            ln_p_z = self.prior().log_prob(z)
            elbo_kl = torch.mean(ln_q_z_x - ln_p_z, dim=0) 
            # use a warmup term of stability
            elbo = elbo_re - beta* elbo_kl
        else:
            z = q.rsample()
            mu, std = q.mean, q.variance.sqrt()
            gaussian = td.Independent(td.Normal(loc=mu, scale=std), 1)
            ln_q_z_x = gaussian.log_prob(z)
            ln_p_z = self.prior().log_prob(z)
            elbo_kl = torch.mean(ln_q_z_x - ln_p_z, dim=0)
            elbo_re = torch.mean(self.decoder(z).log_prob(x), dim=0) 
            elbo = elbo_re - elbo_kl
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x, beta):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x, beta)

def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            if isinstance(model.prior, MoGPrior):
                beta = min(1.0, (epoch+0.1)/(epochs+0.1))
                loss = model(x, beta)
            else:
                loss = model(x, 1.0)
            if isinstance(model.prior, VampPrior):
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

def test(model, data_loader, device):
    data_iter = iter(data_loader)
    loss = 0
    for x in data_iter:
        x = x[0].to(device)
        loss += model(x, 1.0)
    return loss/len(data_loader)

def plot_approx_posterior(model, data_loader, device, M, data_points, figure_name):
    data_iter = iter(data_loader)
    #label_list = []
    z_list = []
    with torch.no_grad():
        #for x, label in data_iter:
        for x, _ in data_iter:
            x = x.to(device)
    #        label_list.append(label)
            q = model.encoder(x)
            z = None
            if isinstance(model.prior, (GaussianPrior, VampPrior)):
                z = q.sample()
            elif isinstance(model.prior, MoGPrior):
                z = reparameterizeMoG(q[0], torch.sqrt(torch.exp(q[1])), q[2]) 
            z_list.append(z)
        z = torch.cat(z_list, dim=0).numpy()
        #labels = torch.cat(label_list, dim=0).numpy()
        if (M > 2):
            pca = PCA(n_components=2)
            z = pca.fit_transform(z)
        if (M == 1):
            z = np.concatenate(z, np.zeros(z.shape[0], 1))
        if len(z) > data_points:
            z = z[0:data_points,:]
            #labels = labels[0:data_points]
        plt.figure(figsize=(8, 6))
        # Plot prior
        n_points = 1000
        x = np.linspace(min(z[:,0]), max(z[:,0]), n_points)
        y = np.linspace(min(z[:,1]), max(z[:,1]), n_points)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack((X.ravel(), Y.ravel()))
        if (M > 2):
            Z = np.exp(model.prior().log_prob(pca.inverse_transform(torch.from_numpy(grid_points)).to(device)).numpy())
        else:
            Z = np.exp(model.prior().log_prob(torch.from_numpy(grid_points).to(device)).numpy())
        Z = Z.reshape(n_points, n_points)
        if isinstance(model.prior, GaussianPrior):
            plt.title("Test Set in Latent Space and contour Gaussian prior")
        elif isinstance(model.prior, MoGPrior):
            plt.title("Test Set in Latent Space and contour MoG prior")
        #plt.contour(X, Y, Z, levels=15, colors='black', alpha=0.7)
        plt.contourf(X, Y, Z, levels=15, cmap='viridis', alpha=0.7)
        #scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='jet', s=8, alpha=0.7)
        plt.scatter(z[:, 0], z[:, 1], color='black', s=5, alpha=0.8)
        #cbar = plt.colorbar(scatter, label="Classes", orientation="vertical")
        #cbar.set_ticks(np.unique(labels))  # Set color bar ticks to class labels
        #cbar.set_label("Class Label") 
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.savefig('pictures/contour_' + figure_name + '.png')
     

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'test', 'plot_test'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='models/Gaussian_prior.pt', choices=['models/Gaussian_prior.pt', 'models/MoG_prior.pt', 'models/Flow_prior.pt', 'models/Vamp_prior.pt'], help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='Gaussian', choices=['Gaussian', 'MoG', 'Flow', 'Vamp'], help='prior distribution to use (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)

    # Define prior distribution
    M = args.latent_dim
    K = 5
    final_layer = nn.Linear(512, 2*M*K + K) if args.prior == "MoG" else nn.Linear(512, 2*M)
    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 512),
        nn.LeakyReLU(),
        final_layer
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )


    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)

    if args.prior == 'Gaussian':
        prior = GaussianPrior(M)
        encoder = GaussianEncoder(encoder_net)

    elif args.prior == 'MoG':
        prior = MoGPrior(M, K=K)
        encoder = MoGEncoder(encoder_net, M, K=K)

    elif args.prior == 'Vamp':
        n_pseudo_p = 5
        prior = VampPrior(M, n_pseudo_p, encoder_net=encoder_net)
        encoder = GaussianEncoder(encoder_net)
    
    elif args.prior == 'Flow':
        prior = FlowPrior(M, num_transforms=4, hidden_dim=64)
        encoder = GaussianEncoder(encoder_net)  

    model = VAE(prior, decoder, encoder, args.prior).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            save_image(samples.view(64, 1, 28, 28), args.samples)

    elif args.mode == 'test':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            Elbo_loss = test(model, mnist_test_loader, args.device)
            print(Elbo_loss)

    elif args.mode == 'plot_test':
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Plot test set in latent space
        plot_approx_posterior(model, mnist_test_loader, args.device, M, 2500, args.prior) # 2500 points

