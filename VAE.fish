#!/usr/bin/env fish

# Run the Python script with training parameters
if test $argv[1] = "Gaussian"
    printf 'Training with Gaussian prior'
    python3 vae_bernoulli.py train --device cpu --latent-dim 10 --epochs 5 --batch-size 128 --model models/Gaussian_prior.pt --samples pictures/gaussian_prior.png --prior Gaussian;
    printf "\n\n"
    printf "Testing with Gaussian prior"
    python3 vae_bernoulli.py test --device cpu --latent-dim 10 --epochs 5 --batch-size 128 --model models/Gaussian_prior.pt --samples pictures/gaussian_prior.png --prior Gaussian;
    printf "\n\n"
    printf "Sampling with Gaussian prior"
    python3 vae_bernoulli.py sample --device cpu --latent-dim 10 --epochs 5 --batch-size 128 --model models/Gaussian_prior.pt --samples pictures/gaussian_prior_sample.png --prior Gaussian;
    printf "\n\n"
    printf "Plotting with Gaussian prior"
    python3 vae_bernoulli.py plot_test --device cpu --latent-dim 10 --epochs 5 --batch-size 128 --model models/Gaussian_prior.pt --samples pictures/gaussian_prior_latent_plot.png --prior Gaussian;
end
if test $argv[1] = "MoG"
    printf 'Training with MoG prior'
    python3 vae_bernoulli.py train --device cpu --latent-dim 10 --epochs 5 --batch-size 128 --model models/MoG_prior.pt --samples pictures/MoG_prior.png --prior MoG;
    printf "\n\n"
    printf "Testing with MoG prior"
    python3 vae_bernoulli.py test --device cpu --latent-dim 10 --epochs 5 --batch-size 128 --model models/MoG_prior.pt --samples pictures/MoG_prior.png --prior MoG;
    printf "\n\n"
    printf "Sampling with MoG prior"
    python3 vae_bernoulli.py sample --device cpu --latent-dim 10 --epochs 5 --batch-size 128 --model models/MoG_prior.pt --samples pictures/MoG_prior_sample.png --prior MoG;
    printf "\n\n"
    printf "Plotting with MoG prior"
    python3 vae_bernoulli.py plot_test --device cpu --latent-dim 10 --epochs 5 --batch-size 128 --model models/MoG_prior.pt --samples pictures/MoG_prior_latent_plot.png --prior MoG;
end
