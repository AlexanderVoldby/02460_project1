#!/usr/bin/env fish

# Run the Python script with training parameters
set prior $argv[1]

if contains $prior Gaussian MoG Vamp
    printf "Training with %s prior\n" $prior
    python3 vae_bernoulli.py train --device cpu --latent-dim 10 --epochs 5 --batch-size 128 --model models/$prior\_prior.pt --samples pictures/$prior\_prior.png --prior $prior

    printf "\n\nTesting with %s prior\n" $prior
    python3 vae_bernoulli.py test --device cpu --latent-dim 10 --epochs 5 --batch-size 128 --model models/$prior\_prior.pt --samples pictures/$prior\_prior.png --prior $prior

    printf "\n\nSampling with %s prior\n" $prior
    python3 vae_bernoulli.py sample --device cpu --latent-dim 10 --epochs 5 --batch-size 128 --model models/$prior\_prior.pt --samples pictures/$prior\_prior_sample.png --prior $prior

    printf "\n\nPlotting with %s prior\n" $prior
    python3 vae_bernoulli.py plot_test --device cpu --latent-dim 10 --epochs 5 --batch-size 128 --model models/$prior\_prior.pt --samples pictures/$prior\_prior_latent_plot.png --prior $prior
end


