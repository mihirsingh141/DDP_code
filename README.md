# 🌀 Symmetry Discovery Framework for Physical Systems

This repository contains the codebase used for discovering and exploiting symmetries in physical systems using **LieGAN** and **LaLiGAN**. The framework supports multiple physical systems such as spring-mass, pendulum, and two-body dynamics. Discovered symmetries are used to improve the learning of downstream models including HNNs, LSTMs, MLPs, and EMLPs.

---

### 📌 Step 1: Generate Data and Discover Symmetries

To generate datasets and discover symmetry generators using the LieGAN architecture, run the following command:

```bash
python main_lagan.py --num_epochs 100 --task "{task_name}"
```


Replace {task_name} with the physical system of interest, such as:

spring_mass

pendulum

two_body

etc.

This will:

Simulate or load datasets for the specified task.

Train the LieGAN generator and discriminator.

Discover Lie algebra symmetry generators.

Save the generated data and trained models to the appropriate folders.

### 📌 Step 2: Run Result Notebooks
After generating data and training the LieGAN models, you can run the analysis and result notebooks independently.

These notebooks are located in the Notebooks/ folder. Each notebook corresponds to a specific experiment and performs visualization, evaluation, and interpretation of results using the generated data and saved models.

For example:

spring_mass_laligan.ipynb

pendulum_liegan.ipynb

To run:

Open a notebook of your choice.

Execute all cells (ensure dependencies are installed).

Visual outputs and evaluation metrics will be shown inline.


### 🧠 Model Definitions
The src/ directory contains implementations of the following models:

MLP/: Standard fully connected networks

LSTM/: Recurrent neural networks for time-series modeling

HNN/: Hamiltonian Neural Networks for conservative systems

EMLP/: Equivariant MLPs designed to respect physical symmetries

Each folder may include:

Vanilla versions of the model

Symmetry-integrated variants using discovered group actions

### 💾 Outputs
Data/: Contains generated datasets as .pkl files named by task

saved_model/: Stores model checkpoints, organized by:

Symmetry method:

LieGAN/

LaLiGAN/

Model architecture:

EMLP/

vanilla/

Task name:

spring_mass/

pendulum/

two_body/

etc.

Example filenames inside these folders:

Copy
Edit
laligan_generator_499.pt
laligan_discriminator_499.pt
autoencoder_499.pt
regressor_499.pt

♻️ Reproducibility
Install dependencies:

```bash
pip install -r req.txt
```

To reproduce results:

Generate data and discover symmetries:

```bash
python main_lagan.py --num_epochs 100 --task "{task_name}"
```

Run any of the result notebooks:

Examples:

Notebooks/spring_mass_laligan.ipynb

Notebooks/pendulum_liegan.ipynb

Use the saved models and datasets in the Data/ and saved_model/ folders.


👤 Author
Mihir Singh
Indian Institute of Technology Madras

