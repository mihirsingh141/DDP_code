# 🌀 Symmetry Discovery Framework for Physical Systems

This repository contains the codebase used for discovering and exploiting symmetries in physical systems using **LieGAN** and **LaLiGAN**. The framework supports multiple physical systems such as spring-mass, pendulum, and two-body dynamics. Discovered symmetries are used to improve the learning of downstream models including HNNs, LSTMs, MLPs, and EMLPs.

---

### 📌 Step 1: Generate Data and Discover Symmetries

To generate datasets and discover symmetry generators using the LieGAN architecture, run the following command:

```bash
python main_lagan.py --num_epochs 100 --task "{task_name}"


Replace {task_name} with the physical system of interest, such as:

spring_mass

pendulum

two_body

etc.

This will:

Simulate or load datasets for the specified task.

Train the LieGAN generator and discriminator.

Discover Lie algebra symmetry generators.

Save the generated data and trained models to appropriate folders.

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

### 📁 Directory Structure
graphql
Copy
Edit
final_ddp_code/
│
├── main_lagan.py            # Main file for LieGAN training and dataset generation
├── train.py                 # Training loop for LaLiGAN-based models
├── utils.py                 # Utility functions
├── req.txt                  # Required Python dependencies
│
├── Notebooks/               # Jupyter notebooks for evaluation
│
├── Data/                    # Generated datasets stored as .pkl files
│
├── saved_model/             # Trained model checkpoints
│   ├── LaLiGAN/             # Models trained using LaLiGAN
│   │   ├── EMLP/            # EMLP-based LaLiGAN models
│   │   └── vanilla/         # Vanilla models trained with LaLiGAN
│   └── LieGAN/              # LieGAN generators and discriminators
│
└── src/                     # Source code for model definitions
    ├── MLP/                 # MLP models (vanilla and equivariant)
    ├── LSTM/                # LSTM models
    ├── HNN/                 # Hamiltonian Neural Networks
    └── EMLP/                # Equivariant MLP architectures

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

bash
Copy
Edit
pip install -r req.txt
To reproduce results:

Generate data and discover symmetries:

bash
Copy
Edit
python main_lagan.py --num_epochs 100 --task "{task_name}"
Run any of the result notebooks:

Examples:

Notebooks/spring_mass_laligan.ipynb

Notebooks/pendulum_liegan.ipynb

Use the saved models and datasets in the Data/ and saved_model/ folders.

📜 License
This repository is part of an academic thesis and is currently licensed for research use only. For commercial or public use, please contact the author.

👤 Author
Mihir Singh
Department of Data Science
Indian Institute of Technology Madras

yaml
Copy
Edit

---

✅ Let me know if you’d like to add:
- A LaTeX-formatted citation block
- A `requirements.txt` preview
- Example results or figures in Markdown

I'm happy to format those too.
