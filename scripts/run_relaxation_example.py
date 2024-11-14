# 
# Download a pre-trained OCP model checkpoint.
# Set up the OCP calculator using the downloaded checkpoint.
# Create a Cu(100) surface with a CH3O adsorbate using ASE.
# Fix the bottom two layers of the slab to mimic bulk behavior.
# Set the OCP calculator for the adslab system.
# Run the relaxation using the LBFGS optimizer for a maximum of 100 steps or until the maximum force on any atom is below 0.03 eV/Å1
# The relaxation trajectory will be saved in the "data" directory as "Cu100_CH3O_relaxation.traj". You can analyze this trajectory using ASE tools or visualize it using software like ASE's GUI or OVITO.

# get the checkpoint
import os
import wget

checkpoint_url = "https://dl.fbaipublicfiles.com/opencatalystproject/models/2022_09/oc22/s2ef/gnoc_oc22_oc20_all_s2ef.pt"
checkpoint_path = "gnoc_oc22_oc20_all_s2ef.pt"

if not os.path.exists(checkpoint_path):
    wget.download(checkpoint_url)


# setup OCP calculator 
from ocpmodels.common.relaxation.ase_utils import OCPCalculator

ocp_calculator = OCPCalculator(
    # config_yml="equiformer_v2/config/oc20.yaml",
    checkpoint=checkpoint_path,
)

# Create an adslab system using ASE. For this example, we'll use a Cu(100) surface with a CH3O adsorbate:

from ase.build import fcc100, molecule
from ase.constraints import FixAtoms

# Create the Cu(100) surface
adslab = fcc100("Cu", size=(3, 3, 3))

# Add the CH3O adsorbate
adsorbate = molecule("CH3O")
adslab.extend(adsorbate)

# Fix the bottom two layers of the slab
constraint = FixAtoms(indices=[atom.index for atom in adslab if atom.position[2] < 7])
adslab.set_constraint(constraint)

# Set the calculator
adslab.set_calculator(ocp_calculator)

print("cell", adslab.cell)


# Run the relaxation
from ase.optimize import LBFGS
from ase.io.trajectory import Trajectory

# Create a directory to store the trajectory
os.makedirs("data", exist_ok=True)

# Set up the optimizer
dyn = LBFGS(adslab, trajectory="data/Cu100_CH3O_relaxation.traj")

# Run the relaxation
dyn.run(fmax=0.03, steps=100)
