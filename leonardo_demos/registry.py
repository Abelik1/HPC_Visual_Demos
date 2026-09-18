from .demos.black_hole import BlackHoleDemo
from .demos.pbh import PBHDemo
from .demos.fluid import FluidDemo
from .demos.cosmic_web import CosmicWebDemo
from .demos.galaxy_collision import GalaxyCollisionDemo
from .demos.galaxy_collision_3d import GalaxyCollision3DDemo
from .demos.reaction_diffusion import ReactionDiffusionDemo
from .demos.crystal import CrystalDemo
from .demos.neural_wall import NeuralWallDemo
from .demos.fusion_plasma import FusionPlasmaDemo
from .demos.weather_ensemble import WeatherEnsembleDemo
from .demos.molecular_dynamics import MolecularDynamicsDemo
from .demos.neuro_racers import NeuroRacersDemo
from .demos.bat_vs_moth import BatVsMothDemo
from .demos.nbody_murb import NBodyMurbDemo

DEMOS={c.id:c for c in [
    BlackHoleDemo,PBHDemo,FluidDemo,CosmicWebDemo,GalaxyCollisionDemo,GalaxyCollision3DDemo,
    ReactionDiffusionDemo,CrystalDemo,NeuralWallDemo,FusionPlasmaDemo,
    WeatherEnsembleDemo,MolecularDynamicsDemo,NeuroRacersDemo,BatVsMothDemo,NBodyMurbDemo,
]}
