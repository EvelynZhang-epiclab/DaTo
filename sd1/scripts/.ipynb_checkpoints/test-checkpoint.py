import inspect
import os
import subprocess
import numpy as np
from pytorch_fid import fid_score
import logging
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from datetime import datetime
from txt2img_search_new import txt2img_gen
import re
import math
import shutil
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.mutation.pm import PM
import random
from pymoo.core.sampling import Sampling
import numpy as np
import glob
import ImageReward as RM
import torch

def print_class_methods(cls):
    methods = [func for func in dir(cls) if callable(getattr(cls, func))]
    for method in methods:
        print(method)

# Example usage:
print_class_methods(Problem)