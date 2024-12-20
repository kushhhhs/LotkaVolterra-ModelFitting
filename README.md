# LotkaVolterra-ModelFitting
This repository contains a Python implementation to reverse engineer a predator-prey model.  

The goal is to estimate the parameters of the Lotka-Volterra model using noisy data generated from its equations. We use optimization algorithms, including both local and global optimization methods, to recover the underlying parameters.  

The data for this analysis is stored in [this file](./predator-prey-data.csv), which contains time-series observations of predator and prey populations with added Gaussian noise.  
pyt