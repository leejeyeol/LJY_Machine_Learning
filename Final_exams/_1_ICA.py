import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy import signal

# #############################################################################
# Generate sample data
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])
B = np.array([[1, 3, 2], [2, 0.1, 1.0], [2.0, 1.5, 0.5]])
C = np.array([[1, 2, 0.5], [2, 0.5, 0.5], [0.5, 0.5, 2.0]])
# Mixing matrix

ica = FastICA(n_components=3)

XA = np.dot(S, A.T)  # Generate observations
SA_ = ica.fit_transform(XA)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

XB = np.dot(S, B.T)  # Generate observations
SB_ = ica.fit_transform(XB)  # Reconstruct signals
B_ = ica.mixing_  # Get estimated mixing matrix

XC = np.dot(S, C.T)  # Generate observations
SC_ = ica.fit_transform(XC)  # Reconstruct signals
C_ = ica.mixing_  # Get estimated mixing matrix

plt.figure()
models = [XA, XB, XC, S, SA_, SB_, SC_]
names = ['Observations(mixed signal by A)',
         'Observations(mixed signal by B)',
         'Observations(mixed signal by C)',
         'True Sources',
         'ICA recovered signals by A',
         'ICA recovered signals by B',
         'ICA recovered signals by C']
colors = ['red', 'steelblue', 'orange']

for ii, (model, name) in enumerate(zip(models, names),1):
    plt.subplot(7, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)
plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()

