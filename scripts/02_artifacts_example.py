import numpy as np
import matplotlib.pyplot as plt

# --- Parametri di base ---
fs = 500  # frequenza di campionamento Hz
t = np.linspace(0, 2, fs*2, endpoint=False)  # 2 secondi di dati

# --- Segnale EEG "pulito" ---
# somma di onde alfa (10 Hz) e theta (6 Hz)
sig_clean = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*6*t)

# --- Plot ---
plt.figure(figsize=(10,4))
plt.plot(t, sig_clean)
plt.xlabel("Tempo (s)")
plt.ylabel("Ampiezza (uV)")
plt.title("Segnale EEG simulato (pulito)")
plt.grid()
plt.show()

# --- Simula un artefatto da blink ---
blink = np.exp(-0.5 * ((t-1.0)/0.1)**2) * 2.0  # picco gaussiano centrato a 1s
sig_with_blink = sig_clean + blink

# --- Plot con blink ---
plt.figure(figsize=(10,4))
plt.plot(t, sig_with_blink, label="EEG con blink")
plt.plot(t, sig_clean, '--', alpha=0.6, label="EEG pulito")
plt.xlabel("Tempo (s)")
plt.ylabel("Ampiezza (uV)")
plt.title("Segnale EEG con artefatto da blink")
plt.legend()
plt.grid()
plt.show()

# --- Simula un artefatto muscolare ---
np.random.seed(42)  # per riproducibilità
muscle_noise = np.random.normal(0, 0.3, len(t))  # rumore bianco
# lo attiviamo solo tra 1.4s e 1.6s
mask = (t > 1.4) & (t < 1.6)
muscle_artifact = muscle_noise * mask

# Segnale con blink + muscolare
sig_with_artifacts = sig_with_blink + muscle_artifact

# --- Plot con entrambi gli artefatti ---
plt.figure(figsize=(10,4))
plt.plot(t, sig_with_artifacts, label="EEG con artefatti")
plt.plot(t, sig_clean, '--', alpha=0.6, label="EEG pulito")
plt.xlabel("Tempo (s)")
plt.ylabel("Ampiezza (uV)")
plt.title("Segnale EEG con artefatti (blink + muscolare)")
plt.legend()
plt.grid()
plt.savefig("../figures/artifacts_example.png")
plt.show()
