import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- Simula un segnale EEG fittizio ---
fs = 500  # frequenza di campionamento (Hz)
t = np.linspace(0, 2, fs*2, endpoint=False)  # 2 secondi di dati
# somma di 10 Hz (alpha) + 50 Hz (powerline noise)
sig = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*50*t)

# --- Calcola PSD prima del filtro ---
f, Pxx = signal.welch(sig, fs, nperseg=1024)

# --- Applica filtro notch a 50 Hz ---
b_notch, a_notch = signal.iirnotch(50, 30, fs)
sig_notch = signal.filtfilt(b_notch, a_notch, sig)

# --- Applica filtro low-pass (taglio 30 Hz) ---
b_lp, a_lp = signal.butter(4, 30, btype='low', fs=fs)
sig_lp = signal.filtfilt(b_lp, a_lp, sig)

# --- PSD dopo filtri ---
f_notch, Pxx_notch = signal.welch(sig_notch, fs, nperseg=1024)
f_lp, Pxx_lp = signal.welch(sig_lp, fs, nperseg=1024)

# --- Plot ---
plt.figure(figsize=(10,6))
plt.semilogy(f, Pxx, label="Originale")
plt.semilogy(f_notch, Pxx_notch, label="Dopo Notch 50 Hz")
plt.semilogy(f_lp, Pxx_lp, label="Dopo Low-pass 30 Hz")
plt.xlabel("Frequenza (Hz)")
plt.ylabel("PSD")
plt.legend()
plt.title("Effetto dei filtri su segnale simulato")
plt.grid()
plt.savefig("../figures/psd_filters.png")
plt.show()
