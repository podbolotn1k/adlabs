"""
Інструкція:
- Слайдери зліва внизу дозволяють змінювати амплітуду, частоту, фазу, шум і фільтр.
- Чекбокс "Show Noise" вмикає або вимикає відображення шуму.
- Кнопка "Reset" повертає всі параметри до початкових значень.
- Синій пунктир — оригінальний сигнал, помаранчевий — зашумлений, фіолетовий — фільтрований.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.signal import butter, filtfilt

start_params = {
    "amplitude": 1.0,
    "frequency": 0.3,
    "phase": 0.0,
    "noise_mean": 0.0,
    "noise_covariance": 0.1,
    "cutoff": 5.0,
    "show_noise": True
}

t = np.linspace(0, 10, 1000)
stored_noise = None 

def harmonic_with_noise(t, amplitude, frequency, phase, noise_mean, noise_covariance, regenerate_noise=False):
    global stored_noise
    clean_signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    if regenerate_noise or stored_noise is None:
        noise = np.random.normal(noise_mean, np.sqrt(noise_covariance), size=t.shape)
        stored_noise = noise
    else:
        noise = stored_noise
    return clean_signal + noise, clean_signal

def lowpass_filter(signal, cutoff, fs=100, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.45)
noisy_signal, clean_signal = harmonic_with_noise(t, **{k: start_params[k] for k in [
    "amplitude", "frequency", "phase", "noise_mean", "noise_covariance"]}, regenerate_noise=True)
filtered_signal = lowpass_filter(noisy_signal, start_params["cutoff"])

noisy_line, = ax.plot(t, noisy_signal, color='orange', label="Noisy Signal")
clean_line, = ax.plot(t, clean_signal, color='blue', linestyle='--', label="Clean Signal")
filtered_line, = ax.plot(t, filtered_signal, color='purple', linewidth=2, label="Filtered Signal")
ax.set_ylim(-2, 2)
ax.legend()

axamp = plt.axes([0.25, 0.37, 0.65, 0.03])
axfreq = plt.axes([0.25, 0.33, 0.65, 0.03])
axphase = plt.axes([0.25, 0.29, 0.65, 0.03])
axnmean = plt.axes([0.25, 0.25, 0.65, 0.03])
axncov = plt.axes([0.25, 0.21, 0.65, 0.03])
axcutoff = plt.axes([0.25, 0.17, 0.65, 0.03])

samp = Slider(axamp, 'Amplitude', 0.1, 2.0, valinit=start_params["amplitude"])
sfreq = Slider(axfreq, 'Frequency', 0.01, 2.0, valinit=start_params["frequency"])
sphase = Slider(axphase, 'Phase', 0.0, 2 * np.pi, valinit=start_params["phase"])
snmean = Slider(axnmean, 'Noise Mean', -1.0, 1.0, valinit=start_params["noise_mean"])
sncov = Slider(axncov, 'Noise Covariance', 0.0, 1.0, valinit=start_params["noise_covariance"])
scutoff = Slider(axcutoff, 'Cutoff Frequency', 0.1, 10.0, valinit=start_params["cutoff"])

resetax = plt.axes([0.25, 0.05, 0.1, 0.04])
button = Button(resetax, 'Reset')

checkax = plt.axes([0.75, 0.05, 0.15, 0.1])
check = CheckButtons(checkax, ['Show Noise'], [start_params["show_noise"]])

def update(val):
    global stored_noise
    params = {
        "amplitude": samp.val,
        "frequency": sfreq.val,
        "phase": sphase.val,
        "noise_mean": snmean.val,
        "noise_covariance": sncov.val,
    }
    cutoff = scutoff.val
    show_noise = check.get_status()[0]

    regenerate_noise = update.prev_noise_mean != params["noise_mean"] or \
                       update.prev_noise_covariance != params["noise_covariance"]

    noisy_signal, clean_signal = harmonic_with_noise(t, **params, regenerate_noise=regenerate_noise)
    filtered_signal = lowpass_filter(noisy_signal, cutoff)

    noisy_line.set_ydata(noisy_signal if show_noise else np.full_like(t, np.nan))
    clean_line.set_ydata(clean_signal)
    filtered_line.set_ydata(filtered_signal)

    update.prev_noise_mean = params["noise_mean"]
    update.prev_noise_covariance = params["noise_covariance"]
    fig.canvas.draw_idle()

update.prev_noise_mean = start_params["noise_mean"]
update.prev_noise_covariance = start_params["noise_covariance"]

for slider in [samp, sfreq, sphase, snmean, sncov, scutoff]:
    slider.on_changed(update)
check.on_clicked(update)

def reset(event):
    global stored_noise
    stored_noise = None
    samp.reset()
    sfreq.reset()
    sphase.reset()
    snmean.reset()
    sncov.reset()
    scutoff.reset()
    check.set_active(0)  
button.on_clicked(reset)

plt.show()
