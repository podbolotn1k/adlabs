import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
from scipy.signal import filtfilt, iirfilter, butter, cheby1, bessel

start_params = {
    "amplitude": 1.0,
    "frequency": 0.3,
    "phase": 0.0,
    "noise_mean": 0.0,
    "noise_covariance": 0.1,
    "cutoff": 5.0,
    "filter_order": 5,
    "filter_type": "butterworth",
    "show_noise": True,
    "show_filtered": True
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

def apply_filter(signal, filter_type, cutoff, order, fs=100):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    if filter_type == "butterworth":
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
    elif filter_type == "chebyshev":
        b, a = cheby1(order, 1.0, normal_cutoff, btype='low', analog=False)
    elif filter_type == "bessel":
        b, a = bessel(order, normal_cutoff, btype='low', analog=False)
    elif filter_type == "elliptic":
        b, a = iirfilter(order, normal_cutoff, btype='low', ftype='ellip', 
                         rp=1, rs=60, analog=False)
    else:
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    return filtfilt(b, a, signal)

def calculate_error(original, filtered):
    return np.mean((original - filtered) ** 2)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
plt.subplots_adjust(left=0.25, bottom=0.35)

noisy_signal, clean_signal = harmonic_with_noise(t, **{k: start_params[k] for k in [
    "amplitude", "frequency", "phase", "noise_mean", "noise_covariance"
]}, regenerate_noise=True)

filtered_signal = apply_filter(
    noisy_signal, 
    start_params["filter_type"], 
    start_params["cutoff"], 
    start_params["filter_order"]
)

noisy_line, = ax1.plot(t, noisy_signal, color='orange', label="Noisy Signal")
clean_line, = ax1.plot(t, clean_signal, color='blue', linestyle='--', label="Clean Signal")
ax1.set_ylim(-2, 2)
ax1.set_title("Original and Noisy Signals")
ax1.legend()

filtered_line, = ax2.plot(t, filtered_signal, color='purple', linewidth=2, label="Filtered Signal")
comp_clean_line, = ax2.plot(t, clean_signal, color='blue', linestyle='--', label="Clean Signal")
ax2.set_ylim(-2, 2)
ax2.set_title("Filtered Signal Comparison")
ax2.legend()

error_text = ax2.text(0.02, 0.95, f"MSE Error: {calculate_error(clean_signal, filtered_signal):.5f}", 
                      transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))

axamp = plt.axes([0.25, 0.27, 0.65, 0.02])
axfreq = plt.axes([0.25, 0.24, 0.65, 0.02])
axphase = plt.axes([0.25, 0.21, 0.65, 0.02])
axnmean = plt.axes([0.25, 0.18, 0.65, 0.02])
axncov = plt.axes([0.25, 0.15, 0.65, 0.02])
axcutoff = plt.axes([0.25, 0.12, 0.65, 0.02])
axorder = plt.axes([0.25, 0.09, 0.65, 0.02])

samp = Slider(axamp, 'Amplitude', 0.1, 2.0, valinit=start_params["amplitude"])
sfreq = Slider(axfreq, 'Frequency', 0.01, 2.0, valinit=start_params["frequency"])
sphase = Slider(axphase, 'Phase', 0.0, 2 * np.pi, valinit=start_params["phase"])
snmean = Slider(axnmean, 'Noise Mean', -1.0, 1.0, valinit=start_params["noise_mean"])
sncov = Slider(axncov, 'Noise Covariance', 0.0, 1.0, valinit=start_params["noise_covariance"])
scutoff = Slider(axcutoff, 'Cutoff Frequency', 0.1, 10.0, valinit=start_params["cutoff"])
sorder = Slider(axorder, 'Filter Order', 1, 10, valinit=start_params["filter_order"], valstep=1)

filterax = plt.axes([0.03, 0.10, 0.15, 0.15])
filter_radio = RadioButtons(filterax, ('butterworth', 'chebyshev', 'bessel', 'elliptic'),
                            active=0 if start_params["filter_type"]=="butterworth" else 1)

checkax1 = plt.axes([0.03, 0.28, 0.15, 0.05])
check_noise = CheckButtons(checkax1, ['Show Noise'], [start_params["show_noise"]])

checkax2 = plt.axes([0.03, 0.23, 0.15, 0.05])
check_filtered = CheckButtons(checkax2, ['Show Filtered'], [start_params["show_filtered"]])

resetax = plt.axes([0.03, 0.04, 0.15, 0.04])
button = Button(resetax, 'Reset')

def update(val=None):
    global stored_noise
    params = {
        "amplitude": samp.val,
        "frequency": sfreq.val,
        "phase": sphase.val,
        "noise_mean": snmean.val,
        "noise_covariance": sncov.val,
    }
    cutoff = scutoff.val
    order = int(sorder.val)
    filter_type = filter_radio.value_selected
    show_noise = check_noise.get_status()[0]
    show_filtered = check_filtered.get_status()[0]

    regenerate_noise = update.prev_noise_mean != params["noise_mean"] or \
                       update.prev_noise_covariance != params["noise_covariance"]

    noisy_signal, clean_signal = harmonic_with_noise(t, **params, regenerate_noise=regenerate_noise)
    filtered_signal = apply_filter(noisy_signal, filter_type, cutoff, order)
    
    error = calculate_error(clean_signal, filtered_signal)
    error_text.set_text(f"MSE Error: {error:.5f}")

    noisy_line.set_ydata(noisy_signal if show_noise else np.full_like(t, np.nan))
    clean_line.set_ydata(clean_signal)
    filtered_line.set_ydata(filtered_signal if show_filtered else np.full_like(t, np.nan))
    comp_clean_line.set_ydata(clean_signal)

    update.prev_noise_mean = params["noise_mean"]
    update.prev_noise_covariance = params["noise_covariance"]
    fig.canvas.draw_idle()

update.prev_noise_mean = start_params["noise_mean"]
update.prev_noise_covariance = start_params["noise_covariance"]

for slider in [samp, sfreq, sphase, snmean, sncov, scutoff, sorder]:
    slider.on_changed(update)
check_noise.on_clicked(update)
check_filtered.on_clicked(update)
filter_radio.on_clicked(update)

def reset(event):
    global stored_noise
    stored_noise = None
    samp.reset()
    sfreq.reset()
    sphase.reset()
    snmean.reset()
    sncov.reset()
    scutoff.reset()
    sorder.reset()
    check_noise.set_active(0)
    check_filtered.set_active(0)
    filter_radio.set_active(0)
    update()
    
button.on_clicked(reset)
plt.show()