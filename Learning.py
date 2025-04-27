from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

start_scope()

runtime = 500*ms

N_thalamus = 20
N_cortex = 40

tau = 10*ms
v_rest = -70*mV
v_reset = -70*mV
v_threshold = -54*mV
refractory_period = 5*ms

# STDP parametreleri
tau_pre = 20*ms
tau_post = 20*ms
A_pre = 0.01
A_post = -A_pre * tau_pre / tau_post * 1.05
w_max = 3.0*mV

# --- Talamik Girişler ---
pattern_neurons = [0, 1, 2, 3, 4]
pattern_times = np.tile([50, 150, 250, 350, 450], len(pattern_neurons)) * ms
pattern_indices = np.repeat(pattern_neurons, 5)
pattern_generator = SpikeGeneratorGroup(N_thalamus, pattern_indices, pattern_times)

noise_generator = PoissonGroup(N_thalamus, rates=5*Hz)

# --- Kortikal Nöronlar ---
eqs = '''
dv/dt = (v_rest - v) / tau : volt (unless refractory)
apre : 1
apost : 1
'''
cortex = NeuronGroup(N_cortex, eqs, threshold='v>v_threshold', reset='''v = v_reset
apost += A_post
''', refractory=refractory_period, method='exact')
cortex.v = v_rest

# --- Sinapslar ---

# Pattern -> Cortex (STDP ile)
syn_pattern = Synapses(pattern_generator, cortex,
                       '''
                       w : volt
                       ''',
                       on_pre='''
                       v_post += w
                       apre_post += A_pre
                       w = clip(w + apost_post*mV, 0*mV, w_max)
                       ''')
syn_pattern.connect(p=0.2)
syn_pattern.w = '0.5*mV + rand()*1.0*mV'

# Noise -> Cortex (Sabit ağırlıklı)
syn_noise = Synapses(noise_generator, cortex,
                     '''
                     w : volt
                     ''',
                     on_pre='v_post += w')
syn_noise.connect(p=0.2)
syn_noise.w = '0.5*mV'

# --- Monitörler ---
pattern_spike_mon = SpikeMonitor(pattern_generator)
noise_spike_mon = SpikeMonitor(noise_generator)
cortex_spike_mon = SpikeMonitor(cortex)
syn_weight_mon = StateMonitor(syn_pattern, 'w', record=range(5))

# --- Simülasyon ---
run(runtime)

# --- Grafikler ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Talamik girişler
axes[0].scatter(pattern_spike_mon.t/ms, pattern_spike_mon.i, s=5, color='blue', label='Pattern')
axes[0].scatter(noise_spike_mon.t/ms, noise_spike_mon.i+N_thalamus, s=5, color='gray', label='Noise')
axes[0].set_ylabel('Talamik Nöron #')
axes[0].set_title('Talamik Giriş Spikeleri (Pattern + Noise)')
axes[0].legend()

# Kortikal spike'lar
axes[1].scatter(cortex_spike_mon.t/ms, cortex_spike_mon.i, s=5, color='red')
axes[1].set_ylabel('Kortikal Nöron #')
axes[1].set_title('Kortikal Nöron Spikeleri')

# STDP Ağırlık değişimi
for i in range(5):
    axes[2].plot(syn_weight_mon.t/ms, syn_weight_mon.w[i]/mV, label=f'Sinaps {i}')
axes[2].set_ylabel('Ağırlık (mV)')
axes[2].set_xlabel('Zaman (ms)')
axes[2].set_title('Talamus-Korteks Sinaptik Ağırlık Değişimi (STDP)')
axes[2].legend()

plt.tight_layout()
plt.show()
