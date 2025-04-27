from brian2 import *
import matplotlib.pyplot as plt

start_scope()

# Simülasyon süresi
duration = 500*ms

# Talamik nöron parametreleri
N_thalamus = 20

# Kortikal nöron parametreleri
N_cortex = 80

# Model: LIF
eqs = '''
dv/dt = (-(v - v_rest))/tau : volt (unless refractory)
'''

# Parametreler
v_rest = -70*mV
v_reset = -65*mV
v_thresh = -50*mV
tau = 10*ms
refractory_period = 5*ms

# Talamik grup - Poisson spike generator
thalamus = PoissonGroup(N_thalamus, rates=30*Hz)  # Burada spike oranını artırdım

# Kortikal grup
cortex = NeuronGroup(
    N_cortex,
    eqs,
    threshold='v>v_thresh',
    reset='v = v_reset',
    refractory=refractory_period,
    method='exact'
)

# Başlangıç voltajını biraz daha yukarı başlatalım
cortex.v = 'v_rest + (v_thresh - v_rest) * 0.6 * rand()'

# Sinapslar
syn_thalamus_cortex = Synapses(thalamus, cortex, on_pre='v_post += 4.0*mV')  # Burada ağırlığı artırdım
syn_thalamus_cortex.connect(p=0.3)  # %30 bağlantı

syn_cortex = Synapses(cortex, cortex, on_pre='v_post += 1.5*mV')
syn_cortex.connect(condition='i!=j', p=0.1)

# Monitörler
spike_mon_thalamus = SpikeMonitor(thalamus)
spike_mon_cortex = SpikeMonitor(cortex)
state_mon_cortex = StateMonitor(cortex, 'v', record=True)

# Simülasyonu çalıştır
run(duration)

# Grafikler
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.title('Talamik Giriş Spikeleri')
plt.plot(spike_mon_thalamus.t/ms, spike_mon_thalamus.i, '.k')
plt.xlabel('Zaman (ms)')
plt.ylabel('Talamik Nöron #')

plt.subplot(2, 1, 2)
plt.title('Kortikal Nöron Spikeleri')
plt.plot(spike_mon_cortex.t/ms, spike_mon_cortex.i, '.r')
plt.xlabel('Zaman (ms)')
plt.ylabel('Kortikal Nöron #')

plt.tight_layout()
plt.show()
