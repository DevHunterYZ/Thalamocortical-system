from brian2 import *
import matplotlib.pyplot as plt

start_scope()

# Simülasyon süresi
duration = 500*ms

# Nöron sayıları
N_thalamus = 20
N_cortex = 80

# Nöron modeli
eqs = '''
dv/dt = (-(v - v_rest))/tau : volt (unless refractory)
'''

# Parametreler
v_rest = -70*mV
v_reset = -65*mV
v_thresh = -50*mV
tau = 10*ms
refractory_period = 5*ms

# Talamik girişler
thalamus = PoissonGroup(N_thalamus, rates=30*Hz)

# Kortikal nöronlar
cortex = NeuronGroup(
    N_cortex,
    eqs,
    threshold='v>v_thresh',
    reset='v = v_reset',
    refractory=refractory_period,
    method='exact'
)
cortex.v = 'v_rest + (v_thresh - v_rest) * 0.6 * rand()'

# Talamus → Korteks sinapsları
syn_thalamus_cortex = Synapses(thalamus, cortex,
    '''
    w : volt
    dApre/dt = -Apre/tau_stdp : volt (event-driven)
    dApost/dt = -Apost/tau_stdp : volt (event-driven)
    ''',
    on_pre='''
    v_post += w
    Apre += dApre
    w = clip(w + Apost, 0*mV, wmax)
    ''',
    on_post='''
    Apost += dApost
    w = clip(w + Apre, 0*mV, wmax)
    '''
)

syn_thalamus_cortex.connect(p=0.3)
syn_thalamus_cortex.w = '2.5*mV'  # Başlangıç ağırlık

# STDP parametreleri
tau_stdp = 20*ms
dApre = 0.5*mV
dApost = -0.5*mV
wmax = 5*mV  # Maksimum ağırlık

# Korteks içi rekürrent bağlantılar (opsiyonel, basit)
syn_cortex = Synapses(cortex, cortex, on_pre='v_post += 1.5*mV')
syn_cortex.connect(condition='i!=j', p=0.1)

# Monitörler
spike_mon_thalamus = SpikeMonitor(thalamus)
spike_mon_cortex = SpikeMonitor(cortex)
state_mon_cortex = StateMonitor(cortex, 'v', record=True)
weight_mon = StateMonitor(syn_thalamus_cortex, 'w', record=True)

# Çalıştır
run(duration)

# Grafikler
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title('Talamik Giriş Spikeleri')
plt.plot(spike_mon_thalamus.t/ms, spike_mon_thalamus.i, '.k')
plt.xlabel('Zaman (ms)')
plt.ylabel('Talamik Nöron #')

plt.subplot(3, 1, 2)
plt.title('Kortikal Nöron Spikeleri')
plt.plot(spike_mon_cortex.t/ms, spike_mon_cortex.i, '.r')
plt.xlabel('Zaman (ms)')
plt.ylabel('Kortikal Nöron #')

plt.subplot(3, 1, 3)
for idx in range(5):  # İlk 5 sinaps için ağırlık değişimini çizelim
    plt.plot(weight_mon.t/ms, weight_mon.w[idx]/mV, label=f'Sinaps {idx}')
plt.title('Talamus-Korteks Sinaptik Ağırlık Değişimi (STDP)')
plt.xlabel('Zaman (ms)')
plt.ylabel('Ağırlık (mV)')
plt.legend()

plt.tight_layout()
plt.show() 
