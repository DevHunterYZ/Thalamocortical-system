from brian2 import *

# Genel Ayarlar
start_scope()
prefs.codegen.target = 'numpy'
defaultclock.dt = 0.1*ms

# Süreler
sim_time = 1000*ms

# Nöron parametreleri
v_rest = -65*mV
v_reset = -65*mV
v_threshold = -50*mV
tau = 10*ms

# Kortikal nöron grubu
N_cortex = 40
eqs = '''
dv/dt = (v_rest - v) / tau : volt (unless refractory)
'''
cortex = NeuronGroup(N_cortex, eqs, threshold='v>v_threshold', reset='v=v_reset', refractory=5*ms, method='exact')
cortex.v = v_rest

# --- Eğitim Girdileri ---
n_pattern_neurons = 5
pattern_times_single = np.arange(50, 800, 100)*ms  # 50,150,...750 ms
pattern_indices = np.repeat(np.arange(n_pattern_neurons), len(pattern_times_single))
pattern_times = np.tile(pattern_times_single, n_pattern_neurons)

assert len(pattern_indices) == len(pattern_times), "Pattern index ve time eşleşmiyor!"

pattern_generator = SpikeGeneratorGroup(40, pattern_indices, pattern_times)

# Gürültü inputu
noise_generator = PoissonGroup(40, rates=0*Hz)  # Gürültüyü kapat

# --- Synapslar ---
# STDP için parametreler
tau_pre = 20*ms
tau_post = 20*ms
A_pre = 0.1  # Öğrenme oranını artır
A_post = -A_pre * tau_pre / tau_post * 1.05
w_max = 20*mV

# Pattern girişinin kortekse bağlanması (ÖĞRENEN synaps)
syn_pattern = Synapses(pattern_generator, cortex,
    model='''
    w : volt
    dapre/dt = -apre/tau_pre : 1 (event-driven)
    dapost/dt = -apost/tau_post : 1 (event-driven)
    ''',
    on_pre='''
    v_post += w
    apre += A_pre
    w = clip(w + apost * mV, 0*mV, w_max)
    ''',
    on_post='''
    apost += A_post
    w = clip(w + apre * mV, 0*mV, w_max)
    ''',
    method='exact')

syn_pattern.connect(p=0.8)  # Daha fazla bağlantı
syn_pattern.w = '5*mV + rand()*5*mV'

# Gürültü girişinin kortekse bağlanması
syn_noise = Synapses(noise_generator, cortex, on_pre='v_post += 1.5*mV')
syn_noise.connect(p=0.1)

# --- Monitörler ---
mon_w = StateMonitor(syn_pattern, 'w', record=range(5))
spike_in_pattern = SpikeMonitor(pattern_generator)
spike_in_noise = SpikeMonitor(noise_generator)
spike_cortex_train = SpikeMonitor(cortex)

# --- Eğitim Aşaması ---
net = Network(collect())
net.run(sim_time)

# Eğitim sonrası ağırlıkları ve bağlantıları kaydet
saved_weights = syn_pattern.w[:]
saved_i = syn_pattern.i[:]
saved_j = syn_pattern.j[:]

# Ağırlıkları kontrol et
print(f"Eğitim sonrası ağırlıklar (ilk 5 sinaps): {saved_weights[:5]/mV} mV")

# --- Test Aşaması ---
# Eğitim nesnelerini devre dışı bırak
pattern_generator.active = False
noise_generator.active = False
syn_pattern.active = False
syn_noise.active = False
mon_w.active = False
spike_in_pattern.active = False
spike_in_noise.active = False
spike_cortex_train.active = False

# Test için yeni input grubu (zamanları kaydır)
inputs_test = SpikeGeneratorGroup(40, pattern_indices, pattern_times + sim_time)

# Cortex voltaj reset
cortex.v = v_rest

# Yeni test synapsı
syn_pattern_test = Synapses(inputs_test, cortex,
    model='w : volt',
    on_pre='v_post += w',
    method='exact')
syn_pattern_test.connect(i=saved_i, j=saved_j)
syn_pattern_test.w = saved_weights

# Monitörler
spike_in_test = SpikeMonitor(inputs_test)
spike_cortex_test = SpikeMonitor(cortex)

# Test nesnelerini mevcut ağa ekle
net.add(inputs_test)
net.add(syn_pattern_test)
net.add(spike_in_test)
net.add(spike_cortex_test)

# Test simülasyonunu çalıştır
net.run(sim_time)

# Test spike'larını kontrol et
print(f"Testte korteks spike sayısı: {len(spike_cortex_test.t)}")

# --- Grafikler ---
import matplotlib.pyplot as plt

fig, axs = plt.subplots(4, 1, figsize=(14, 18))

# Eğitim inputları
axs[0].plot(spike_in_noise.t/ms, spike_in_noise.i, '.', color='gray', markersize=3, label='Noise')
axs[0].plot(spike_in_pattern.t/ms, spike_in_pattern.i, '.', color='blue', markersize=3, label='Pattern (Train)')
axs[0].set_title('Eğitim Girdileri (Pattern + Gürültü)')
axs[0].set_ylabel('Talamik Nöron #')
axs[0].legend()

# Eğitimde korteks spike'ları
axs[1].plot(spike_cortex_train.t/ms, spike_cortex_train.i, '.', color='red')
axs[1].set_title('Eğitimde Korteks Tepkisi')
axs[1].set_ylabel('Korteks Nöron #')

# Ağırlık değişimi
for idx in range(5):
    axs[2].plot(mon_w.t/ms, mon_w.w[idx]/mV, label=f'Sinaps {idx}')
axs[2].set_title('Talamus -> Korteks Sinaptik Ağırlık Değişimi (Öğrenme)')
axs[2].set_ylabel('Ağırlık (mV)')
axs[2].legend()

# Test aşamasında korteks spike'ları
axs[3].plot(spike_in_test.t/ms, spike_in_test.i, '.', color='blue', markersize=4, label='Pattern (Test)')
axs[3].plot(spike_cortex_test.t/ms, spike_cortex_test.i, '.', color='green', markersize=4, label='Korteks (Test)')
axs[3].set_title('TEST Aşaması: Öğrenilen Pattern Tepkisi')
axs[3].set_xlabel('Zaman (ms)')
axs[3].set_ylabel('Nöron #')
axs[3].legend()

plt.tight_layout()
plt.show()
