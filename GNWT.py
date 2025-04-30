# Simülasyon süresi
duration = 500*ms
defaultclock.dt = 1*ms

# Uyaran (görsel gibi) → sinüs şeklinde giriş
stim_freq = 20  # Hz (düşük frekans bilinçli algı eşiğinde)
times = arange(0, int(duration/defaultclock.dt)) * defaultclock.dt
stim_input = 2.5 * np.sin(2 * np.pi * stim_freq * times / second)
stim = TimedArray(stim_input, dt=defaultclock.dt)

# Ortak nöron modeli
eqs = '''
dv/dt = (stim(t) - v) / (10*ms) : 1
'''

# Nöron grupları
thal = NeuronGroup(5, eqs, threshold='v > 0.6', reset='v = 0', method='exact')
v1 = NeuronGroup(5, eqs, threshold='v > 0.6', reset='v = 0', method='exact')  # Primer görsel korteks
assoc = NeuronGroup(5, eqs, threshold='v > 0.6', reset='v = 0', method='exact')  # Birleşim korteksi
workspace = NeuronGroup(5, eqs, threshold='v > 0.6', reset='v = 0', method='exact')  # Bilinçli farkındalık bölgesi

# Başlangıç voltajları
for grp in [thal, v1, assoc, workspace]:
    grp.v = '0.4 + 0.2*rand()'

# Bağlantılar (Talamus → V1 → Assoc → Workspace)
S_th_v1 = Synapses(thal, v1, on_pre='v_post += 0.4')
S_v1_assoc = Synapses(v1, assoc, on_pre='v_post += 0.4')
S_assoc_ws = Synapses(assoc, workspace, on_pre='v_post += 0.5')  # Daha yüksek etki

S_th_v1.connect(p=1.0)
S_v1_assoc.connect(p=1.0)
S_assoc_ws.connect(p=1.0)

# Spike Monitörleri
spike_mon_th = SpikeMonitor(thal)
spike_mon_v1 = SpikeMonitor(v1)
spike_mon_assoc = SpikeMonitor(assoc)
spike_mon_ws = SpikeMonitor(workspace)

# Simülasyonu çalıştır
run(duration)

# Grafikler
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(411)
plt.plot(spike_mon_th.t/ms, spike_mon_th.i, 'r.', label='Thalamus')
plt.legend()
plt.subplot(412)
plt.plot(spike_mon_v1.t/ms, spike_mon_v1.i, 'g.', label='V1 (Primary Cortex)')
plt.legend()
plt.subplot(413)
plt.plot(spike_mon_assoc.t/ms, spike_mon_assoc.i, 'b.', label='Association Cortex')
plt.legend()
plt.subplot(414)
plt.plot(spike_mon_ws.t/ms, spike_mon_ws.i, 'k.', label='Global Workspace (Conscious Awareness)')
plt.legend()
plt.xlabel('Time (ms)')
plt.tight_layout()
plt.show()

# Bilinçli algı kontrolü
if len(spike_mon_ws.t) > 0:
    print(">>> Bilinçli algı oluştu: Workspace spike üretti.")
else:
    print(">>> Bilinçli algı oluşmadı.")
