from brian2 import *

start_scope()

duration = 1000*ms
defaultclock.dt = 0.1*ms

# Ortak nöron modeli
eqs = '''
dv/dt = -v / (10*ms) : 1
'''

# Nöron grupları
thal = NeuronGroup(5, eqs, threshold='v>0.3', reset='v=0', refractory=2*ms, method='exact')
v1 = NeuronGroup(5, eqs, threshold='v>0.3', reset='v=0', refractory=2*ms, method='exact')
assoc = NeuronGroup(5, eqs, threshold='v>0.3', reset='v=0', refractory=2*ms, method='exact')
workspace = NeuronGroup(5, eqs, threshold='v>0.3', reset='v=0', refractory=2*ms, method='exact')
attention = NeuronGroup(5, eqs, threshold='v>0.3', reset='v=0', refractory=2*ms, method='exact')

# Giriş spike’ları (thalamus’a dış uyaran)
input_spikes = SpikeGeneratorGroup(1, [0]*5, [100, 300, 500, 700, 900]*ms)
stim_syn = Synapses(input_spikes, thal, on_pre='v_post += 2.0')
stim_syn.connect(j='i % 5')

# Bağlantılar (düzgün sinaptik ağırlıklarla)
thal_v1 = Synapses(thal, v1, on_pre='v_post += 2.0')
thal_v1.connect()

v1_assoc = Synapses(v1, assoc, on_pre='v_post += 2.0')
v1_assoc.connect()

assoc_ws = Synapses(assoc, workspace, on_pre='v_post += 2.0')
assoc_ws.connect()

# Global Workspace’ten geri bağlantı
ws_assoc = Synapses(workspace, assoc, on_pre='v_post += 1.5')
ws_assoc.connect()

# Attention sistemi bağlantıları
attn_ws = Synapses(attention, workspace, on_pre='v_post += 1.5')
attn_ws.connect()
ws_attn = Synapses(workspace, attention, on_pre='v_post += 1.5')
ws_attn.connect()

# Attention’ın Assoc ve Thal’a modülasyonu
attn_assoc = Synapses(attention, assoc, on_pre='v_post += 1.5')
attn_assoc.connect()
attn_thal = Synapses(attention, thal, on_pre='v_post += 1.5')
attn_thal.connect()

# Spike Monitörleri
M_thal = SpikeMonitor(thal)
M_v1 = SpikeMonitor(v1)
M_assoc = SpikeMonitor(assoc)
M_ws = SpikeMonitor(workspace)
M_attn = SpikeMonitor(attention)

# Simülasyonu çalıştır
run(duration)

# Spike raster plot
import matplotlib.pyplot as plt

fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
axs[0].plot(M_thal.t/ms, M_thal.i, 'ro', label='Thalamus')
axs[1].plot(M_v1.t/ms, M_v1.i, 'go', label='V1')
axs[2].plot(M_assoc.t/ms, M_assoc.i, 'bo', label='Association Cortex')
axs[3].plot(M_ws.t/ms, M_ws.i, 'ko', label='Global Workspace')
axs[4].plot(M_attn.t/ms, M_attn.i, 'mo', label='Attention System')

for ax in axs:
    ax.legend()
    ax.set_ylabel('Nöron')
axs[-1].set_xlabel('Zaman (ms)')
plt.tight_layout()
plt.show()
