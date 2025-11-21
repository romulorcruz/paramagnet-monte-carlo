import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

L = 20              # Tamanho do lado da rede
N_spins = L * L
B = 1.0             # Campo Magnético Externo
kB = 1.0            # Constante de Boltzmann
steps_equil = 1000  # Passos para termalização
steps_prod = 5000   # Passos para média

T_range = np.linspace(0.1, 5.0, 30)


def inicializar_rede(L):
    # Rede inicial com spins +1 ou -1 aleatórios
    return np.random.choice([-1, 1], size=(L, L))

def passo_metro(rede, T, B):
    L = rede.shape[0]
    for _ in range(L*L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        S = rede[i, j]
        # Desconsiderando vizinhos
        dE = 2 * B * S # Ajustada em relação a ising
        
        if dE < 0 or np.random.rand() < np.exp(-dE / (kB * T)):
            rede[i, j] *= -1
    return rede

def simular_paramagneto(L, temps, equil, prod, B):
    # Campo magnético inserido para calculo de energia
    mag_mean = []
    mag_std = []
    
    print(f"Iniciando simulação para N={L*L} spins...")
    rede = inicializar_rede(L)
    
    for T in tqdm(temps):
        for _ in range(equil):
            passo_metro(rede, T, B)
            
        m_coleta = []
        for _ in range(prod):
            passo_metro(rede, T, B)
            m_atual = np.sum(rede)
            m_coleta.append(m_atual)
            
        m_avg = np.mean(m_coleta) / (L*L)
        m_err = np.std(m_coleta) / (L*L)
        
        mag_mean.append(m_avg)
        mag_std.append(m_err)
        
    return np.array(mag_mean), np.array(mag_std)

def teoria_paramagneto(T_array, B):
    # Valores teóricos por cálculo analítico
    beta = 1.0 / (kB * T_array)
    x = beta * B
    m_teoria = np.tanh(x)
    s_teoria = np.log(2 * np.cosh(x)) - x * np.tanh(x)
    return m_teoria, s_teoria


m_sim, m_err = simular_paramagneto(L, T_range, steps_equil, steps_prod, B)

m_teo, s_teo = teoria_paramagneto(T_range, B)

eps = 1e-9 # evitar erro de log 0
p_up = (1 + m_sim) / 2.0 
p_down = (1 - m_sim) / 2.0
p_up = np.clip(p_up, eps, 1-eps)
p_down = np.clip(p_down, eps, 1-eps)
s_sim = - (p_up * np.log(p_up) + p_down * np.log(p_down))


def salvar_dados(filename):
    print(f"Salvando dados em {filename}...")
    dados = np.column_stack((T_range, m_sim, m_err, m_teo, s_teo, s_sim))
    
    header = (f"Simulação Paramagneto Spin 1/2 (L={L}, B={B})\n"
              f"Colunas:\n"
              f"1: Temperatura (T)\n"
              f"2: Magnetização Simulada (m_sim)\n"
              f"3: Erro Magnetização (m_err)\n"
              f"4: Magnetização Teórica (m_teo)\n"
              f"5: Entropia Teórica (s_teo)\n"
              f"6: Entropia Inferida Simulação (s_sim)")
    
    np.savetxt(filename, dados, header=header, fmt='%10.5f')
    print("Dados salvos com sucesso.")

salvar_dados('dados_paramagneto.txt')

plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

# Magnetização
ax1.plot(T_range, m_teo, 'b-', label='Teoria')
ax1.errorbar(T_range, m_sim, yerr=m_err, fmt='ro', alpha=0.7, label='Simulação')
ax1.set_ylabel(r'$\langle m \rangle$')
ax1.legend()
ax1.set_title('Paramagneto: Validação e Entropia')

# Entropia
ax2.plot(T_range, s_teo, 'g-', label='Entropia Teórica')
ax2.plot(T_range, s_sim, 'rx', label='Entropia Simulada')
ax2.axhline(np.log(2), color='k', linestyle='--', label='ln(2)')
ax2.set_ylabel(r'Entropia ($S/k_B$)')
ax2.set_xlabel('Temperatura (T)')
ax2.legend()

plt.tight_layout()
plt.savefig('paramagneto_com_dados.png')
plt.show()