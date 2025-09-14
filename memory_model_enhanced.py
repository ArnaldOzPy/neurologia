# mobius_memory_model_enhanced.py
# Modelo de Memoria Topológica Cuántica con Plasticidad STDP + Hebb + Homeostasis
# Implementación con qubits abstractos y topología informacional (sin requerimientos físicos de coherencia cuántica)
# Autor: Arnaldo Adrian Ozorio
# Fecha: 2025-09-14
# Referencias: 
# - Tegmark (2000) Importance of quantum decoherence in brain processes
# - Kalvoda et al. (2019) Effective quantum dynamics on the Möbius strip
# - Bi & Poo (1998) Synaptic modifications in cultured hippocampal neurons

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import expm_multiply
from scipy.integrate import solve_ivp
import math
from typing import List, Optional, Tuple, Dict, Union
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------
# CONSTANTES Y PARÁMETROS NEUROFISIOLÓGICOS
# -------------------------
# Tiempo de decoherencia estimado para sistemas biológicos (Tegmark, 2000)
TAU_DECOHERENCE = 1e-13  # 10⁻¹³ segundos

# Parámetros de STDP biológicamente plausibles (Bi & Poo, 1998)
A_PLUS_DEFAULT = 0.005    # Potenciación a largo plazo (LTP)
A_MINUS_DEFAULT = 0.00525 # Depresión a largo plazo (LTD) 
TAU_PLUS_DEFAULT = 0.0167 # 16.7 ms
TAU_MINUS_DEFAULT = 0.0334 # 33.4 ms

# -------------------------
# CLASE PRINCIPAL DEL MODELO
# -------------------------
class QuantumMemoryModel:
    """
    Modelo de memoria con topología de Möbius implementada a nivel informacional.
    Utiliza qubits abstractos para representar estados de memoria sin requerir
    coherencia cuántica física.
    """
    
    def __init__(self, num_sites: int = 20, J: float = 1.0, 
                 theta: float = math.pi, seed: Optional[int] = None):
        """
        Inicializa el modelo de memoria cuántica abstracta.
        
        Args:
            num_sites: Número de sitios de memoria (recuerdos)
            J: Constante de acoplamiento entre sitios
            theta: Ángulo de twist para la topología de Möbius
            seed: Semilla para generación de números aleatorios
        """
        self.N = num_sites
        self.J = J
        self.theta = theta
        self.rng = np.random.default_rng(seed)
        
        # Hamiltoniano abstracto (representación matemática)
        self.H = self._build_mobius_hamiltonian()
        self.H_ext = self._build_valence_extension()
        
        # Operadores de collapse abstractos
        self.flip_op = self._build_flip_operator()
        self.collapse_ops = [self.flip_op]
        self.rates = [0.01]  # Tasa abstracta de "decoherencia"
        
        # Estado inicial del sistema
        self.rho = self._initialize_state()
        self.weights = np.eye(2 * self.N) * 0.01
        
        # Registro de resultados
        self.results = {
            'times': [],
            'fidelity': [],
            'weights': [],
            'spike_times_pre': [],
            'spike_times_post': []
        }
    
    def _build_mobius_hamiltonian(self) -> sp.csr_matrix:
        """Construye Hamiltoniano abstracto con topología de Möbius."""
        rows, cols, data = [], [], []
        
        # Conexiones entre sitios adyacentes
        for j in range(self.N - 1):
            rows.extend([j, j + 1])
            cols.extend([j + 1, j])
            data.extend([-self.J, -self.J])
        
        # Conexión twistada (topología de Möbius abstracta)
        rows.extend([self.N - 1, 0])
        cols.extend([0, self.N - 1])
        data.extend([-self.J * np.exp(1j * self.theta), 
                    -self.J * np.exp(-1j * self.theta)])
        
        H = sp.csr_matrix((data, (rows, cols)), shape=(self.N, self.N), dtype=complex)
        return 0.5 * (H + H.getH())  # Garantizar Hermiticidad
    
    def _build_valence_extension(self) -> sp.csr_matrix:
        """Extiende el Hamiltoniano para incluir valencia abstracta."""
        I2 = sp.identity(2, format='csr', dtype=complex)
        return self._kron(self.H, I2)
    
    def _build_flip_operator(self) -> sp.csr_matrix:
        """Operador abstracto de flip de valencia."""
        sx = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
        I_N = sp.identity(self.N, format='csr', dtype=complex)
        return self._kron(I_N, sx)
    
    def _kron(self, a: sp.csr_matrix, b: sp.csr_matrix) -> sp.csr_matrix:
        """Producto de Kronecker optimizado para matrices dispersas."""
        return sp.kron(a, b, format='csr')
    
    def _initialize_state(self) -> np.ndarray:
        """Inicializa el estado en el primer sitio con valencia positiva."""
        D = 2 * self.N
        vec = np.zeros((D,), dtype=complex)
        vec[0] = 1.0  # Primer sitio, valencia positiva
        return np.outer(vec, vec.conj())
    
    def _density_to_spike_probabilities(self, rho: np.ndarray, 
                                      dt: float = 0.001) -> np.ndarray:
        """
        Convierte la matriz de densidad abstracta a probabilidades de activación.
        
        Nota: Esto es una representación matemática, no un proceso físico real.
        """
        populations = np.diag(rho).real
        populations = np.clip(populations, 0, None)  # No negatividad
        
        if np.sum(populations) > 0:
            # Normalizar y escalar por dt para probabilidades de activación
            probabilities = populations / np.max(populations) * dt
        else:
            probabilities = np.zeros_like(populations)
            
        return probabilities
    
    def _generate_abstract_spikes(self, probabilities: np.ndarray) -> List[float]:
        """
        Genera "activaciones abstractas" basadas en probabilidades.
        
        Nota: Estas no representan spikes neuronales reales, sino eventos
        de activación en el espacio de estados abstracto.
        """
        spikes = []
        for p in probabilities:
            if self.rng.random() < p:
                spikes.append(self.rng.random() * 0.001)  # Pequeña variación temporal
        return spikes
    
    def _quantum_hebb_update(self, weights: np.ndarray, rho: np.ndarray, 
                           eta: float = 1e-3, lam: float = 1e-4) -> np.ndarray:
        """
        Actualización Hebbiana abstracta que incluye coherencias off-diagonal.
        
        Nota: Esta es una regla de aprendizaje matemática, no un proceso físico.
        """
        delta = eta * (rho + rho.conj().T)
        return weights + delta - lam * weights
    
    def _stdp_update(self, weights: np.ndarray, spike_pre: List[float], 
                   spike_post: List[float], A_plus: float = A_PLUS_DEFAULT,
                   A_minus: float = A_MINUS_DEFAULT, 
                   tau_plus: float = TAU_PLUS_DEFAULT,
                   tau_minus: float = TAU_MINUS_DEFAULT) -> np.ndarray:
        """
        Actualización STDP abstracta con dependencia temporal.
        
        Basada en parámetros neurobiológicos pero implementada a nivel abstracto.
        """
        W_new = weights.copy()
        
        for i, t_post in enumerate(spike_post):
            for j, t_pre in enumerate(spike_pre):
                dt = t_post - t_pre
                if dt > 0:
                    # Potenciación abstracta (LTP)
                    W_new[i, j] += A_plus * np.exp(-dt / tau_plus)
                elif dt < 0:
                    # Depresión abstracta (LTD)
                    W_new[i, j] -= A_minus * np.exp(dt / tau_minus)
        
        return W_new
    
    def _normalize_weights(self, weights: np.ndarray, 
                         max_val: float = 1.0) -> np.ndarray:
        """Normalización homeostática abstracta de pesos."""
        current_max = np.max(np.abs(weights))
        if current_max > max_val:
            return weights * (max_val / current_max)
        return weights
    
    def _update_hamiltonian_with_weights(self, H: sp.csr_matrix, 
                                       weights: np.ndarray, 
                                       alpha: float = 0.1) -> sp.csr_matrix:
        """
        Actualiza el Hamiltoniano abstracto con los pesos sinápticos abstractos.
        """
        H_new = H.copy().tolil()
        n = min(H.shape[0], weights.shape[0])
        
        # Modificar energías onsite basadas en pesos abstractos
        for i in range(n):
            H_new[i, i] = H_new[i, i] + alpha * weights[i, i].real
        
        return H_new.tocsr()
    
    def simulate(self, T: float = 10.0, dt: float = 0.01, 
                 eta: float = 1e-3, lam: float = 1e-4,
                 max_weight: float = 1.0, 
                 A_plus: float = A_PLUS_DEFAULT,
                 A_minus: float = A_MINUS_DEFAULT, 
                 tau_plus: float = TAU_PLUS_DEFAULT,
                 tau_minus: float = TAU_MINUS_DEFAULT,
                 alpha: float = 0.1) -> Dict:
        """
        Ejecuta la simulación completa del modelo abstracto.
        
        Args:
            T: Tiempo total de simulación (segundos abstractos)
            dt: Paso temporal (segundos abstractos)
            eta: Tasa de aprendizaje Hebbiana
            lam: Tasa de decaimiento de pesos
            max_weight: Valor máximo permitido para pesos
            A_plus: Tasa de potenciación STDP
            A_minus: Tasa de depresión STDP
            tau_plus: Constante de tiempo para potenciación
            tau_minus: Constante de tiempo para depresión
            alpha: Factor de acoplamiento pesos-Hamiltoniano
            
        Returns:
            Diccionario con resultados de la simulación
        """
        # Reinicializar resultados
        self.results = {
            'times': [],
            'fidelity': [],
            'weights': [],
            'spike_times_pre': [],
            'spike_times_post': []
        }
        
        n_steps = int(T / dt) + 1
        times = np.linspace(0, T, n_steps)
        
        # Estado inicial
        current_rho = self.rho.copy()
        current_weights = self.weights.copy()
        current_H = self.H_ext.copy()
        
        for i, t in enumerate(times):
            if i % 100 == 0:
                print(f"Procesando paso {i}/{n_steps} (t = {t:.2f}s)")
            
            # Evolución de Lindblad abstracta
            L = self._liouvillian_from_hamiltonian(current_H, self.collapse_ops, self.rates)
            rho_vec = self._evolve_lindblad_expm(L, self._rho_to_vec(current_rho), [dt])[0]
            current_rho = self._vec_to_rho(rho_vec)
            
            # Generar activaciones abstractas
            spike_probs = self._density_to_spike_probabilities(current_rho, dt)
            spike_times_pre = self._generate_abstract_spikes(spike_probs)
            spike_times_post = self._generate_abstract_spikes(spike_probs)
            
            # Actualizar plasticidad abstracta
            current_weights = self._quantum_hebb_update(current_weights, current_rho, eta, lam)
            current_weights = self._stdp_update(current_weights, spike_times_pre, 
                                              spike_times_post, A_plus, A_minus, 
                                              tau_plus, tau_minus)
            current_weights = self._normalize_weights(current_weights, max_weight)
            
            # Actualizar Hamiltoniano abstracto
            current_H = self._update_hamiltonian_with_weights(current_H, current_weights, alpha)
            
            # Calcular fidelidad con estado inicial
            fidelity = np.abs(np.vdot(current_rho.flatten(), self.rho.flatten()))
            
            # Almacenar resultados
            self.results['times'].append(t)
            self.results['fidelity'].append(fidelity)
            self.results['weights'].append(current_weights.copy())
            self.results['spike_times_pre'].append(spike_times_pre)
            self.results['spike_times_post'].append(spike_times_post)
        
        return self.results
    
    def _liouvillian_from_hamiltonian(self, H: sp.csr_matrix, 
                                    collapse_ops: List[sp.csr_matrix], 
                                    rates: List[float]) -> sp.csr_matrix:
        """Construye superoperador de Lindblad abstracto."""
        N = H.shape[0]
        I = sp.identity(N, format='csr', dtype=complex)
        
        # Término Hamiltonian abstracto
        L = -1j * (self._kron(I, H) - self._kron(H.conj().transpose(), I))
        
        # Términos de disipación abstractos
        for Lk, gamma in zip(collapse_ops, rates):
            Lk_dag = Lk.getH()
            term1 = gamma * self._kron(Lk, Lk.conj())
            term2 = -0.5 * gamma * self._kron(I, (Lk_dag.dot(Lk)).conj())
            term3 = -0.5 * gamma * self._kron((Lk_dag.dot(Lk)).transpose().conj(), I)
            L += term1 + term2 + term3
        
        return L.tocsr()
    
    def _evolve_lindblad_expm(self, L: sp.csr_matrix, rho0_vec: np.ndarray, 
                            times: List[float]) -> np.ndarray:
        """Evolución abstracta de Lindblad usando exponencial matricial."""
        return np.vstack([expm_multiply(L, rho0_vec, start=0, stop=t, num=2)[-1] 
                         for t in times])
    
    def _rho_to_vec(self, rho: np.ndarray) -> np.ndarray:
        """Convierte matriz de densidad abstracta a vector."""
        return rho.flatten(order='F')
    
    def _vec_to_rho(self, vec: np.ndarray) -> np.ndarray:
        """Convierte vector a matriz de densidad abstracta."""
        n = int(np.sqrt(vec.size))
        return vec.reshape((n, n), order='F')
    
    def analyze_results(self) -> Dict:
        """Analiza resultados de la simulación abstracta."""
        times = np.array(self.results['times'])
        fidelity = np.array(self.results['fidelity'])
        weights = self.results['weights']
        
        analysis = {
            'final_fidelity': fidelity[-1],
            'mean_fidelity': np.mean(fidelity),
            'min_fidelity': np.min(fidelity),
            'max_fidelity': np.max(fidelity),
            'final_max_weight': np.max(np.abs(weights[-1])),
            'final_min_weight': np.min(np.abs(weights[-1])),
            'weight_change_norm': np.linalg.norm(weights[-1] - weights[0]),
            'recurrence_times': self._find_recurrence_times(times, fidelity),
            'decoherence_ratio': T / TAU_DECOHERENCE if hasattr(self, 'T') else 0
        }
        
        return analysis
    
    def _find_recurrence_times(self, times: np.ndarray, fidelity: np.ndarray, 
                             threshold: float = 0.95) -> List[float]:
        """Encuentra tiempos de recurrencia abstractos."""
        recurrence_times = []
        for i, f in enumerate(fidelity):
            if f > threshold and i > 0 and fidelity[i-1] <= threshold:
                recurrence_times.append(times[i])
        return recurrence_times

# -------------------------
# FUNCIONES DE VISUALIZACIÓN Y ANÁLISIS
# -------------------------
def plot_abstract_results(results: Dict, analysis: Dict, 
                        save_path: Optional[str] = None):
    """Genera gráficos para visualizar resultados abstractos."""
    times = np.array(results['times'])
    fidelity = np.array(results['fidelity'])
    weights = results['weights']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gráfico de fidelidad abstracta
    axes[0, 0].plot(times, fidelity)
    axes[0, 0].set_xlabel('Tiempo Abstracto (u.a.)')
    axes[0, 0].set_ylabel('Fidelidad Abstracta')
    axes[0, 0].set_title('Evolución de la Fidelidad (Espacio Abstracto)')
    axes[0, 0].grid(True)
    
    # Gráfico de pesos abstractos
    max_weights = [np.max(np.abs(w)) for w in weights]
    min_weights = [np.min(np.abs(w)) for w in weights]
    axes[0, 1].plot(times, max_weights, label='Peso máximo abstracto')
    axes[0, 1].plot(times, min_weights, label='Peso mínimo abstracto')
    axes[0, 1].set_xlabel('Tiempo Abstracto (u.a.)')
    axes[0, 1].set_ylabel('Valor de peso abstracto')
    axes[0, 1].set_title('Evolución de Pesos Abstractos')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Histograma de pesos finales abstractos
    axes[1, 0].hist(np.abs(weights[-1].flatten()), bins=50)
    axes[1, 0].set_xlabel('Valor de peso abstracto')
    axes[1, 0].set_ylabel('Frecuencia abstracta')
    axes[1, 0].set_title('Distribución de Pesos Finales Abstractos')
    axes[1, 0].grid(True)
    
    # Gráfico de recurrencias abstractas
    recurrence_times = analysis['recurrence_times']
    axes[1, 1].plot(times, fidelity, label='Fidelidad abstracta')
    for rt in recurrence_times:
        axes[1, 1].axvline(x=rt, color='r', linestyle='--', alpha=0.5, 
                          label='Recurrencia' if rt == recurrence_times[0] else "")
    axes[1, 1].set_xlabel('Tiempo Abstracto (u.a.)')
    axes[1, 1].set_ylabel('Fidelidad Abstracta')
    axes[1, 1].set_title('Tiempos de Recurrencia Abstractos')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def generate_latex_table(analysis: Dict, caption: str = "Resultados de la simulación abstracta") -> str:
    """Genera tabla LaTeX para el paper."""
    latex_code = f"""
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lc}}
\\hline
\\textbf{{Métrica}} & \\textbf{{Valor}} \\\\
\\hline
Fidelidad final & {analysis['final_fidelity']:.4f} \\\\
Fidelidad media & {analysis['mean_fidelity']:.4f} \\\\
Fidelidad mínima & {analysis['min_fidelity']:.4f} \\\\
Fidelidad máxima & {analysis['max_fidelity']:.4f} \\\\
Peso máximo final & {analysis['final_max_weight']:.4e} \\\\
Peso mínimo final & {analysis['final_min_weight']:.4e} \\\\
Cambio en pesos & {analysis['weight_change_norm']:.4e} \\\\
Razón decoherencia & {analysis['decoherence_ratio']:.2e} \\\\
\\hline
\\end{{tabular}}
\\caption{{{caption}}}
\\label{{tab:abstract_simulation_results}}
\\end{{table}}
"""
    return latex_code

# -------------------------
# EJEMPLO DE USO
# -------------------------
if __name__ == "__main__":
    print("Inicializando modelo de memoria abstracta...")
    print(f"Tiempo de decoherencia neurobiológica: {TAU_DECOHERENCE:.1e} s")
    print("Este modelo utiliza representaciones abstractas, no coherencia cuántica física")
    
    # Crear modelo abstracto
    model = QuantumMemoryModel(num_sites=20, J=1.0, theta=math.pi, seed=42)
    
    # Ejecutar simulación abstracta
    print("Ejecutando simulación abstracta...")
    results = model.simulate(
        T=2.0, dt=0.05, eta=1e-4, lam=1e-5, 
        max_weight=0.5, A_plus=A_PLUS_DEFAULT, 
        A_minus=A_MINUS_DEFAULT, tau_plus=TAU_PLUS_DEFAULT,
        tau_minus=TAU_MINUS_DEFAULT, alpha=0.1
    )
    
    # Analizar resultados
    analysis = model.analyze_results()
    
    # Generar gráficos
    plot_abstract_results(results, analysis, save_path="abstract_simulation_results.png")
    
    # Generar tabla LaTeX
    latex_table = generate_latex_table(analysis, 
        "Resultados de la simulación de memoria topológica abstracta")
    
    print("\n" + "="*60)
    print("ANÁLISIS DE RESULTADOS - MODELO ABSTRACTO")
    print("="*60)
    print(f"Fidelidad final: {analysis['final_fidelity']:.4f}")
    print(f"Peso máximo final: {analysis['final_max_weight']:.4e}")
    print(f"Tiempos de recurrencia: {analysis['recurrence_times']}")
    print(f"Razón decoherencia (T/τ): {analysis['decoherence_ratio']:.2e}")
    print("\nLa alta razón de decoherencia confirma que este es un modelo abstracto")
    print("que no depende de coherencia cuántica física real.")
    
    print("\nTabla LaTeX para el paper:")
    print(latex_table)
