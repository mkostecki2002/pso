import math
import statistics
import time
import matplotlib.pyplot as plt
from pso import Swarm

# --- 1. Bezpieczne Funkcje Celu ---
def function_goldstein_price(x: float, y: float) -> float:
    try:
        term1 = 1 + (x + y + 1)**2 * (
            19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2
        )
        term2 = 30 + (2*x - 3*y)**2 * (
            18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2
        )
        return term1 * term2
    except (OverflowError, ValueError):
        return float('inf')

def function_beale(x: float, y: float) -> float:
    try:
        return (
            (1.5 - x + x*y)**2
            + (2.25 - x + x*y**2)**2
            + (2.625 - x + x*y**3)**2
        )
    except (OverflowError, ValueError):
        return float('inf')

# --- 2. Silnik Eksperymentów ---
def run_experiments(func, func_name, bounds, param_name, param_values, base_params, iterations=100):
    results_mean = []
    results_best = []
    results_std = []

    print(f"\n>>> Rozpoczynam badanie parametru: '{param_name}' dla {func_name}")
    
    for val in param_values:
        # WAŻNE: Kopia parametrów, aby nie nadpisywać ich dla kolejnych pętli
        current_params = base_params.copy()
        current_params[param_name] = val
        
        trial_scores = []
        
        # Uruchamiamy 5 prób dla jednej wartości parametru
        for i in range(5):
            swarm = Swarm(
                num_of_particles=current_params['n'], 
                bounds=bounds, 
                inertia=current_params['inertia'], 
                cognitive_factor=current_params['c'], 
                social_factor=current_params['s']
            )
            score, pos = swarm.run(func, iterations=iterations)
            trial_scores.append(score)
        
        # Obliczamy statystyki dla tej wartości parametru
        mean_val = statistics.mean(trial_scores)
        best_val = min(trial_scores)
        std_val = statistics.stdev(trial_scores) if len(trial_scores) > 1 else 0.0
        
        results_mean.append(mean_val)
        results_best.append(best_val)
        results_std.append(std_val)
        
        print(f"  {param_name}={val}: Średnia={mean_val:.4f}, Best={best_val:.4f}, StdDev={std_val:.4f}")
    
    return results_mean, results_best, results_std

# --- 3. Rysowanie Wykresów ---
def plot_experiment(func_name, param_label, x_values, mean_scores, best_scores, std_scores, filename):
    plt.figure(figsize=(10, 6))
    
    # 1. Rysowanie średniej z Error Bars (odchylenie standardowe)
    plt.errorbar(x_values, mean_scores, yerr=std_scores, 
                 fmt='-o', linewidth=1, capsize=2, label='Średnia ± StdDev', ecolor='red')
    
    # 2. Rysowanie najlepszego wyniku
    plt.plot(x_values, best_scores, 's--', alpha=0.7, label='Najlepszy wynik (Best)')
    
    plt.title(f'Wpływ: {param_label} na wynik ({func_name})')
    plt.xlabel(param_label)
    plt.ylabel('Wartość funkcji celu (skala log)')
    
    # Używamy skali logarytmicznej, bo wyniki mogą się różnić rzędami wielkości
    # plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Wygenerowano wykres: {filename}")
    plt.close()

# --- 4. Main ---
def main():
    print("== PSO: Optymalizacja Funkcji Dwóch Zmiennych ==")
    
    while True:
        print("\nDostępne funkcje:")
        print("1. Funkcja Goldstein-Price (domyślna)")
        print("2. Funkcja Beale")
        print("q. Wyjście")
        
        choice = input("Wybierz funkcję (1/2/q): ")
        if choice == 'q': break
        
        if choice == '2':
            func = function_beale
            func_name = "Beale"
            bounds = (-4.5, 4.5)
        else:
            func = function_goldstein_price
            func_name = "Goldstein-Price"
            bounds = (-2.0, 2.0)

        base_params = {
            'n': 30,
            'iterations': 50,
            'inertia': 0.5,
            'c': 1.5,
            's': 1.5
        }
        
        # --- ZESTAWY WARTOŚCI DO BADAŃ ---
        n_vals = [10, 20, 50, 100, 150]
        iterations_vals = [10, 30, 50, 100, 200]
        inertia_vals = [0.1, 0.3, 0.5, 0.7, 0.9] 
        components_vals = [0.1, 0.5, 0.7, 1.5, 2.0]

        base_path = 'results/experiment_'
        # 1. Badanie parametru inertia
        means, bests, stds = run_experiments(func, func_name, bounds, 'inertia', inertia_vals, base_params)
        plot_experiment(func_name, 'Inertia (w)', inertia_vals, means, bests, stds, f'{base_path}inertia_{func_name}.png')
        
        # 2. Badanie parametru n (populacja)
        means, bests, stds = run_experiments(func, func_name, bounds, 'n', n_vals, base_params)
        plot_experiment(func_name, 'Liczba cząstek (n)', n_vals, means, bests, stds, f'{base_path}population_{func_name}.png')

        # 3. Badanie parametru cognitive (c)
        means, bests, stds = run_experiments(func, func_name, bounds, 'c', components_vals, base_params)
        plot_experiment(func_name, 'Współczynnik kognitywny (c)', components_vals, means, bests, stds, f'{base_path}cognitive_{func_name}.png')

        # 4. Badanie parametru social (s)
        means, bests, stds = run_experiments(func, func_name, bounds, 's', components_vals, base_params)
        plot_experiment(func_name, 'Współczynnik społeczny (s)', components_vals, means, bests, stds, f'{base_path}social_{func_name}.png')

        # 5. Badanie liczby iteracji
        means, bests, stds = run_experiments(func, func_name, bounds, 'iterations', iterations_vals, base_params)
        plot_experiment(func_name, 'Liczba iteracji', iterations_vals, means, bests, stds, f'{base_path}iterations_{func_name}.png')

if __name__ == "__main__":
    main()