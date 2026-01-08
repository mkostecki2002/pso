import os
import statistics
import matplotlib.pyplot as plt
import numpy as np
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

def function_eggholder(x, y):
    # Minimum globalne: -959.6407 w punkcie (512, 404.2319)
    # Zakres DUŻY: [-512, 512]
    term1 = -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47))))
    term2 = -x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    return term1 + term2

def run_experiments(func, func_name, bounds, param_name, param_values, base_params, runs=5):
    results_mean = []
    results_median = []
    results_best = []
    results_worst = []
    results_std = []

    print(f"\n>>> Rozpoczynam badanie parametru: '{param_name}' dla {func_name}")
    
    for val in param_values:
        # WAŻNE: Kopia parametrów, aby nie nadpisywać ich dla kolejnych pętli
        current_params = base_params.copy()
        current_params[param_name] = val
        
        trial_scores = []
        
        # Uruchamiamy 5 prób dla jednej wartości parametru
        for i in range(runs):
            swarm = Swarm(
                num_of_particles=current_params['n'], 
                bounds=bounds, 
                inertia=current_params['inertia'], 
                cognitive_factor=current_params['c'], 
                social_factor=current_params['s']
            )
            score, _pos = swarm.run(func, iterations=current_params["iterations"])
            trial_scores.append(score)
        
        # Obliczamy statystyki dla tej wartości parametru
        trial_scores.sort()
        best_val = trial_scores[0]
        worst_val = trial_scores[-1]
        mean_val = statistics.mean(trial_scores)
        median_val = statistics.median(trial_scores)
        std_val = statistics.stdev(trial_scores) if len(trial_scores) > 1 else 0.0

        results_mean.append(mean_val)
        results_median.append(median_val)
        results_best.append(best_val)
        results_worst.append(worst_val)
        results_std.append(std_val)

        print(
            f"  {param_name}={val}: mean={mean_val:.4f}, median={median_val:.4f}, "
            f"best={best_val:.4f}, worst={worst_val:.4f}, std={std_val:.4f}"
        )
    
    return results_mean, results_median, results_best, results_worst, results_std

# --- 3. Rysowanie Wykresów ---
def plot_experiment(func_name, param_label, x_values, mean_scores, median_scores, best_scores, worst_scores,
                    std_scores, filename):
    plt.figure(figsize=(10, 6))

    # mean ± std
    plt.errorbar(
        x_values, mean_scores, yerr=std_scores, fmt='-o', linewidth=1, capsize=3, label='Średnia ± StdDev',
        ecolor='red'
    )

    # median
    plt.plot(x_values, median_scores, 'd-.', alpha=0.85, label='Mediana')

    # best / worst
    plt.plot(x_values, best_scores, 's--', alpha=0.8, label='Najlepszy (Best)')
    plt.plot(x_values, worst_scores, 'x:', alpha=0.8, label='Najgorszy (Worst)')

    plt.title(f'Wpływ: {param_label} na wynik ({func_name})')
    plt.xlabel(param_label)
    plt.ylabel('Wartość funkcji celu')
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    print(f"Wygenerowano wykres: {filename}")

# --- 4. Main ---
def main():
    os.makedirs("results", exist_ok=True)
    print("== PSO: Optymalizacja Funkcji Dwóch Zmiennych ==")
    
    while True:
        print("\nDostępne funkcje:")
        print("1. Funkcja Goldstein-Price (domyślna)")
        print("2. Funkcja Eggholder")
        print("q. Wyjście")
        
        choice = input("Wybierz funkcję (1/2/q): ")
        if choice == 'q': break
        elif choice == '2':
            func = function_eggholder
            func_name = 'Eggholder'
            bounds = (-512,512)
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

        base_path = 'results/'
        # 1. Badanie parametru inertia
        means, medians, bests, worsts, stds = run_experiments( func, func_name, bounds, "inertia",
                                                               inertia_vals, base_params, runs=5)
        plot_experiment(func_name,"Inertia (w)", inertia_vals, means, medians, bests, worsts, stds,
            f"{base_path}{func_name}_inertia.png")

        # 2. Badanie parametru n (populacja)
        means, medians, bests, worsts, stds = run_experiments(func, func_name, bounds, "n",
                                                              n_vals, base_params, runs=5)
        plot_experiment(func_name,"Liczba cząstek (n)",n_vals, means, medians, bests, worsts, stds,
            f"{base_path}{func_name}_population.png")

        # 3. Badanie parametru cognitive (c)
        means, medians, bests, worsts, stds = run_experiments(
            func, func_name, bounds, "c", components_vals, base_params, runs=5)
        plot_experiment(func_name,"Współczynnik kognitywny (c)", components_vals, means, medians,
                        bests, worsts, stds,
            f"{base_path}{func_name}_cognitive.png")

        # 4. Badanie parametru social (s)
        means, medians, bests, worsts, stds = run_experiments(func, func_name, bounds, "s",
                                                              components_vals, base_params, runs=5)
        plot_experiment(func_name,"Współczynnik społeczny (s)",components_vals,means, medians,
                        bests, worsts, stds,
            f"{base_path}{func_name}_social.png")

        # 5. Badanie liczby iteracji
        means, medians, bests, worsts, stds = run_experiments(func, func_name, bounds, "iterations",
                                                              iterations_vals, base_params, runs=5)
        plot_experiment(func_name,"Liczba iteracji",iterations_vals,means, medians, bests, worsts, stds,
            f"{base_path}{func_name}_iterations.png",
        )

if __name__ == "__main__":
    main()