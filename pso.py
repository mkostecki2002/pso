import random
from collections.abc import Sequence
from typing import Tuple, Callable

class Particle:
    def __init__(self, bounds: Tuple[float, float], inertia: float, cognitive_factor: float, social_factor: float):
        # Inicjalizacja pozycji (x, y) wewnątrz granic
        x = random.uniform(bounds[0], bounds[1])
        y = random.uniform(bounds[0], bounds[1])
        self.position = [x, y]
        
        # Inicjalizacja prędkości (wektor)
        self.velocity = [0.0, 0.0]
        
        # Parametry
        self.inertia = inertia
        self.cognitive_factor = cognitive_factor
        self.social_factor = social_factor
        
        # Najlepsze znane rozwiązanie tej cząstki
        self.best_position = [x, y]
        self.best_score = float('inf')
        self.current_score = float('inf')

    def evaluate(self, function: Callable):
        """Oblicza wartość funkcji celu i aktualizuje p_best"""
        self.current_score = function(self.position[0], self.position[1])
        
        if self.current_score < self.best_score:
            self.best_score = self.current_score
            self.best_position = self.position.copy()


    def update_position(self, global_best_position: Sequence[float], bounds: Tuple[float, float]):
        r1 = random.random()
        r2 = random.random()

        for i in range(2):
            self.velocity[i] = (
                self.inertia * self.velocity[i]
                + self.cognitive_factor * r1 * (self.best_position[i] - self.position[i])
                + self.social_factor * r2 * (global_best_position[i] - self.position[i])
            )

            self.position[i] += self.velocity[i]

            # ograniczenie - jeszcze nie wiem jak obsługiwać wyjścia poza zakres 
            self.position[i] = max(bounds[0], min(self.position[i], bounds[1]))

class Swarm:
    def __init__(self, num_of_particles: int, bounds: Tuple[float, float], 
                 inertia=0.5, cognitive_factor=1.5, social_factor=1.5):
        
        self.particles = [Particle(bounds, inertia, cognitive_factor, social_factor) for _ in range(num_of_particles)]
        self.bounds = bounds
        
        # Najlepsze rozwiązanie roju
        self.global_best_position = self.particles[0].position.copy()
        self.global_best_score = float('inf')

    def run(self, function: Callable, iterations: int):
        for _ in range(iterations):
            for p in self.particles:
                # Ocena przystosowania cząstki
                p.evaluate(function)
                
                # Aktualizacja globalnego minimum
                if p.current_score < self.global_best_score:
                    self.global_best_score = p.current_score
                    self.global_best_position = p.best_position.copy()
            
            # Aktualizacja pozycji
            for p in self.particles:
                p.update_position(self.global_best_position, self.bounds)
        
        return self.global_best_score, tuple(self.global_best_position)