import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid, quad

class LogAestheticCurve:
    """
    Клас для моделювання та побудови лог-естетичних кривих 
    за степеневою моделлю натурального рівняння k(s) = sign * lambda_ * s^alpha.
    """
    def __init__(self, alpha=1.0, lambda_=1.0, sign=1.0):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.sign = np.sign(sign)

    def calc_L_for_angle(self, theta_target):
        """
        Аналітичний розрахунок необхідної довжини дуги L для досягнення заданого кута.
        Theta = \int \lambda s^\alpha ds = \lambda * L^(\alpha+1) / (\alpha+1)
        """
        theta_abs = abs(theta_target)
        L = (theta_abs * (self.alpha + 1.0) / self.lambda_) ** (1.0 / (self.alpha + 1.0))
        return L

    def generate_curve(self, L, N=1000):
        """
        Чисельна побудова кривої методом кумулятивних трапецій.
        Повертає масиви довжини дуги, кривизни, кута та декартових координат.
        """
        s = np.linspace(0, L, N)
        
        # Обчислення кривизни
        kappa = self.sign * self.lambda_ * (s ** self.alpha)
        
        # Інтегрування для знаходження кута theta(s)
        theta = cumulative_trapezoid(kappa, s, initial=0.0)
        
        # Інтегрування для знаходження координат x(s), y(s)
        x = cumulative_trapezoid(np.cos(theta), s, initial=0.0)
        y = cumulative_trapezoid(np.sin(theta), s, initial=0.0)
        
        return s, kappa, theta, x, y

def experiment_comparison():
    """Експеримент 5: Порівняння клотоїди (alpha=1) та LEC (alpha=2) для 90 градусів."""
    theta_target = np.pi / 2.0
    
    # Побудова Клотоїди
    clothoid = LogAestheticCurve(alpha=1.0, lambda_=1.0)
    L_C = clothoid.calc_L_for_angle(theta_target)
    s_C, k_C, th_C, x_C, y_C = clothoid.generate_curve(L_C, N=2000)
    
    # Побудова Лог-естетичної кривої (alpha=2)
    lec = LogAestheticCurve(alpha=2.0, lambda_=1.0)
    L_LEC = lec.calc_L_for_angle(theta_target)
    s_LEC, k_LEC, th_LEC, x_LEC, y_LEC = lec.generate_curve(L_LEC, N=2000)
    
    print("=== Експеримент 5: Порівняння для кута 90 градусів (pi/2) ===")
    print(f"Клотоїда (alpha=1): L = {L_C:.4f}, kappa_max = {k_C[-1]:.4f}, End Pt = ({x_C[-1]:.4f}, {y_C[-1]:.4f})")
    print(f"LEC      (alpha=2): L = {L_LEC:.4f}, kappa_max = {k_LEC[-1]:.4f}, End Pt = ({x_LEC[-1]:.4f}, {y_LEC[-1]:.4f})")

def experiment_convergence():
    """Експеримент 6: Дослідження збіжності методу інтегрування."""
    theta_target = np.pi / 2.0
    clothoid = LogAestheticCurve(alpha=1.0)
    L_C = clothoid.calc_L_for_angle(theta_target)
    
    # Генерація високоточного референсного розв'язку (еталон)
    _, _, _, x_ref, y_ref = clothoid.generate_curve(L_C, N=100000)
    ref_pt = np.array([x_ref[-1], y_ref[-1]])
    
    Ns = 
    print("\n=== Експеримент 6: Збіжність чисельного методу (трапецій) ===")
    for N in Ns:
        _, _, _, x, y = clothoid.generate_curve(L_C, N=N)
        test_pt = np.array([x[-1], y[-1]])
        error = np.linalg.norm(ref_pt - test_pt)
        print(f"Кількість вузлів N = {N:4d} | Евклідова похибка = {error:.2e} метрів")

if __name__ == "__main__":
    experiment_comparison()
    experiment_convergence()