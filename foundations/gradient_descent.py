class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        x_old = init
        if iterations == 0:
            return init
        if x_old != 0:
            x_new = int()
            for it in range(iterations):
                grad = 2*x_old
                x_new = x_old - learning_rate*grad
                if math.isclose(x_new,x_old):
                    break
                x_old = x_new
            return float(round(x_new, 5))
                
        return float(round(x_old, 5))

