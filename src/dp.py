# %%
import numpy as np

# Parameters
prices = np.array([10, 20, 30, 40, 30, 20, 10])  # c/kWh, example price series
T = len(prices)
capacity = 2.0  # kWh
max_charge = 1.0  # kW per timestep
max_discharge = 1.0  # kW per timestep
eff = 0.9  # round-trip efficiency
soc_grid = np.linspace(0, capacity, 11)  # 11 discrete SoC levels

# DP table: value[t, soc_idx] = max profit from t to end, starting at soc
value = np.zeros((T + 1, len(soc_grid)))
action = np.zeros((T, len(soc_grid)))  # -1: discharge, 0: idle, 1: charge

for t in range(T - 1, -1, -1):
    for i, soc in enumerate(soc_grid):
        # Try all actions: charge, discharge, idle
        best = -np.inf
        best_a = 0
        # Charge
        if soc + max_charge <= capacity:
            soc_next = soc + max_charge * eff
            j = np.argmin(np.abs(soc_grid - soc_next))
            profit = -prices[t] * max_charge  # pay to charge
            total = profit + value[t + 1, j]
            if total > best:
                best = total
                best_a = 1
        # Discharge
        if soc - max_discharge >= 0:
            soc_next = soc - max_discharge / eff
            j = np.argmin(np.abs(soc_grid - soc_next))
            profit = prices[t] * max_discharge  # earn from discharge
            total = profit + value[t + 1, j]
            if total > best:
                best = total
                best_a = -1
        # Idle
        total = value[t + 1, i]
        if total > best:
            best = total
            best_a = 0
        value[t, i] = best
        action[t, i] = best_a

# Recover optimal path
soc = 0.0  # start empty
soc_path = [soc]
actions = []
for t in range(T):
    i = np.argmin(np.abs(soc_grid - soc))
    a = action[t, i]
    actions.append(a)
    if a == 1:
        soc = min(soc + max_charge * eff, capacity)
    elif a == -1:
        soc = max(soc - max_discharge / eff, 0)
    soc_path.append(soc)

print("Optimal actions:", actions)
print("SoC path:", soc_path)
print("Max profit:", value[0, 0])

# %%
