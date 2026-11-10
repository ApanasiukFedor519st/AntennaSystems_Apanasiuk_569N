import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1

# ==========================================
# ВИХІДНІ ДАНІ (Варіант №2)
# ==========================================
WAVELENGTH = 3.2    # см (довжина хвилі lambda)
H_DIST = 8.0        # см (відстань між стрижнями)
L_ROD = 23.7        # см (довжина стрижня)
D_AVG = 2.3         # см (середній діаметр стрижня)
EPS_R = 2.2         # відносна діелектрична проникність

# ==========================================
# 1. ОСНОВНІ РОЗРАХУНКИ ПАРАМЕТРІВ
# ==========================================
k = 2 * np.pi / WAVELENGTH
xi_opt = 1 + WAVELENGTH / (2 * L_ROD)
A_param = np.pi * L_ROD / WAVELENGTH
u0 = A_param * (xi_opt - 1)

print("-" * 55)
print(f"РОЗРАХУНОК ПАРАМЕТРІВ ДСА (ВАРІАНТ 2)")
print("-" * 55)
print(f"Довжина хвилі: {WAVELENGTH} см | Відстань h: {H_DIST} см | Довжина L: {L_ROD} см")
print(f"Оптимальний коеф. уповільнення (xi): {xi_opt:.4f}")
print(f"Спрямованість 1-стрижневої (D1): ~ {10 * np.log10(7.5 * L_ROD / WAVELENGTH):.2f} дБ")
print(f"Спрямованість 2-стрижневої (D2): ~ {10 * np.log10(15 * L_ROD / WAVELENGTH):.2f} дБ")

# ==========================================
# 2. ВЕКТОРИЗОВАНИЙ РОЗРАХУНОК ДС (0..90 град)
# ==========================================
# Масив кутів від 0.0001 до 90 градусів (у радіанах)
theta = np.linspace(0.0001, np.pi / 2, 5000)
theta_deg = np.degrees(theta)

# Функція Лямбда 1-го порядку (через вбудований Бессель зі scipy)
def lambda_func(x):
    return np.where(np.abs(x) < 1e-9, 1.0, 2 * j1(x) / x)

# Множник стрижня
v_val = xi_opt - np.cos(theta)
F_B = np.abs((xi_opt - 1) / np.sin(u0) * np.sin(A_param * v_val) / v_val)

# Множник одиночного випромінювача
x_arg = np.pi * D_AVG / WAVELENGTH * np.sin(theta)
L1 = np.abs(lambda_func(x_arg))

F_1H = L1
F_1E = L1 * np.cos(theta)

# Множник ґрат (для 2 стрижнів)
F_C = np.abs(np.cos(np.pi * H_DIST / WAVELENGTH * np.sin(theta)))

# Підсумкові ненормовані значення
F1H_raw = F_1H * F_B
F1E_raw = F_1E * F_B
F2H_raw = F1H_raw * F_C
F2E_raw = F1E_raw * F_C

# Нормування (масштабування від 0 до 1)
F1H = F1H_raw / np.max(F1H_raw)
F1E = F1E_raw / np.max(F1E_raw)
F2H = F2H_raw / np.max(F2H_raw)
F2E = F2E_raw / np.max(F2E_raw)

# ==========================================
# 3. АНАЛІЗ ХАРАКТЕРИСТИК (Ширина, екстремуми)
# ==========================================
def get_hpbw(angles, pattern):
    """Ширина діаграми на рівні 0.707 (половинна потужність)"""
    idx = np.where(pattern < 0.707)[0][0]
    t1, t2 = angles[idx - 1], angles[idx]
    p1, p2 = pattern[idx - 1], pattern[idx]
    th = t1 + (0.707 - p1) * (t2 - t1) / (p2 - p1) # Лінійна інтерполяція
    return 2 * th, th

def get_extrema(angles, pattern):
    """Пошук мінімумів (нулів) та максимумів (бічних пелюсток)"""
    mins_t, mins_val, maxs_t, maxs_val = [], [], [], []
    for i in range(1, len(pattern) - 1):
        if pattern[i] < pattern[i - 1] and pattern[i] < pattern[i + 1]:
            mins_t.append(angles[i]); mins_val.append(pattern[i])
        if pattern[i] > pattern[i - 1] and pattern[i] > pattern[i + 1] and pattern[i] > 0.001:
            maxs_t.append(angles[i]); maxs_val.append(pattern[i])
    return mins_t, mins_val, maxs_t, maxs_val

def print_extrema_table(name, extrema_data):
    mins_t, mins_v, maxs_t, maxs_v = extrema_data
    rows = max(len(mins_t), len(maxs_t))
    print(f"\n[{name}] - Екстремуми ДС")
    print("-" * 55)
    print(f"| № | Нуль (град) | F(нуля) | Пік (град) | F(піка) |")
    print("-" * 55)
    for i in range(rows):
        z_ang = f"{mins_t[i]:.2f}" if i < len(mins_t) else "  -  "
        z_val = f"{mins_v[i]:.4f}" if i < len(mins_v) else "  -  "
        p_ang = f"{maxs_t[i]:.2f}" if i < len(maxs_t) else "  -  "
        p_val = f"{maxs_v[i]:.4f}" if i < len(maxs_v) else "  -  "
        print(f"| {i+1} |    {z_ang:>5}    | {z_val:>7} |    {p_ang:>5}   | {p_val:>7} |")
    print("-" * 55)

sgp_1H, _ = get_hpbw(theta_deg, F1H)
sgp_1E, _ = get_hpbw(theta_deg, F1E)
sgp_2H, _ = get_hpbw(theta_deg, F2H)
sgp_2E, _ = get_hpbw(theta_deg, F2E)

print(f"\nШирина головної пелюстки (рівень 0.707):")
print(f"• Однострижнева: H = {sgp_1H:.2f}°, E = {sgp_1E:.2f}°")
print(f"• Двострижнева:  H = {sgp_2H:.2f}°, E = {sgp_2E:.2f}°")

print_extrema_table("ОДНОСТРИЖНЕВА ДСА (ПЛОЩИНА H)", get_extrema(theta_deg, F1H))
print_extrema_table("ДВОСТРИЖНЕВА ДСА (ПЛОЩИНА H)", get_extrema(theta_deg, F2H))

# ==========================================
# 4. ВІЗУАЛІЗАЦІЯ (ПРЕМІУМ-СТИЛЬ)
# ==========================================
plt.rcParams.update({
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.color': '#dee2e6',
    'grid.linestyle': '--',
    'font.family': 'sans-serif'
})

# Нова колірна палітра для графіків (фіолетовий, смарагдовий, бордовий, помаранчевий)
c_1H, c_1E = '#4a235a', '#117a65'
c_2H, c_2E = '#900c3f', '#d35400'


# --- Лінійний графік ---
fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=120)
ax1.plot(theta_deg, F1H, lw=2.2, color=c_1H, label=r'Однострижнева ($H$)')
ax1.plot(theta_deg, F1E, lw=2.2, color=c_1E, label=r'Однострижнева ($E$)')
ax1.plot(theta_deg, F2H, lw=2.5, color=c_2H, ls=':', label=r'Синфазна 2-ел ($H$)')
ax1.plot(theta_deg, F2E, lw=2.5, color=c_2E, ls=':', label=r'Синфазна 2-ел ($E$)')

mins_x, mins_y, _, _ = get_extrema(theta_deg, F1H)
ax1.plot(mins_x, mins_y, "o", markersize=4, color="black", label="$\\theta_{min}$")
for mx, my in zip(mins_x, mins_y):
    ax1.vlines(x=mx, ymin=0, ymax=my, colors='green', linestyles='dashed', linewidth=0.5)
    ax1.annotate(f'({mx:.1f}$^\\circ$)', xy=(mx, my), xytext=(mx - 3, my + 0.05), fontsize=6)

# Напівпрозора заливка площі під головною пелюсткою однострижневої антени (H)
ex_1H = get_extrema(theta_deg, F1H)
ax1.fill_between(theta_deg, 0, F1H, where=(theta_deg <= ex_1H[0][0]), color=c_1H, alpha=0.1)

ax1.axhline(0.707, color='black', lw=1, ls='--', alpha=0.5)
ax1.text(2, 0.72, 'Рівень 0.707', fontsize=9, color='#555555')

ax1.set_xlim(0, 90)
ax1.set_ylim(0, 1.05)
ax1.set_xticks(np.arange(0, 95, 5))
ax1.set_yticks(np.arange(0, 1.1, 0.1))
ax1.set_xlabel('Кут $\\theta$, градуси', fontsize=12, fontweight='bold')
ax1.set_ylabel('Нормована амплітуда $|F(\\theta)|$', fontsize=12, fontweight='bold')
ax1.set_title('Нормовані ДС діелектричної стрижневої антени (Вар. 2)', fontsize=14, pad=15)
ax1.legend(frameon=True, shadow=True, fancybox=True, fontsize=10)

plt.tight_layout()
fig1.savefig('DSA_Linear.png')
plt.show()

# --- Радіальний (полярний) графік ---
fig2 = plt.figure(figsize=(8, 8), dpi=120)
ax2 = plt.subplot(111, projection='polar')

# Відображаємо масив кутів для симетрії
theta_full = np.concatenate((-theta[::-1], theta))
F1H_full = np.concatenate((F1H[::-1], F1H))
F1E_full = np.concatenate((F1E[::-1], F1E))
F2H_full = np.concatenate((F2H[::-1], F2H))
F2E_full = np.concatenate((F2E[::-1], F2E))

ax2.plot(theta_full, F1H_full, lw=2, color=c_1H, label='Однострижнева (H)')
ax2.plot(theta_full, F1E_full, lw=2, color=c_1E, label='Однострижнева (E)')
ax2.plot(theta_full, F2H_full, lw=2.5, color=c_2H, ls='--', label='Двострижнева (H)')
ax2.plot(theta_full, F2E_full, lw=2.5, color=c_2E, ls='--', label='Двострижнева (E)')

ax2.fill_between(theta_full, 0, F1H_full, color=c_1H, alpha=0.08)

ax2.set_theta_zero_location("N")
ax2.set_theta_direction(-1)
ax2.set_thetamin(-90)
ax2.set_thetamax(90)
ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax2.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color='grey', size=8)

ax2.set_title('Просторові ДС ДСА (Радіальний вигляд)', fontsize=14, pad=20, fontweight='bold')
ax2.legend(loc='lower left', bbox_to_anchor=(-0.1, -0.1), fancybox=True, shadow=True)

plt.tight_layout()
fig2.savefig('DSA_Polar.png')
plt.show()