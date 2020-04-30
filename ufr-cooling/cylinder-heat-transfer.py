def calc_prandtl(kinematic_viscosity, thermal_diffusivity):
    return kinematic_viscosity / thermal_diffusivity

def calc_kinematic_viscosity(dynamic_viscosity, density):
    return dynamic_viscosity / density

def calc_dynamic_viscosity(kinematic_viscosity, density):
    return kinematic_viscosity * density

def calc_thermal_diffusivity(thermal_conductivity, density, specific_heat_const_pressure):
    return thermal_conductivity / (density * specific_heat_const_pressure)

def calc_reynolds(velocity, length_scale, dynamic_viscosity):
    return (velocity * length_scale) / dynamic_viscosity

def calc_nusselt(reynolds, prandtl):
    A = 0.62 * (reynolds ** 0.5)
    B = (1 + (0.4 / prandtl) ** (2 / 3)) ** (1 / 4)
    C = (1 + (reynolds / 282000) ** (5 / 8)) ** (-4 / 5)
    return 0.3 + (A / B) * C

def plot_nusselt():
    velocities = np.linspace(0, 500, 100)
    nusselt_list = np.zeros_like(velocities)
    for i in range(len(velocities)):
        reynolds = calc_reynolds(velocities[i], diameter, dynamic_viscosity)
        nusselt_list[i] = calc_nusselt(reynolds, prandtl)
    plt.figure()
    plt.plot(velocities, nusselt_list)
    plt.xlabel(r'Air Velocity [$\mathrm{ms}^{-1}$]')
    plt.ylabel(r'Nusselt Number [dimensionless]')
    #plt.show()
    plt.savefig('nusselt-vs-air-velocity.svg', format='svg')
    plt.savefig('nusselt-vs-air-velocity.png', format='png')

def plot_deltaT_vs_velocity():
    velocities = np.linspace(5, 200, 50)
    q = 70 # [W]
    deltaT_list = np.zeros_like(velocities)
    plt.figure()
    for q in range(50, 200, 50):
        for i in range(len(velocities)):
            reynolds = calc_reynolds(velocities[i], diameter, dynamic_viscosity)
            nus = calc_nusselt(reynolds, prandtl)
            deltaT_list[i] = q / (0.025 * np.pi * nus)
        
        plt.plot(velocities, deltaT_list, label=f'{q} W')
    plt.xlabel(r'Air Velocity [$\mathrm{ms}^{-1}$]')
    plt.ylabel(r'Delta T [K]')
    #plt.title('q = ' + str(q) + 'W')
    plt.legend()
    plt.savefig('deltaT-vs-air-velocity.svg', format='svg')
    plt.savefig('deltaT-vs-air-velocity.png', format='png')
    #plt.show()

import numpy as np
import matplotlib.pyplot as plt

plt.close('all') 

density = 1.29 # [kg][m^-3]
kinematic_viscosity = 13e-6 # [m^2][s^-1]
thermal_conductivity = 0.024 # [W][m^-1][K_-1]

specific_heat_const_pressure = 1006 # [J][kg^-1][K^-1]
#https://www.engineeringtoolbox.com/air-specific-heat-capacity-d_705.html

diameter = 0.025 # [m]
T_w = 273.15 + 30 # [K]
T_inf = 273.15 + 20 # [K]
axial_length = 0.03 # [m]

dynamic_viscosity = calc_dynamic_viscosity(kinematic_viscosity, density) # [kg][m^-1][s^-1]

velocity = 10 # [m][s^-1]

reynolds = calc_reynolds(velocity, diameter, dynamic_viscosity) # [dimensionless]

print(f'Reynolds number = {reynolds}')

thermal_diffusivity = calc_thermal_diffusivity(thermal_conductivity, density, specific_heat_const_pressure) # [m^2][s^-1]

prandtl = calc_prandtl(kinematic_viscosity, thermal_diffusivity) # [dimensionless]

print(f'Prandtl number = {prandtl}')

nusselt = calc_nusselt(reynolds, prandtl) # [dimensionless]

heat_flux = np.pi * axial_length * (T_w - T_inf) * nusselt # [W][m^-1]

print(f'Heat flux  = {heat_flux} W')

plot_nusselt()

plot_deltaT_vs_velocity()