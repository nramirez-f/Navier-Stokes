from aprox_spacial import *

def intermediate_velocity_euler(u , v, Re, mesh, dt):

    Fu = conv_diff_u(u, v, Re, mesh)
    Fv = conv_diff_v(u, v, Re, mesh)

    u_star = u + dt * Fu
    v_star = v + dt * Fv

    return (u_star, v_star, Fu, Fv)

def intermediate_velocity_adamsB(u , v, Re, mesh, Fu_old, Fv_old, dt):

    Fu = conv_diff_u(u, v, Re, mesh)
    Fv = conv_diff_v(u, v, Re, mesh)

    u_star = u + dt * (1.5 * Fu - 0.5 * Fu_old)
    v_star = v + dt * (1.5 * Fv - 0.5 * Fv_old)

    return (u_star, v_star, Fu, Fv)

