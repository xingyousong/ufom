import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 20

np.random.seed(42)

def simulate():

    r = 10
    alpha = 0.1
    a1 = 0.5
    a2 = 1.5
    D = 0.06

    A_offset = 1
    draw_points_count = 1000
    samples_count = 10

    b1 = 0.0
    pow_r1 = (1 - alpha*a1)**r
    pow_r2 = (1 - alpha*a2)**r
    pow_2r1 = (1 - alpha*a1)**(2*r)
    pow_2r2 = (1 - alpha*a2)**(2*r)

    b2 = np.sqrt(D*2)*2/np.abs((a1*pow_2r1 + a2*pow_2r2)*pow_r2/(a1*pow_r1 + a2*pow_r2) - pow_2r2)
    A = np.abs(b1/a1 - b2/a2) + A_offset

    '''
    fomaml_opt = pow_r2*b2/(a1*pow_r1 + a2*pow_r2)
    print((a1*pow_2r1 + a2*pow_2r2)*fomaml_opt/2 - b2*pow_2r2/2, np.sqrt(D*2))
    print(maml_and_deriv(r, alpha, a1, b1, A, fomaml_opt)[1] + maml_and_deriv(r, alpha, a2, b2, A, fomaml_opt)[1])
    print(maml_and_deriv(r, alpha, a1, b1, A, fomaml_opt)[2]/2 + maml_and_deriv(r, alpha, a2, b2, A, fomaml_opt)[2]/2)
    '''

    task_draw_xs = np.linspace(-15, 30, draw_points_count)

    task1_ys = np.array([f_and_deriv(a1, b1, A, x)[0] for x in task_draw_xs])
    task2_ys = np.array([f_and_deriv(a2, b2, A, x)[0] for x in task_draw_xs])

    fig = plt.figure(figsize=(18, 5))

    plt.subplot(131)

    plt.plot(task_draw_xs, task1_ys, 'm--', label='task 1', linewidth=3)
    plt.plot(task_draw_xs, task2_ys, 'c', label='task 2', linewidth=3)
    plt.xlabel('$\phi$')
    plt.legend()

    sim_xs = []
    sim_vals = []
    sim_derivs = []

    for q in [0, 0.1]:

        sim_xs.append([])
        sim_vals.append([])
        sim_derivs.append([])

        for sample_index in range(samples_count):

            x0 = np.random.uniform(-10, 30)
            cur_sim_xs = simulate_maml(r, alpha, [a1, a2], [b1, b2], A, x0, q)
            cur_sim_val = get_true_maml_values(r, alpha, [a1, a2], [b1, b2], A, cur_sim_xs[:-1])[-1]
            cur_sim_derivs = get_true_maml_derivs(r, alpha, [a1, a2], [b1, b2], A, cur_sim_xs)

            sim_xs[-1].append(cur_sim_xs[-1])
            sim_vals[-1].append(cur_sim_val)
            sim_derivs[-1].append(cur_sim_derivs)

    obj_draw_xs = np.linspace(-2, 9, draw_points_count)
    obj_draw_vals = get_true_maml_values(r, alpha, [a1, a2], [b1, b2], A, obj_draw_xs)

    plt.subplot(132)

    plt.plot(obj_draw_xs, obj_draw_vals, 'k', label='$\mathcal{M} (\\theta)$', linewidth=2)
    plt.scatter(sim_xs[0], sim_vals[0], c='r', s=200, label='FOMAML $\\theta^*$', zorder=10, marker='+')
    plt.scatter(sim_xs[1], sim_vals[1], c='b', s=200, label='UFO-MAML $\\theta^*$', zorder=10, marker='x')
    plt.xlabel('$\\theta$')
    plt.legend()

    plt.subplot(133)

    iter_range = np.arange(0, len(sim_derivs[0][0]), 200)

    for sample_index in range(samples_count):

        if sample_index == 0:
            label1 = 'FOMAML'
            label2 = 'UFO-MAML'
        else:
            label1 = None
            label2 = None

        plt.plot(iter_range, np.abs(sim_derivs[0][sample_index])[iter_range], 'r--', label=label1,
                linewidth=2)
        plt.plot(iter_range, np.abs(sim_derivs[1][sample_index])[iter_range], 'b', label=label2,
                linewidth=2)

    plt.ylim([0, 1])
    plt.xlabel('iteration index $k$')
    plt.ylabel('$| \\nabla_{\\theta_k} \mathcal{M} (\\theta_k) |$')
    plt.legend()

    fig.tight_layout()

    plt.savefig('toy.pdf', bbox_inches='tight')

def get_true_maml_values(r, alpha, as_, bs, A, xs):

    return [np.mean([maml_and_deriv(r, alpha, a, b, A, x)[0] for a, b in zip(as_, bs)]) for x in xs]

def get_true_maml_derivs(r, alpha, as_, bs, A, xs):

    return [np.mean([maml_and_deriv(r, alpha, a, b, A, x)[2] for a, b in zip(as_, bs)]) for x in xs]

def simulate_maml(r, alpha, as_, bs, A, x, q):

    iter_count = 10000
    start_step = 10

    xs = []

    for iter_index in range(iter_count):

        xs.append(x)

        random_index = np.random.choice(2)
        a = as_[random_index]
        b = bs[random_index]

        _, fomaml_deriv, deriv = maml_and_deriv(r, alpha, a, b, A, x)

        step = start_step/(iter_index + 1)

        if np.random.uniform() < q:
            x -= step*(fomaml_deriv*(1 - 1/q) + deriv/q)
        else:
            x -= step*fomaml_deriv

    return xs

def maml_and_deriv(r, alpha, a, b, A, x):

    xs = [x]

    for j in range(r):
        new_x = xs[-1] - alpha*f_and_deriv(a, b, A, xs[-1])[1]
        xs.append(new_x)

    maml_value, fomaml_deriv, _ = f_and_deriv(a, b, A, xs[-1])
    deriv = fomaml_deriv

    for j in range(r - 1, -1, -1):
        deriv = (1 - alpha*f_and_deriv(a, b, A, xs[j])[2])*deriv

    return maml_value, fomaml_deriv, deriv

def f_and_deriv(a, b, A, x):

    z = np.abs(x - b/a)

    if z <= A:
        return a*z*z/2, a*x - b, a
    elif z <= A + 1:
        return -a*((z - A)**3)/6 + a*((z - A)**2)/2 + a*A*z - a*A*A/2, (-a*((z - A)**2)/2 + \
                a*z)*np.sign(x - b/a), -a*z + a + a*A
    else:
        return (a/2 + a*A)*z - a/6 - a*A*A/2 - a*A/2, (a/2 + a*A)*np.sign(x - b/a), 0


if __name__ == '__main__':

    simulate()
