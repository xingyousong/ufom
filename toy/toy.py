import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 20

np.random.seed(42)

def simulate_fo_div(subplot_indices):

    '''
    r = 10
    alpha = 0.1
    a1 = 0.5
    a2 = 1.5
    D = 0.06
    b1, b2, A = get_div_problem(r, alpha, a1, a2, D)
    '''


    alpha = 0.1
    r = 10
    a1 = 0.5#1.0
    a2 = 1.5
    b1 = 0
    b2 = 10
    A = 400#0

    x0_min = -10
    x0_max = 30

    draw_points_count = 1000
    samples_count = 10

    as_ = np.array([a1, a2])
    bs = np.array([b1, b2])

    task_draw_xs = np.linspace(-15, 30, draw_points_count)

    task1_ys = np.array([f_and_deriv(a1, b1, A, x)[0] for x in task_draw_xs])
    task2_ys = np.array([f_and_deriv(a2, b2, A, x)[0] for x in task_draw_xs])

    plt.subplot(subplot_indices[0])

    plt.plot(task_draw_xs, task1_ys, 'm--', label='task 1', linewidth=3)
    plt.plot(task_draw_xs, task2_ys, 'c', label='task 2', linewidth=3)
    plt.xlabel('$\phi$')
    plt.legend()

    sim_xs = []
    sim_vals = []
    sim_derivs = []

    for q in [0.5, 0.9]:#[0, 0.1]:

        x0 = np.random.uniform(x0_min, x0_max, size=samples_count)

        cur_sim_xs, _ = simulate_blo(r, alpha, as_, bs, A, x0, q, 10)
        cur_sim_val = get_true_blo_values(r, alpha, as_, bs, A, cur_sim_xs)[-1]
        cur_sim_derivs = get_true_blo_derivs(r, alpha, as_, bs, A, cur_sim_xs)

        sim_xs.append(cur_sim_xs[-1])
        sim_vals.append(cur_sim_val)
        sim_derivs.append(cur_sim_derivs)

    obj_draw_xs = np.linspace(-2, 9, draw_points_count)
    obj_draw_vals = get_true_blo_values(r, alpha, as_, bs, A, obj_draw_xs)

    plt.subplot(subplot_indices[1])

    plt.plot(obj_draw_xs, obj_draw_vals, 'k', label='$\mathcal{M} (\\theta)$', linewidth=2)
    plt.scatter(sim_xs[0], sim_vals[0], c='r', s=200, label='FO-BLO $\\theta^*$', zorder=10, marker='+')
    plt.scatter(sim_xs[1], sim_vals[1], c='b', s=200, label='UFO-BLO $\\theta^*$', zorder=10, marker='x')
    plt.xlabel('$\\theta$')
    plt.legend()

    plt.subplot(subplot_indices[2])

    iter_range = np.arange(0, sim_derivs[0].shape[0], 200)

    for sample_index in range(samples_count):

        if sample_index == 0:
            label1 = 'FOM'
            label2 = 'UFOM'
        else:
            label1 = None
            label2 = None

        plt.plot(iter_range, np.abs(sim_derivs[0][iter_range, sample_index]), 'r--', label=label1,
                linewidth=2)
        plt.plot(iter_range, np.abs(sim_derivs[1][iter_range, sample_index]), 'b', label=label2,
                linewidth=2)

    plt.ylim([0, 1])
    plt.xlabel('iteration index $k$')
    plt.ylabel('$| \\frac{\\partial}{\\partial \\theta} \mathcal{M}^{(r)} (\\theta_k) |$')
    plt.legend()

def simulate_bounds(subplot_indices):

    r = 10
    a1 = 0.5#1.0
    a2 = 1.5
    b1 = 0
    b2 = 10
    A = 400#0

    '''
    r = 10
    alpha = 0.1
    a1 = 0.5
    a2 = 1.5
    D = 0.06
    b1, b2, A = get_div_problem(r, alpha, a1, a2, D)
    '''

    x_min = -50#1000
    x_max = 50#1000
    x0_min = -100
    x0_max = 100

    min_alpha = 0.001
    max_alpha = 0.01

    x0_count = 500
    runs_count = 1
    xs_count = 10000
    alphas_count = 5#10

    as_ = np.array([a1, a2])
    bs = np.array([b1, b2])

    a_max = as_.max()

    M1 = 1
    L1 = a_max/2 + a_max*A
    L2 = a_max
    L3 = a_max

    alpha_range = np.exp(np.linspace(np.log(min_alpha), np.log(max_alpha), num=alphas_count))

    xs = np.linspace(x_min, x_max, xs_count)

    D2s_true = []
    V2s_true = []
    D2s = []
    V2s = []

    for alpha in alpha_range:

        _, fo_deriv1, deriv1 = blo_and_deriv(r, alpha, as_[0], bs[0], A, xs)
        _, fo_deriv2, deriv2 = blo_and_deriv(r, alpha, as_[1], bs[1], A, xs)

        D2_true = ((fo_deriv1 - deriv1)**2 + (fo_deriv2 - deriv2)**2).max()/2
        V2_true = (deriv1**2 + deriv2**2).max()/2

        common_term = L1*L2*alpha*((1 + alpha*L2)**(np.arange(r) + 1)).sum()

        D = (1 + M1)*common_term
        D2 = D**2
        V = L1 + M1*L1*((1 + alpha*L2)**r) + common_term
        V2 = V**2

        D2s_true.append(D2_true)
        V2s_true.append(V2_true)
        D2s.append(D2)
        V2s.append(V2)

    plt.subplot(subplot_indices[0])

    plt.plot(alpha_range, D2s_true, 'b', label='$\\mathbb{D}^2_{true}$')
    plt.plot(alpha_range, D2s, 'b--', label='$\\mathbb{D}^2$')
    plt.plot(alpha_range, V2s_true, 'r', label='$\\mathbb{V\\,}^2_{true}$')
    plt.plot(alpha_range, V2s, 'r--', label='$\\mathbb{V\\,}^2$')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$\\alpha$')
    plt.legend()

    qs_count = 5#10
    qs = np.linspace(0.1, 1.0, qs_count)
    delta = 0.001

    best_q_for_alphas = []

    tensor = np.ones((x0_count, runs_count, alphas_count, qs_count))

    x0 = np.random.uniform(x0_min, x0_max, size=x0_count)

    x0_all = (tensor*x0[:, None, None, None]).ravel()
    alphas_all = (tensor*alpha_range[None, None, :, None]).ravel()
    qs_all = (tensor*qs[None, None, None, :]).ravel()

    sim_xs, f_calls = simulate_blo(r, alphas_all, as_, bs, A, x0_all, qs_all, 1)
    print('Done simulation')
    sqr_derivs = get_true_blo_derivs(r, alphas_all, as_, bs, A, sim_xs)**2
    print('Done derivative computation')

    reached_delta = (sqr_derivs < delta).astype(int)
    first_indices = reached_delta.argmax(axis=0)
    first_indices[sqr_derivs[first_indices, np.arange(len(first_indices))] == 0] = sqr_derivs.shape[0] - 1

    total_calls = first_indices#f_calls[first_indices, np.arange(len(first_indices))]
    total_calls = total_calls.reshape(x0_count, runs_count, alphas_count, qs_count)
    total_calls = total_calls.mean(axis=(0, 1))#.max(axis=0)

    print(total_calls)

    best_q_for_alphas = qs[total_calls.argmin(axis=1)]
 
    plt.subplot(subplot_indices[1])

    plt.plot(alpha_range, best_q_for_alphas, label='Practice')

    plt.xscale('log')
    plt.xlabel('$\\alpha$')
    plt.ylabel('$q$')
    plt.legend()       

def get_div_problem(r, alpha, a1, a2, D):

    A_offset = 1

    b1 = 0.0
    pow_r1 = (1 - alpha*a1)**r
    pow_r2 = (1 - alpha*a2)**r
    pow_2r1 = (1 - alpha*a1)**(2*r)
    pow_2r2 = (1 - alpha*a2)**(2*r)

    b2 = np.sqrt(D*2)*2/np.abs((a1*pow_2r1 + a2*pow_2r2)*pow_r2/(a1*pow_r1 + a2*pow_r2) - pow_2r2)
    A = np.abs(b1/a1 - b2/a2) + A_offset

    '''
    fo_opt = pow_r2*b2/(a1*pow_r1 + a2*pow_r2)
    print((a1*pow_2r1 + a2*pow_2r2)*fomaml_opt/2 - b2*pow_2r2/2, np.sqrt(D*2))
    print(blo_and_deriv(r, alpha, a1, b1, A, fo_opt)[1] + blo_and_deriv(r, alpha, a2, b2, A, fo_opt)[1])
    print(blo_and_deriv(r, alpha, a1, b1, A, fo_opt)[2]/2 + blo_and_deriv(r, alpha, a2, b2, A, fo_opt)[2]/2)
    '''

    return b1, b2, A

def get_true_blo_values(r, alpha, as_, bs, A, xs):

    return (blo_and_deriv(r, alpha, as_[0], bs[0], A, xs)[0] + blo_and_deriv(r, alpha, as_[1], bs[1], A, xs)[0])/2

def get_true_blo_derivs(r, alpha, as_, bs, A, xs):

    return (blo_and_deriv(r, alpha, as_[0], bs[0], A, xs)[2] + blo_and_deriv(r, alpha, as_[1], bs[1], A, xs)[2])/2

def simulate_blo(r, alpha, as_, bs, A, x, q, start_step):

    iter_count = 10000

    xs = []
    f_calls = []

    for iter_index in range(iter_count):

        if iter_index%1000 == 0:
            print(iter_index)

        random_index = np.random.choice(2, size=x.shape)
        a = as_[random_index]
        b = bs[random_index]

        _, fo_deriv, deriv = blo_and_deriv(r, alpha, a, b, A, x)

        step = start_step/(iter_index + 1)

        coin_flips = (np.random.uniform(size=x.shape) < q).astype(float)
        update = (fo_deriv*(1 - 1/(q + 1e-8)) + deriv/(q + 1e-8))*coin_flips + fo_deriv*(1 - coin_flips)

        x = x - step*update

        xs.append(x)
        f_calls.append(r + 1 + coin_flips*r)


    result = np.array(xs)
    f_calls = np.cumsum(f_calls, axis=0)

    return result, f_calls

def blo_and_deriv(r, alpha, a, b, A, x):

    xs = [x]

    for j in range(r):
        new_x = xs[-1] - alpha*f_and_deriv(a, b, A, xs[-1])[1]
        xs.append(new_x)

    blo, fo_deriv, _ = f_and_deriv(a, b, A, xs[-1])
    deriv = fo_deriv

    for j in range(r - 1, -1, -1):
        deriv = (1 - alpha*f_and_deriv(a, b, A, xs[j])[2])*deriv

    return blo, fo_deriv, deriv

def f_and_deriv(a, b, A, x):

    z = np.abs(x - b/a)

    f = np.zeros_like(x)
    deriv = np.zeros_like(x)
    hess = np.zeros_like(x)

    mask = (z <= A)
    f[mask] = (a*z*z/2)[mask]
    deriv[mask] = (a*x - b)[mask]
    hess[mask] = (a*np.ones_like(x))[mask]

    mask = (z > A)&(z <= A + 1)
    f[mask] = (-a*((z - A)**3)/6 + a*((z - A)**2)/2 + a*A*z - a*A*A/2)[mask]
    deriv[mask] = ((-a*((z - A)**2)/2 + a*z)*np.sign(x - b/a))[mask]
    hess[mask] = (-a*z + a + a*A)[mask]

    mask = (z > A + 1)
    f[mask] = ((a/2 + a*A)*z - a/6 - a*A*A/2 - a*A/2)[mask]
    deriv[mask] = ((a/2 + a*A)*np.sign(x - b/a))[mask]

    return f, deriv, hess

if __name__ == '__main__':

    fig = plt.figure(figsize=(30, 10))

    #simulate_fo_div([131, 132, 133])
    simulate_bounds([121, 122])

    fig.tight_layout()
    plt.savefig('toy.pdf', bbox_inches='tight')
