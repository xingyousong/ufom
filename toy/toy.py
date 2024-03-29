import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.size'] = 20

np.random.seed(42)

def simulate_fo_div():

    r = 10
    alpha = 0.1
    a1 = 0.5
    a2 = 1.5
    D = 0.06
    b1, b2, A = get_div_problem(r, alpha, a1, a2, D)

    print(b1, b2, A)

    x0_min = -10
    x0_max = 30

    draw_points_count = 1000
    samples_count = 5

    as_ = np.array([a1, a2])
    bs = np.array([b1, b2])

    task_draw_xs = np.linspace(-15, 30, draw_points_count)

    task1_ys = np.array([f_and_deriv(a1, b1, A, x)[0] for x in task_draw_xs])
    task2_ys = np.array([f_and_deriv(a2, b2, A, x)[0] for x in task_draw_xs])
 
    fig = plt.figure(figsize=(6, 5))

    plt.plot(task_draw_xs, task1_ys, 'm--', label='$\\mathcal{T\\,}^{(1)}$', linewidth=3)
    plt.plot(task_draw_xs, task2_ys, 'c', label='$\\mathcal{T\\,}^{(2)}$', linewidth=3)
    plt.xlabel('$\\phi$')
    plt.legend()
 
    fig.tight_layout()
    plt.savefig('toy_tasks.pdf', bbox_inches='tight')

    sim_xs = []
    sim_vals = []
    sim_derivs = []

    for q in [0, 0.1]:

        x0 = np.random.uniform(x0_min, x0_max, size=samples_count)

        cur_sim_xs, _ = simulate_blo(r, alpha, as_, bs, A, x0, q, 10, 10000)
        cur_sim_val = get_true_blo_values(r, alpha, as_, bs, A, cur_sim_xs)[-1]
        cur_sim_derivs = get_true_blo_derivs(r, alpha, as_, bs, A, cur_sim_xs)

        sim_xs.append(cur_sim_xs[-1])
        sim_vals.append(cur_sim_val)
        sim_derivs.append(cur_sim_derivs)

    obj_draw_xs = np.linspace(-2, 9, draw_points_count)
    obj_draw_vals = get_true_blo_values(r, alpha, as_, bs, A, obj_draw_xs)

    fig = plt.figure(figsize=(6, 5))

    plt.plot(obj_draw_xs, obj_draw_vals, 'k', label='$\mathcal{M}^{(r)} (\\theta)$', linewidth=2)
    plt.scatter(sim_xs[0], sim_vals[0], c='r', s=200, label='FOM $\\theta^*$', zorder=10, marker='+')
    plt.scatter(sim_xs[1], sim_vals[1], c='b', s=200, label='UFOM $\\theta^*$', zorder=10, marker='x')
    plt.xlabel('$\\theta$')
    plt.legend()

    fig.tight_layout()
    plt.savefig('toy_loss.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(6, 5))

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

    fig.tight_layout()
    plt.savefig('toy_curve.pdf', bbox_inches='tight')

def simulate_curves():

    r = 10
    a1 = 0.5
    a2 = 1.5
    b1 = 0
    b2 = 10
    A = 400

    x0_min = -50
    x0_max = 50

    alpha = 0.01

    x0_count = 1000

    iter_count = 20000

    as_ = np.array([a1, a2])
    bs = np.array([b1, b2])

    x_min = -50
    x_max = 50
    xs_count = 10000

    xs = np.linspace(x_min, x_max, xs_count)

    _, fo_deriv1, deriv1 = blo_and_deriv(r, alpha, as_[0], bs[0], A, xs)
    _, fo_deriv2, deriv2 = blo_and_deriv(r, alpha, as_[1], bs[1], A, xs)

    D2_true = ((fo_deriv1 - deriv1)**2 + (fo_deriv2 - deriv2)**2).max()/2
    V2_true = (deriv1**2 + deriv2**2).max()/2

    a = r*(V2_true - D2_true)
    b = D2_true*0.5*r
    c = -D2_true*0.5*(r + 1)

    q1 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    q2 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)

    if q1 > 0 and q1 < 1:
        q_star = q1
    else:
        q_star = q2

    qs_count = 3
    qs = np.array([0.0, q_star, 1.0])

    tensor = np.ones((x0_count, qs_count))

    x0 = np.random.uniform(x0_min, x0_max, size=x0_count)

    x0_all = (tensor*x0[:, None]).ravel()
    qs_all = (tensor*qs[None, :]).ravel()

    #sim_xs, f_calls = simulate_blo(r, alpha, as_, bs, A, x0_all, qs_all, 10, iter_count)
    print('Done simulation')
    #deriv_norms = np.abs(get_true_blo_derivs(r, alpha, as_, bs, A, sim_xs))
    print('Done derivative computation')

    #np.savez('toy_qcurves', sim_xs=sim_xs, f_calls=f_calls, deriv_norms=deriv_norms)
    data = np.load('toy_qcurves.npz')
    sim_xs, f_calls, deriv_norms = data['sim_xs'], data['f_calls'], data['deriv_norms']

    deriv_norms = deriv_norms.reshape(-1, x0_count, qs_count)
    deriv_norm_means = deriv_norms.mean(axis=1)
    deriv_norm_stds = deriv_norms.std(axis=1)

    f_calls = f_calls.reshape(-1, x0_count, qs_count)
    f_calls = f_calls.mean(axis=1)

    fig = plt.figure(figsize=(6, 5))

    colors = []

    for index in range(qs_count):

        iters_to_plot = int(iter_count*(r + 1 + r*qs[0])/(r + 1 + r*qs[index]))

        p = plt.plot(f_calls[:iters_to_plot, index], deriv_norm_means[:iters_to_plot, index],
                label='q={:.2f}'.format(qs[index]), linewidth=3)
        color = p[0].get_color()
        lower = deriv_norm_means[:, index] - deriv_norm_stds[:, index]
        upper = deriv_norm_means[:, index] + deriv_norm_stds[:, index]
        plt.fill_between(f_calls[:iters_to_plot, index], lower[:iters_to_plot], upper[:iters_to_plot],
                color=color, alpha=0.1)

    plt.yscale('log') 
    plt.ylim([None, 0.4])
    plt.xlabel('time')
    plt.ylabel('$| \\frac{\\partial}{\\partial \\theta} \mathcal{M}^{(r)} (\\theta_k) |$')
    plt.legend()

    fig.tight_layout()
    plt.savefig('toy_qcurve.pdf', bbox_inches='tight')

def simulate_bounds():

    r = 10
    a1 = 0.5
    a2 = 1.5
    b1 = 0
    b2 = 10
    A = 400

    x_min = -50
    x_max = 50
    x0_min = -50
    x0_max = 50

    min_alpha = 0.001
    max_alpha = 0.05

    x0_count = 2000
    runs_count = 1
    xs_count = 10000
    alphas_count = 10#10

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

    fig = plt.figure(figsize=(6, 5))

    plt.plot(alpha_range, D2s_true, 'k', label='$\\mathbb{D}^2$', linewidth=3)
    #plt.plot(alpha_range, D2s, 'b--', label='$\\mathbb{D}^2$')
    plt.plot(alpha_range, V2s_true, 'k--', label='$\\mathbb{V\\,}^2$', linewidth=3)
    #plt.plot(alpha_range, V2s, 'r--', label='$\\mathbb{V\\,}^2$')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$\\alpha$')
    plt.legend()

    fig.tight_layout()
    plt.savefig('toy_dv.pdf', bbox_inches='tight')

    qs_count = 20
    qs = np.linspace(0.05, 1.0, qs_count)

    tensor = np.ones((x0_count, runs_count, alphas_count, qs_count))

    x0 = np.random.uniform(x0_min, x0_max, size=x0_count)

    x0_all = np.random.uniform(x0_min, x0_max, size=len(tensor.ravel()))#(tensor*x0[:, None, None, None]).ravel()
    alphas_all = (tensor*alpha_range[None, None, :, None]).ravel()
    qs_all = (tensor*qs[None, None, None, :]).ravel()

    #sim_xs, f_calls = simulate_blo(r, alphas_all, as_, bs, A, x0_all, qs_all, 10, 100)
    print('Done simulation')
    #sqr_derivs = get_true_blo_derivs(r, alphas_all, as_, bs, A, sim_xs)**2
    print('Done derivative computation')

    #np.savez('toy', sim_xs=sim_xs, f_calls=f_calls, sqr_derivs=sqr_derivs)
    data = np.load('toy.npz')
    sim_xs, f_calls, sqr_derivs = data['sim_xs'], data['f_calls'], data['sqr_derivs']

    sqr_derivs = sqr_derivs.reshape(-1, x0_count, runs_count, alphas_count, qs_count)
    sqr_derivs = sqr_derivs.mean(axis=(1, 2))
    deltas = sqr_derivs[:, :, 0, None].min(axis=0)*np.ones((alphas_count, qs_count))
    sqr_derivs = sqr_derivs.reshape(sqr_derivs.shape[0], -1)
    deltas = deltas.ravel()

    f_calls = f_calls.reshape(-1, x0_count, runs_count, alphas_count, qs_count)
    f_calls = f_calls.mean(axis=(1, 2))
    f_calls = f_calls.reshape(f_calls.shape[0], -1)

    reached_delta = (sqr_derivs <= deltas).astype(int)
    first_indices = reached_delta.argmax(axis=0)
    first_indices[reached_delta[first_indices, np.arange(len(first_indices))] == 0] = sqr_derivs.shape[0] - 1

    print(first_indices.reshape(alphas_count, qs_count))

    total_calls = f_calls[first_indices, np.arange(len(first_indices))]
    total_calls = total_calls.reshape(alphas_count, qs_count)

    print(total_calls)

    best_qs = qs[total_calls.argmin(axis=1)]

    best_th_qs = []

    for D2_true, V2_true in zip(D2s_true, V2s_true):

        if D2_true >= 2*r*V2_true/(2*r + 1):
            best_th_qs.append(1)
            continue

        a = r*(V2_true - D2_true)
        b = D2_true*0.5*r
        c = -D2_true*0.5*(r + 1)

        q1 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
        q2 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)

        if q1 > 0 and q1 < 1:
            best_th_qs.append(q1)
        else:
            best_th_qs.append(q2)

    fig = plt.figure(figsize=(6, 5))

    print(best_qs)
    print(best_th_qs)

    plt.plot(alpha_range, best_qs, 'g', label='experiment', linewidth=3)
    plt.plot(alpha_range, best_th_qs, 'y--', label='theory', linewidth=3)
    plt.plot()

    plt.xscale('log')
    plt.xlabel('$\\alpha$')
    plt.ylabel('$q^*$')
    plt.legend()

    fig.tight_layout()
    plt.savefig('toy_q.pdf', bbox_inches='tight')

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

def simulate_blo(r, alpha, as_, bs, A, x, q, start_step, iter_count):

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

    simulate_fo_div()
    simulate_curves()
    simulate_bounds()
