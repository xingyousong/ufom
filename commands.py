all_h_params = []

# Mini-ImageNet.
for n_layers in [3, 4]:
    for inner_iters in [5, 10]:
        for learning_rate in [0.001, 0.01]:
            for prob in [0, 0.1, 0.2, 1]:
                h_params = {
                    "config.dataset": "miniimagenet",
                    "config.seed": 0,
                    "config.shots": 1,
                    "config.classes": 10,
                    "config.inner_batch": 10,
                    "config.inner_iters": inner_iters,
                    "config.meta_step": 0.001/learning_rate,
                    "config.meta_batch": 5,
                    "config.meta_iters": 30000,
                    "config.eval_batch": 5,
                    "config.eval_iters": 50,
                    "config.learning_rate": learning_rate,
                    "config.meta_step_final": 0,
                    "config.checkpoint": "ckpt_m110_FOML_pr={0}_l={1}_lr={2}_ii={3}".format(prob, n_layers, learning_rate, inner_iters),
                    "config.mode": "FOML",
                    "config.exact_prob": prob,
                    "config.n_layers": n_layers
                }
                all_h_params.append(h_params)

# Omniglot.
for n_layers in [3, 4]:
    for inner_iters in [5, 10]:
        for learning_rate in [0.0005, 0.005]:
            for prob in [0, 0.1, 0.2, 1]:
                h_params = {
                    "config.dataset": "omniglot",
                    "config.shots": 1,
                    "config.classes": 30,
                    "config.inner_batch": 30,
                    "config.inner_iters": inner_iters,
                    "config.meta_step": 0.0005/learning_rate,
                    "config.meta_batch": 5,
                    "config.meta_iters": 60000,
                    "config.eval_batch": 10,
                    "config.eval_iters": 50,
                    "config.learning_rate": learning_rate,
                    "config.meta_step_final": 0,
                    "config.checkpoint": "ckpt_o130_FOML_pr={0}_l={1}_lr={2}_ii={3}".format(prob, n_layers, learning_rate, inner_iters),
                    "config.mode": 'FOML',
                    "config.exact_prob": prob,
                    "config.n_layers": n_layers
                }
                all_h_params.append(h_params)
