all_h_params = []

# Mini-ImageNet.
for n_layers in [2, 3, 4]:
    for clip_grads in [True, False]:
        for prob in [0, 0.1, 0.2, 1]:
            h_params = {
                "config.dataset": "miniimagenet",
                "config.seed": 0,
                "config.shots": 1,
                "config.classes": 10,
                "config.inner_batch": 10,
                "config.inner_iters": 15,
                "config.meta_step": 1,
                "config.meta_batch": 5,
                "config.meta_iters": 100000,
                "config.eval_batch": 5,
                "config.eval_iters": 50,
                "config.learning_rate": 0.001,
                "config.meta_step_final": 0,
                "config.checkpoint": "ckpt_m110_FOML_pr={0}_l={1}_cg={2}".format(prob, n_layers, clip_grads),
                "config.mode": "FOML",
                "config.exact_prob": prob,
                "config.n_layers": n_layers,
                "config.clip_grads": clip_grads
            }
            all_h_params.append(h_params)

# Omniglot.
for n_layers in [2, 3, 4]:
    for clip_grads in [True, False]:
        for prob in [0, 0.1, 0.2, 1]:
            h_params = {
                "config.dataset": "omniglot",
                "config.shots": 1,
                "config.classes": 30,
                "config.inner_batch": 20,
                "config.inner_iters": 20,
                "config.meta_step": 1,
                "config.meta_batch": 5,
                "config.meta_iters": 200000,
                "config.eval_batch": 10,
                "config.eval_iters": 50,
                "config.learning_rate": 0.0005,
                "config.meta_step_final": 0,
                "config.checkpoint": "ckpt_o130_FOML_pr={0}_l={1}_cg={2}".format(prob, n_layers, clip_grads),
                "config.mode": 'FOML',
                "config.exact_prob": prob,
                "config.n_layers": n_layers,
                "config.clip_grads": clip_grads
            }
            all_h_params.append(h_params)
