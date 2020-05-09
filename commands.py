all_h_params = []

# Omniglot Reptile

for n_layers in [3, 4]:
    for n_classes in [20, 30]:
        for learning_rate in [0.005, 0.0005]:
            for adam in [False, True]:
                for nontransductive in [False, True]:
                    h_params = {
                        "config.dataset": "omniglot",
                        "config.shots": 1,
                        "config.classes": n_classes,
                        "config.inner_batch": n_classes,
                        "config.inner_iters": 10,
                        "config.meta_step": 0.0005/learning_rate,
                        "config.meta_batch": 5,
                        "config.meta_iters": 200000,
                        "config.eval_batch": 10,
                        "config.eval_iters": 50,
                        "config.learning_rate": learning_rate,
                        "config.meta_step_final": 0,
                        "config.train_shots": 10,
                        "config.checkpoint": "ckpt_o1{0}_Reptile_l={1}_lr={2}_adam={3}_trn={4}".format(n_clases,
                                n_layers, learning_rate, adam, not nontransductive),
                        "config.mode": 'Reptile',
                        "config.adam": adam,
                        "config.nontransductive": nontransductive,
                        "config.n_layers": n_layers
                    }
                    all_h_params.append(h_params)

# Omniglot UnbMAML 15 iters
for n_layers in [3, 4]:
    for n_classes in [20, 30]:
        for learning_rate in [0.005, 0.0005]:
            for prob in [0, 0.1, 0.2, 1]:
                h_params = {
                    "config.dataset": "omniglot",
                    "config.shots": 1,
                    "config.classes": n_classes,
                    "config.inner_batch": 20,
                    "config.inner_iters": 15,
                    "config.meta_step": 0.0005/learning_rate,
                    "config.meta_batch": 5,
                    "config.meta_iters": 60000,
                    "config.eval_batch": 10,
                    "config.eval_iters": 50,
                    "config.learning_rate": learning_rate,
                    "config.meta_step_final": 0,
                    "config.checkpoint": "ckpt_o1{0}_FOML_prob={1}_l={2}_ii=15_lr={3}".format(n_classes, prob, n_layers, learning_rate),
                    "config.mode": 'FOML',
                    "config.n_layers": n_layers,
                    "config.exact_prob": prob,
                    "config.clip_grads": True,
                    "config.clip_grad_value": 0.1
                }
                all_h_params.append(h_params)

'''
# Mini-ImageNet.
for learning_rate in [0.001, 0.01]:
    for clip_value in [0.1, 1.0, 10.0]:
        for prob in [0, 0.1, 0.2, 1]:
            h_params = {
                "config.dataset": "miniimagenet",
                "config.seed": 0,
                "config.shots": 1,
                "config.classes": 5,
                "config.inner_batch": 10,
                "config.inner_iters": 10,
                "config.meta_step": 0.001/learning_rate,
                "config.meta_batch": 5,
                "config.meta_iters": 30000,
                "config.eval_batch": 5,
                "config.eval_iters": 50,
                "config.learning_rate": learning_rate,
                "config.meta_step_final": 0,
                "config.checkpoint": "ckpt_m110_FOML_pr={0}_lr={1}_clip={2}".format(prob, learning_rate, clip_value),
                "config.mode": "FOML",
                "config.exact_prob": prob,
                "config.clip_grads": True,
                "config.clip_grad_value": clip_value
            }
            all_h_params.append(h_params)
'''
