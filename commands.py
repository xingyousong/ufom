all_h_params = []


# CIFAR100 FOML
for seed in [0]:
    for n_classes in [20, 30]: #[5, 10, 15]:
        for inner_iters in [10]:
                for prob in [0.0, -1.0, 1.0]:
                    h_params = {
                        "config.dataset": "cifar100",
                        "config.seed": seed,
                        "config.shots": 1,
                        "config.classes": n_classes,
                        "config.inner_batch": n_classes,
                        "config.inner_iters": inner_iters,
                        "config.meta_step": 0.1,
                        "config.meta_batch": 5,
                        "config.meta_iters": 40000,
                        "config.eval_batch": n_classes,
                        "config.eval_iters": inner_iters,
                        "config.learning_rate": 0.005,
                        "config.train_shots": None,
                        "config.meta_step_final": 0,
                        "config.checkpoint": "ckpt_c1{0}_FOML_prob={1}_ii={2}_seed={3}".format(n_classes, prob, inner_iters, seed),
                        "config.mode": 'FOML',
                        "config.exact_prob": prob,
                        "config.clip_grads": True,
                        "config.clip_grad_value": 0.1,
                        "config.on_resampling": False
                    }
                    all_h_params.append(h_params)

# CIFAR100 Reptile
for n_classes in [20, 30]:#[5, 10, 15]:
    for learning_rate in [0.0005]:
        for seed in [0]:
            h_params = {
                "config.dataset": "cifar100",
                "config.seed": 0,
                "config.shots": 1,
                "config.classes": n_classes,
                "config.inner_batch": n_classes,
                "config.inner_iters": 10,
                "config.meta_step": 0.0005/learning_rate,
                "config.meta_batch": 5,
                "config.meta_iters": 40000,
                "config.eval_batch": 10,
                "config.eval_iters": 50,
                "config.learning_rate": learning_rate,
                "config.meta_step_final": 0,
                "config.train_shots": 10,
                "config.checkpoint": "ckpt_c1{0}_Reptile_lr={1}_seed={2}".format(n_classes, learning_rate, seed),
                "config.mode": 'Reptile',
                "config.adam": True
            }
            all_h_params.append(h_params)

'''
# FOML Mini-ImageNet.
for seed in [0]:
    for n_shots, n_classes, eval_batch in [(5, 5, 15), (1, 5, 5), (1, 15, 10)]:
        for prob in [0.0, -1.0, 1.0]:
            h_params = {
                "config.dataset": "miniimagenet",
                "config.seed": seed,
                "config.shots": n_shots,
                "config.classes": n_classes,
                "config.inner_batch": 10,
                "config.inner_iters": 8,
                "config.meta_step": 1,
                "config.meta_batch": 5,
                "config.meta_iters": 100000,
                "config.eval_batch": eval_batch,
                "config.eval_iters": 8,
                "config.learning_rate": 0.001,
                "config.meta_step_final": 0,
                "config.checkpoint": "ckpt_m{0}{1}_FOML_pr={2}_seed={3}".format(n_shots, n_classes, prob, seed),
                "config.mode": "FOML",
                "config.exact_prob": prob,
                "config.clip_grads": True,
                "config.clip_grad_value": 0.1,
                "config.train_shots": None,
                "config.on_resampling": False
            }
            all_h_params.append(h_params)

# Mini-ImageNet Reptile
for n_shots, n_classes, eval_batch in [(5, 5, 15), (1, 5, 5), (1, 15, 10)]:
    for seed in [0]:
        h_params = {
            "config.dataset": "miniimagenet",
            "config.seed": seed,
            "config.shots": n_shots,
            "config.classes": n_classes,
            "config.inner_batch": 10,
            "config.inner_iters": 8,
            "config.meta_step": 1,
            "config.meta_batch": 5,
            "config.meta_iters": 100000,
            "config.eval_batch": eval_batch,
            "config.eval_iters": 50,
            "config.learning_rate": 0.001,
            "config.meta_step_final": 0,
            "config.train_shots": 15,
            "config.checkpoint": "ckpt_m{0}{1}_Reptile_seed={2}".format(n_shots, n_classes, seed),
            "config.mode": 'Reptile',
            "config.adam": True
        }
        all_h_params.append(h_params)
'''

# Omniglot FOML
for seed in [0]:
    for n_classes in [50, 60]:#[5, 10, 15, 20, 30, 40]:
        for inner_iters in [10]:
                for prob in [0.0, -1.0, 1.0]:
                    h_params = {
                        "config.dataset": "omniglot",
                        "config.seed": seed,
                        "config.shots": 1,
                        "config.classes": n_classes,
                        "config.inner_batch": n_classes,
                        "config.inner_iters": inner_iters,
                        "config.meta_step": 0.1,
                        "config.meta_batch": 5,
                        "config.meta_iters": 200000,
                        "config.eval_batch": n_classes,
                        "config.eval_iters": inner_iters,
                        "config.learning_rate": 0.005,
                        "config.train_shots": None,
                        "config.meta_step_final": 0,
                        "config.checkpoint": "ckpt_o1{0}_FOML_prob={1}_ii={2}_seed={3}".format(n_classes, prob, inner_iters, seed),
                        "config.mode": 'FOML',
                        "config.exact_prob": prob,
                        "config.clip_grads": (inner_iters == 10),
                        "config.clip_grad_value": 0.1,
                        "config.on_resampling": False
                    }
                    all_h_params.append(h_params)

# Omniglot Reptile
for n_classes in [50, 60]:#[5, 10, 15, 20, 30, 40]:
    for learning_rate in [0.0005]:
        for seed in [0]:
            h_params = {
                "config.dataset": "omniglot",
                "config.seed": seed,
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
                "config.checkpoint": "ckpt_o1{0}_Reptile_lr={1}_seed={2}".format(n_classes, learning_rate, seed),
                "config.mode": 'Reptile',
                "config.adam": True
            }
            all_h_params.append(h_params)

'''
# FOML Mini-ImageNet.
for seed in [0, 1, 2]:
    for learning_rate in [0.001]:
        for n_classes in [10]:
            for inner_iters in [8]:
                for prob in [1.0]: #[0.0, 0.2, 0.4, 0.6, 0.8]:
                    h_params = {
                        "config.dataset": "miniimagenet",
                        "config.seed": seed,
                        "config.shots": 1,
                        "config.classes": n_classes,
                        "config.inner_batch": 10,
                        "config.inner_iters": inner_iters,
                        "config.meta_step": 0.001/learning_rate,
                        "config.meta_batch": 5,
                        "config.meta_iters": int(100000*45/(9 + 36*prob)),
                        "config.eval_batch": 10,
                        "config.eval_iters": inner_iters,
                        "config.learning_rate": learning_rate,
                        "config.meta_step_final": 0,
                        "config.checkpoint": "ckpt_m1{0}_FOML_pr={1}_lr={2}_ii={3}_seed={4}".format(
                                n_classes, prob, learning_rate, inner_iters, seed),
                        "config.mode": "FOML",
                        "config.exact_prob": prob,
                        "config.clip_grads": (inner_iters == 10),
                        "config.clip_grad_value": 0.1,
                        "config.train_shots": None,
                        "config.on_resampling": False
                    }
                    all_h_params.append(h_params)

# Mini-ImageNet Reptile
for n_classes in [10]:
    for learning_rate in [0.001]:
        for seed in [0, 1, 2]:
            h_params = {
                "config.dataset": "miniimagenet",
                "config.seed": seed,
                "config.shots": 1,
                "config.classes": n_classes,
                "config.inner_batch": 10,
                "config.inner_iters": 8,
                "config.meta_step": 0.001/learning_rate,
                "config.meta_batch": 5,
                "config.meta_iters": int(100000*45/8),
                "config.eval_batch": 5,
                "config.eval_iters": 50,
                "config.learning_rate": learning_rate,
                "config.meta_step_final": 0,
                "config.train_shots": 15,
                "config.checkpoint": "ckpt_m1{0}_Reptile_lr={1}_seed={2}".format(n_classes, learning_rate, seed),
                "config.mode": 'Reptile',
                "config.adam": True
            }
            all_h_params.append(h_params)
'''
