all_h_params = []

# old commands
'''
# FOML Mini-ImageNet.
for seed in [0, 1, 2]:
    for n_shots, n_classes, eval_batch in [(5, 5, 15), (1, 5, 5), (1, 15, 10), (1, 20, 10)]:
        for prob in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
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
for n_shots, n_classes, eval_batch in [(5, 5, 15), (1, 5, 5), (1, 15, 10), (1, 20, 10)]:
    for seed in [0, 1, 2]:
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

'''
omniglot_meta_iters = {(20, 0.0): 352000, (30, 0.0): 579000, (40, 0.0): 553000, (50, 0.0): 507000,
                       (20, 0.2): 340000, (30, 0.2): 433000, (40, 0.2): 408000, (50, 0.2): 381000,
                       (20, 0.4): 312000, (30, 0.4): 366000, (40, 0.4): 312000, (50, 0.4): 301000,
                       (20, 0.6): 248000, (30, 0.6): 297000, (40, 0.6): 271000, (50, 0.6): 268000,
                       (20, 0.8): 233000, (30, 0.8): 261000, (40, 0.8): 237000, (50, 0.8): 228000}

mini_imagenet_meta_iters = {(1, 10, 0.0): 238000,
                            (1, 10, 0.2): 162000,
                            (1, 10, 0.4): 139000,
                            (1, 10, 0.6): 134000,
                            (1, 10, 0.8): 107000}
'''

'''
# Omniglot FOML
for seed in [0, 1, 2]:
    for n_classes in [20, 30, 40, 50]:
        for inner_iters in [10]:
                for prob in [0.0, 0.2, 0.4, 0.6, 0.8]:
                    h_params = {
                        "config.dataset": "omniglot",
                        "config.seed": seed,
                        "config.shots": 1,
                        "config.classes": n_classes,
                        "config.inner_batch": n_classes,
                        "config.inner_iters": inner_iters,
                        "config.meta_step": 0.1,
                        "config.meta_batch": 5,
                        "config.meta_iters": int(200000*66/(11 + prob*55)),
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
for n_classes in [20, 30, 40, 50]:
    for learning_rate in [0.0005]:
        for seed in [0, 1, 2]:
            h_params = {
                "config.dataset": "omniglot",
                "config.seed": seed,
                "config.shots": 1,
                "config.classes": n_classes,
                "config.inner_batch": n_classes,
                "config.inner_iters": 10,
                "config.meta_step": 0.0005/learning_rate,
                "config.meta_batch": 5,
                "config.meta_iters": int(200000*66/10),
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

'''
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
