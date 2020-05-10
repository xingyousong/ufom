all_h_params = []

# Omniglot FOML
for seed in [0, 1, 2]:
    for n_classes in [20, 30]:
        for inner_iters in [5, 10]:
                for prob in [0, 0.1, 0.2, 1]:
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

for n_classes in [20, 30]:
    for learning_rate in [0.005, 0.0005]:
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

