import numpy as np


def relative_error(analytical, numerical, eps=1e-12):
    denom = np.maximum(np.abs(analytical), np.abs(numerical))
    denom = np.maximum(denom, eps)
    return np.abs(analytical - numerical) / denom


def numerical_gradient(param, loss_fn, eps=1e-5):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original_value = param[idx]

        param[idx] = original_value + eps
        loss_plus = loss_fn()

        param[idx] = original_value - eps
        loss_minus = loss_fn()

        param[idx] = original_value  # restore
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)

        it.iternext()
    return grad


def check_dense_layer_gradients(model, loss_fn_full, X, y, eps=1e-5,
                                 sample_fraction=1.0, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    results = []

    for i, dense in enumerate(model.dense_layers):
        for param, analytical_grad, name in dense.params_and_grads():
            if sample_fraction < 1.0:
                mask = rng.random(param.shape) < sample_fraction
                if not mask.any():
                    mask.flat[0] = True
            else:
                mask = np.ones_like(param, dtype=bool)

            num_grad = np.zeros_like(param)
            it = np.nditer(param, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                if mask[idx]:
                    original = param[idx]
                    param[idx] = original + eps
                    loss_plus = loss_fn_full()
                    param[idx] = original - eps
                    loss_minus = loss_fn_full()
                    param[idx] = original
                    num_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
                it.iternext()

            checked = analytical_grad[mask]
            numerical = num_grad[mask]
            rel_err = relative_error(checked, numerical)

            results.append({
                "layer": i,
                "param": name,
                "n_checked": mask.sum(),
                "max_relative_error": float(np.max(rel_err)),
                "mean_relative_error": float(np.mean(rel_err)),
                "sample_analytical": float(checked.flat[0]),
                "sample_numerical": float(numerical.flat[0]),
            })

    return results


def print_gradient_check_report(results, threshold=1e-5):
    print(f"{'Layer':<7}{'Param':<7}{'#checked':<10}"
          f"{'Max rel err':<15}{'Mean rel err':<15}{'Status'}")
    print("-" * 65)
    all_passed = True
    for r in results:
        status = "PASS" if r["max_relative_error"] < threshold else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"{r['layer']:<7}{r['param']:<7}{r['n_checked']:<10}"
              f"{r['max_relative_error']:<15.2e}"
              f"{r['mean_relative_error']:<15.2e}{status}")
    print("-" * 65)
    print(f"Example: Dense layer 0, param W")
    print(f"  Analytical gradient : {results[0]['sample_analytical']:.6f}")
    print(f"  Finite difference   : {results[0]['sample_numerical']:.6f}")
    print(f"  Relative error      : {results[0]['max_relative_error']:.1e}")
    print()
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    return all_passed