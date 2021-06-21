"""Custom python decorators.
"""
import torch
import numpy as np


def concatenate(x: torch.Tensor):
    """Concatenate torch Tensor or numpy array."""
    if len(x) == 0:
        return x
    if isinstance(x[0], torch.Tensor):
        return torch.cat(x)
    elif isinstance(x[0], np.ndarray):
        return np.concatenate(x)
    return None


def to_minibatch(expected_outputs: int):
    """Decorator to process a function as a minibatch."""

    def minibatch_wrapper(func):
        def func_wrapper(*args, **kwargs):

            assert "num_points" in kwargs
            num_points = kwargs["num_points"]

            assert "minibatch_size" in kwargs
            mb_size = min(kwargs["minibatch_size"], num_points)

            start_idx, end_idx = 0, mb_size
            outputs = [[] for _ in range(expected_outputs)]

            while start_idx < num_points:

                # Subselect args
                sub_args = []
                for a in args:
                    if isinstance(a, (np.ndarray, torch.Tensor)):
                        sub_args.append(a[start_idx:end_idx])
                    else:
                        sub_args.append(a)

                # Subselect kwargs
                sub_kwargs = {}
                for k, v in kwargs.items():
                    if k in ["num_points", "minibatch_size"]:
                        continue
                    if isinstance(v, (np.ndarray, torch.Tensor)) and (
                        "descriptors" in k or "keypoints" in k or "norm_coarse" in k
                    ):
                        sub_kwargs[k] = v[start_idx:end_idx]
                    else:
                        sub_kwargs[k] = v

                # Function call
                sub_outputs = func(*sub_args, **sub_kwargs)
                if isinstance(sub_outputs, (list, tuple)):
                    for i, o in enumerate(sub_outputs):
                        outputs[i].append(o)
                else:
                    assert expected_outputs == 1, "Wrong expected_outputs"
                    outputs[0].append(sub_outputs)

                # Update indices
                start_idx += mb_size
                end_idx = min(start_idx + mb_size, num_points)

            # Concatenate predictions
            outputs = [concatenate(x) for x in outputs]

            # Sanity checks
            for o in outputs:
                assert len(o) == num_points

            if expected_outputs == 1:
                return outputs[0]
            return outputs

        return func_wrapper

    return minibatch_wrapper
