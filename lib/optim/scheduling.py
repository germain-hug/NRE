import torch


def get_sigmas(sigma_min: float, sigma_max: float):
    """GNC Sigma scheduling."""
    sigmas = (
        [1024 * i for i in range(1, 12)][::-1]
        + [2 ** i for i in range(10)][::-1]
        + [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
    )
    sigmas = [s for s in sigmas if s <= sigma_max and s >= sigma_min]
    return sigmas


def rho_test(
    cost: torch.Tensor,
    new_cost: torch.Tensor,
    total_cost: torch.Tensor,
    new_total_cost: torch.Tensor,
):
    """Modified gain ratio test."""
    improved_maps = new_cost < cost
    psi_inf = (cost[improved_maps]).sum()
    psi_sup = (cost[~improved_maps]).sum()
    psi_new_inf = (new_cost[improved_maps]).sum()
    psi_new_sup = (new_cost[~improved_maps]).sum()
    delta_inf = psi_inf - psi_new_inf
    delta_sup = psi_new_sup - psi_sup
    assert delta_inf.item() >= 0 and delta_sup.item() >= 0
    rho_delta = (total_cost - new_total_cost) / (delta_inf + delta_sup)
    return rho_delta
