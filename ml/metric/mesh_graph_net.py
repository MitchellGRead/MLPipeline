import torch


def evaluate(
    loader,
    device,
    test_model,
    mean_vec_x,
    std_vec_x,
    mean_vec_edge,
    std_vec_edge,
    mean_vec_y,
    std_vec_y,
    delta_t=0.01,
):
    loss = 0
    velo_rmse = 0
    num_loops = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = test_model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss += test_model.loss(pred, data, mean_vec_y, std_vec_y)

            loss_mask = torch.logical_or(
                (torch.argmax(data.x[:, 2:], dim=1) == torch.tensor(0)),
                (torch.argmax(data.x[:, 2:], dim=1) == torch.tensor(5)),
            )

            eval_velo = data.x[:, 0:2] + pred[:] * delta_t
            gs_velo = data.x[:, 0:2] + data.y[:] * delta_t

            error = torch.sum((eval_velo - gs_velo) ** 2, axis=1)
            velo_rmse += torch.sqrt(torch.mean(error[loss_mask]))

        num_loops += 1
        # if velocity is evaluated, return velo_rmse as 0
    return loss / num_loops, velo_rmse / num_loops
