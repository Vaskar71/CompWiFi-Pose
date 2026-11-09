import argparse
from pathlib import Path

import numpy as np
import torch

from feeder.person_in_wifi_3d import PersonInWif3D, piw3_make_dataloader
from model.model import MAE_ViT, ViT_Pose_Decoder


def run_training_smoke(data_root):
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    dataset = PersonInWif3D("training", str(data_root), "one-person")
    if len(dataset) != 1:
        raise AssertionError(f"Expected one eligible training sample, found {len(dataset)}")
    loader = piw3_make_dataloader(
        dataset,
        is_training=True,
        generator=torch.Generator().manual_seed(0),
        batch_size=1,
        num_workers=0,
    )
    if loader.num_workers != 0:
        raise AssertionError("The smoke loader must use num_workers=0")
    item = dataset.data_list[0]
    sample_ids = [
        Path(item["csi_path"]).stem,
        Path(item["csi_path_next_frame"]).stem,
    ]

    mae = MAE_ViT(
        image_size=(180, 20),
        patch_size=(2, 2),
        encoder_layer=4,
        encoder_head=4,
        decoder_layer=2,
        decoder_head=4,
        emb_dim=256,
        input_dim=3,
    )
    model = ViT_Pose_Decoder(
        mae.encoder,
        keypoints=14,
        coor_num=3,
        token_num=900,
        dataset="person-in-wifi-3d",
        num_person=1,
    ).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    batch = next(iter(loader))
    if batch["person_num"].tolist() != [1]:
        raise AssertionError("The smoke batch must contain exactly one labelled person")
    csi = batch["input_wifi-csi"].to(device)
    pose_gt = batch["output"][:, :1].to(device)
    optimizer.zero_grad()
    predicted_pose, _ = model(csi)
    loss = torch.mean(torch.norm(predicted_pose - pose_gt, dim=-1))
    if not torch.isfinite(loss):
        raise AssertionError(f"Non-finite loss: {loss}")
    loss.backward()

    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    if not gradients:
        raise AssertionError("No gradients were produced")
    nonfinite_gradient_names = [
        name for name, gradient in gradients.items() if not torch.isfinite(gradient).all()
    ]
    if nonfinite_gradient_names:
        raise AssertionError(f"Non-finite gradients: {nonfinite_gradient_names}")
    nonzero_gradient_names = {
        name for name, gradient in gradients.items() if torch.count_nonzero(gradient).item() > 0
    }
    if not nonzero_gradient_names:
        raise AssertionError("All gradients are zero")

    optimizer.step()
    changed_parameter_names = [
        name
        for name, parameter in model.named_parameters()
        if name in nonzero_gradient_names
        and not torch.equal(parameters_before[name], parameter.detach())
    ]
    if not changed_parameter_names:
        raise AssertionError("No parameter with a nonzero gradient changed")

    print(f"Sample IDs: {', '.join(sample_ids)}")
    print(f"Device: {device}")
    print(f"Forward passes: 1; backward passes: 1; optimizer steps: 1")
    print(f"Finite loss: {loss.detach().item():.9f}")
    print(
        f"Finite gradients: {len(gradients)}; "
        f"nonzero-gradient tensors: {len(nonzero_gradient_names)}"
    )
    print(
        f"Changed nonzero-gradient parameters: {len(changed_parameter_names)}; "
        f"first: {changed_parameter_names[0]}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/_fixtures/piw3d_two_frame"),
    )
    args = parser.parse_args()
    run_training_smoke(args.data_root)


if __name__ == "__main__":
    main()
