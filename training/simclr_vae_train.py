import os
import argparse
import torch


def train_one_epoch(
    model,
    projector,
    criterion,
    dataloader,
    optimizer,
    device,
    epoch,
    log_every=20,
    use_amp=False,
):
    model.train()
    projector.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for batch_idx, (imgs, _) in enumerate(dataloader):
        x1 = imgs[0].to(device, non_blocking=True)
        x2 = imgs[1].to(device, non_blocking=True)

        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                h1 = model(x1)
                h2 = model(x2)
                z1 = projector(h1)
                z2 = projector(h2)
                loss = criterion(z1, z2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            h1 = model(x1)
            h2 = model(x2)
            z1 = projector(h1)
            z2 = projector(h2)
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % log_every == 0:
            avg = total_loss / (batch_idx + 1)
            print(
                f"Epoch [{epoch}] Batch [{batch_idx+1}/{len(dataloader)}] AvgLoss: {avg:.4f}"
            )

    avg_loss = total_loss / max(1, len(dataloader))
    return avg_loss


# ----------------------------
# Main
# ----------------------------
def main(args):
    # Device
    device = "cuda" if (torch.cuda.is_available() and not args.disable_cuda) else "cpu"
    print("Using device:", device)

    # Transforms and Dataset
    transform = SimCLRXRayTransform(
        image_size=args.image_size,
        to_3ch=not args.use_1ch_vit,
        hflip_prob=args.hflip_prob,
    )
    dataset = ChestXRayDataset(
        images_dir=args.images,
        csv_path=args.data_csv,
        transform=transform,
        return_label=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    vit_name = args.vit_name
    use_timm = args.use_timm and has_timm
    if use_timm:
        print("Using timm model:", vit_name)
    else:
        print(
            "timm not available or disabled; falling back to torchvision/resnet variant."
        )

    backbone = ViTBackbone(
        model_name=vit_name,
        pretrained=args.pretrained_backbone,
        use_timm=use_timm,
        in_chans=(1 if args.use_1ch_vit else 3),
    )
    backbone = backbone.to(device)

    projector = ProjectionHead(
        in_dim=backbone.feat_dim,
        hidden_dim=args.proj_hidden_dim,
        out_dim=args.proj_out_dim,
    ).to(device)

    # Loss & Optimizer
    criterion = NTXentLoss(
        batch_size=args.batch_size, temperature=args.temperature, device=device
    ).to(device)
    params = list(backbone.parameters()) + list(projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # optionally use mixed precision
    use_amp = args.use_amp and device == "cuda"

    # Training loop
    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            backbone,
            projector,
            criterion,
            loader,
            optimizer,
            device,
            epoch,
            log_every=args.print_freq,
            use_amp=use_amp,
        )
        scheduler.step()
        print(
            f"Epoch {epoch} finished. AvgLoss={avg_loss:.4f}; LR={scheduler.get_last_lr()[0]:.6e}"
        )

        if epoch % args.save_every == 0:
            path = os.path.join(args.save_dir, f"simclr_vit_epoch{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state": backbone.state_dict(),
                    "projector_state": projector.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                path,
            )
            print("Saved checkpoint:", path)

    # final save
    final_path = os.path.join(args.save_dir, "simclr_vit_final.pth")
    torch.save(
        {
            "backbone_state": backbone.state_dict(),
            "projector_state": projector.state_dict(),
        },
        final_path,
    )
    print("Training complete. Model saved to:", final_path)

    # Optional: skeleton for linear probe training (user can fill specifics)
    if args.do_linear_probe:
        print("Starting linear probe (skeleton).")
        # Freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False
        # Extract features for dataset (one pass) and train a linear classifier on labeled subset
        # (Implementation left as skeleton due to dataset split variations)
        # Suggestion: build a new DataLoader returning single view, compute features h = backbone(x), store features & labels,
        # then train nn.Linear(feature_dim, num_classes) with CrossEntropyLoss on labeled training split.


# ----------------------------
# CLI args
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SimCLR (ViT) pretraining on ChestX-ray14"
    )
    parser.add_argument(
        "--images", type=str, required=True, help="directory with chest x-ray images"
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default=None,
        help="optional CSV for image filenames (NIH Data_Entry_2017.csv)",
    )
    parser.add_argument(
        "--vit-name",
        type=str,
        default="vit_base_patch16_224",
        help="timm model name (or torchvision variant name)",
    )
    parser.add_argument(
        "--use-timm", action="store_true", help="use timm (recommended)"
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="start from ImageNet-pretrained ViT",
    )
    parser.add_argument(
        "--use-1ch-vit",
        action="store_true",
        help="use ViT configured for 1 channel (do not repeat to 3ch)",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="batch size (N), effective batch will be 2N for two views)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--proj-hidden-dim", type=int, default=2048)
    parser.add_argument("--proj-out-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-dir", type=str, default="./checkpoints_simclr_vit")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="use automatic mixed precision (cuda only)",
    )
    parser.add_argument(
        "--hflip-prob",
        type=float,
        default=0.0,
        help="horizontal flip probability (0.0 recommended for chest x-rays)",
    )
    parser.add_argument("--do-linear-probe", action="store_true")
    args = parser.parse_args()

    main(args)
