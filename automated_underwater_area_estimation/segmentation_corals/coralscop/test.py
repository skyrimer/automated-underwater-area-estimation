from automated_underwater_area_estimation.segmentation_corals.coralscop.coralscop_model import CoralSCOP
model = CoralSCOP()  # Automatically downloads to ./checkpoints/vit_b_coralscop.pth
print(model)