import sdf_unet256

sdf_unet256.train(
    imshape=256,
    # sdf_type='diaphragm_axial',
    # sdf_type='coronal',
    sdf_type="sagittal",
    # sdf_type='surface',
    skip_h5=False,
    batch_size=16,
    epochs=100,
    filename_prefix="",
    validation_ids=[19, 39],
    test_ids=[20, 40],
    n_data=40,
)
