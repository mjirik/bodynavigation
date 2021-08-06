import io3d
import sed3
from loguru import logger
import numpy as np
import bodynavigation
import imma
from matplotlib import pyplot as plt
import itertools
# logger.disable("io3d")
# logger.remove()
# logger.add(sys.stderr, level='INFO')
#%%

def enumerated_product(*args):
    yield from zip(itertools.product(*(range(len(x)) for x in args)), itertools.product(*args))

def test_different_parameters():
    axcodes="SPL"
    dataset = "3Dircadb1"
    data_id = 2
    datap = io3d.datasets.read_dataset(dataset, "data3d", data_id, orientation_axcodes=axcodes)
    bn0 = bodynavigation.body_navigation.BodyNavigation(datap.data3d, datap.voxelsize_mm)
    # which data are reference center each from the middle
    id_reference = 1

    params = [
        # bn._bones_threshold_hu.__name__,
        # bn._body_gaussian_sigma_mm.__name__,
        "_symmetry_bones_threshold_hu",
        "_symmetry_gaussian_sigma_mm",
        # "_body_threshold",
        # "_body_gaussian_sigma_mm"
    ]
    param_koeffs = [.2, 1.]
    default_values = [
        getattr(bn0, param)
        for param in params
    ]
    logger.debug(default_values)
    dfd = {
        "dataset":[],
        "data_id":[],
        "param":[],
        'koeff':[],
        "center_0":[],
        "center_1": [],
        "center_2": [],
        "error": [],
        "error_axial":[],
        "body_center_0": [],
        "body_center_1": [],
        "body_center_2": [],
        "body_error": [],
        "body_error_axial": [],

    }
    rows = len(params)
    cols = len(param_koeffs)

    for dataset, data_id in itertools.product(["3Dircadb1"],list(range(1,3))):

        datap = io3d.datasets.read_dataset(dataset, "data3d", data_id, orientation_axcodes=axcodes)
        fig, axs = plt.subplots(rows, cols, figsize=(15,10))

        reference_center = np.asarray(bn0.get_center_mm())
        reference_body_center = np.asarray(bn0.body_center_mm)

        # plt.figure()
        if hasattr(axs, 'flat'):
            # ax = axs.flat[0]
            axs = axs
        else:
            axs = [axs]

        for ax, koeff in zip(axs[0], param_koeffs):
            ax.set_title(koeff)

        for ax, param in zip(axs[:, 0], params):
            ax.set_ylabel(param, rotation=90, size='large')
            # ax.suptitle(param)

        fig.tight_layout()

        for (i,j), (param, koeff) in enumerated_product(params, param_koeffs):

            if koeff == 1.:
                # to improce efficiency, we use the calcultion done before
                bn = bn0
            else:
                bn = bodynavigation.body_navigation.BodyNavigation(datap.data3d, datap.voxelsize_mm)
            print(i,j)
            ax = axs[i,j]
            # ax.axis("off")
            # Turn off tick labels
            ax.set_yticklabels([])
            ax.set_xticklabels([])

            # plt.subplot(121)
            slice_i = 50
            setattr(bn, param, default_values[i]*koeff)
            logger.debug(f"{param}={getattr(bn, param)}")
            ax.imshow(datap.data3d[slice_i,:,:], cmap='gray')
            fcnlist = [
                bn.dist_to_sagittal,
                bn.dist_to_coronal,
                bn.dist_to_surface,
                # bn.dist_to_diaphragm_axial,
            ]
            colors = [
                'r', 'g', 'c',#'c'
            ]
            for fn, color in zip(fcnlist, colors):
                dist = fn()
                ax.contour(dist[slice_i,:,:]>0, colors=color, linewidths=2)

            bn.get_center_mm()
            bn.get_body()
            logger.debug(f"center={bn.center_mm}")
            dfd["dataset"].append(dataset)
            dfd['data_id'].append(data_id)
            dfd["param"].append(param)
            dfd['koeff'].append(koeff)
            dfd["center_0"].append(bn.center_mm[0])
            dfd["center_1"].append(bn.center_mm[1])
            dfd["center_2"].append(bn.center_mm[2])
            dfd["error"].append(np.linalg.norm(reference_center - bn.center_mm))
            dfd["error_axial"].append(np.linalg.norm(reference_center[1:] - bn.center_mm[1:]))
            dfd["body_center_0"].append(bn.body_center_mm[0])
            dfd["body_center_1"].append(bn.body_center_mm[1])
            dfd["body_center_2"].append(bn.body_center_mm[2])
            dfd["body_error"].append(np.linalg.norm(reference_body_center - bn.body_center_mm))
            dfd["body_error_axial"].append(np.linalg.norm(reference_body_center[1:] - bn.body_center_mm[1:]))

            logger.debug(f"error={dfd['error']}")
            logger.debug(f"error_axial={dfd['error_axial']}")
            # ax = axs[1]
            # ax.imshow(dist[:,256,:])
        plt.savefig(f"sensitivity_{dataset}_{data_id:02d}.pdf")
        # plt.show()


    import pandas as pd
    df = pd.DataFrame(dfd)
    # df["center_reference_0"] = dfd["center_0"][id_reference]
    # df["center_reference_1"] = dfd["center_1"][id_reference]
    # df["center_reference_2"] = dfd["center_2"][id_reference]
    # ref = np.array([dfd["center_0"][id_reference],
    # dfd["center_1"][id_reference],
    # dfd["center_2"][id_reference]])

    # df["err_flat"] = np.linalg.norm(array([df['center_1'],
    df.to_excel("internal_parameters_sensitivity.xlsx")

#%%


