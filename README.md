# p-brain-seg2roi
_An addon for [p-brain](https://github.com/edtireli/p-brain) that handles grey/white matter segmentation and boundary estimation and computes the tissue concentration time function. It does so in the following steps:_

  ## 0. GUI Slice selection
  The user is presented with a simple GUI in which they must select the 2D T2 slice desired for this analysis. It will be this slice that will be correlated with the 3D data.
  ## 1. Segmentation of GM/WM
  Here the script will run fsl_anat on the 3D T1W data if the three segmentation files are not present, and will look something like the below image:
  ![279743278-799168e4-051e-4de2-9db0-89be2caf2326-1](https://github.com/edtireli/p-brain-seg2roi/assets/129996957/8a996d89-80cf-4c04-8231-7a01c8bbddaf)
  ## 2. Correlation of 2D T2 to 3D T2 to 3D T1
  The script will correlate the 2D axial T2W image to the higher resolution 3D T2W image. It will downscale the 3D image to match the resolution as well as FOV used in the 2D.
  ![279742571-a65f7c38-d3cb-4844-8c3e-6f77dd8a9083](https://github.com/edtireli/p-brain-seg2roi/assets/129996957/db6e3af5-235d-41d4-84f4-4ce98db6cdaa)
  The geometry of the 3D images are identical, as well as the 2D and DCE. Therefore it is possible to map the 3D T2W slice directly to the 3D T1W slice. 
  ## 3. (a) Grey/White matter boundary estimation
  Now that the segmented files are ready, the script will begin dilating the white and grey matter areas of the user defined slice and the overlap between the two areas are to be isolated and used as the grey/white matter boundary
  ![279751273-2d6fb014-20c6-4cf1-9e3b-9b6b12ebf65f](https://github.com/edtireli/p-brain-seg2roi/assets/129996957/bd39a950-4c47-4601-b6d6-1de90cb0953c)
  ## 3. (b) Grey/White matter ROIs
  Now that the segmented files are ready, the script will begin dilating the white and grey matter areas of the user defined slice and the overlap between the two areas are to be isolated and used as the grey/white matter boundary
  ![dce_with_White Matter](https://github.com/edtireli/p-brain-seg2roi/assets/129996957/d3094807-df41-4ea8-aec2-ac958b8e7ac5)
  ![dce_with_Grey Matter](https://github.com/edtireli/p-brain-seg2roi/assets/129996957/3c09c9b8-4a25-40f0-bfe9-9cd28ee255de)
  ## 4. Concentration time curve computation
  Finally, given the boundary voxels one can define a ROI and compute the concentration time curves using similar methodology as in [p-brain](https://github.com/edtireli/p-brain) however this addon automatically computes the concentration time curve and places the data to be used by p-brain. The DCE data has the same geometry as the 2D T2 and thus can be used to compute the concentration time functions, demonstrated below:
  ![CTC+ROI_slice_7](https://github.com/edtireli/p-brain-boundary/assets/129996957/94682f60-beae-46c7-a930-a14da0ea7460)
  ![CTC+ROI_slice_7](https://github.com/edtireli/p-brain-seg2roi/assets/129996957/640898d3-3dd2-4e0b-b096-36f925116a92)
  ![CTC+ROI_slice_6](https://github.com/edtireli/p-brain-seg2roi/assets/129996957/53828f12-6921-4370-8d2d-72dc289489ed)

  For any further analysis of the data, consult p-brain: the files are readily accessible and are integrated to be used by it. 
