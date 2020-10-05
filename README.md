# Lab 2

## 1. Select descriptor and object for research

Discriptors that were covered in lectures:
1. DCD (Dominant Color Descriptor),
2. CLD (Color Layout Descriptor)(https://en.wikipedia.org/wiki/Color_layout_descriptor),
3. HTD (Homogeneous Texture Descriptor),
4. EHD (Edge Histogram Descriptor),
5. MAD (Motion Activity Descriptor) (https://link.springer.com/chapter/10.1007%2F3-540-45453-5_58),
6. SIFT (Scale-invariant feature transform) (https://en.wikipedia.org/wiki/Scale-invariant_feature_transform),
7. ASIFT (Affine SIFT),
8. KAZE (https://www.researchgate.net/publication/236985005_KAZE_Features),
9. AKAZE (),
10. SURF (Speeded Up Robust Features) (https://link.springer.com/chapter/10.1007/11744023_32),
11. BRISK (Binary Robust Invariant Scalable Keypoints) (http://www.margaritachli.com/papers/ICCV2011paper.pdf),
12. ORB (Oriented FAST and Rotated BRIEF) (https://www.researchgate.net/publication/221111151_ORB_an_efficient_alternative_to_SIFT_or_SURF)
13. MSER (Maximally stable extremal regions) (https://en.wikipedia.org/wiki/Maximally_stable_extremal_regions),
14. Harris Affine,
15. Hessian Affine, and
16. FAST (Features from Accelerated Segment Test) (https://www.researchgate.net/publication/215458901_Machine_Learning_for_High-Speed_Corner_Detection)    

The object selected for the study should be relatevily unique to be distinctive enough from objects of other teams.

## 2. Create a data set

Take at least 100 photos of the selected object. Try to make data as variable as possible: translate, rotate, project, change illumination, add occlusion, noise, change the scene. Also take 20 photos of the other object similar to the original and without any object at all. 

## 3. Generate descriptor

Implement yourself or use impleteneted algorithm in OPENCV.

## 4. Recognize object on every image

Select keypoints for every image and create feature discriptors.

## 5. Collect metrics

* Relative number of "well matched" features
* localization error
* time to process one image

## 5. Repeat steps 3 to 5 on your teammate's data set

Teammate's should have created data sets too. So go through the same processes again on the teammate's data.

## 6. Compare descriptors

Compare results of different descriptors.

## 7. Conclusion

Make conclusions about the work done.
