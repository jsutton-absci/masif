import open3d as o3d
from packaging import version

_v = version.parse(o3d.__version__)

if _v >= version.parse('0.13.0'):
    # >= 0.13: registration moved to o3d.pipelines.registration
    PointCloud = o3d.geometry.PointCloud
    Vector3dVector = o3d.utility.Vector3dVector
    Feature = o3d.pipelines.registration.Feature
    read_point_cloud = o3d.io.read_point_cloud
    registration_ransac_based_on_feature_matching = o3d.pipelines.registration.registration_ransac_based_on_feature_matching
    registration_icp = o3d.pipelines.registration.registration_icp
    TransformationEstimationPointToPoint = o3d.pipelines.registration.TransformationEstimationPointToPoint
    TransformationEstimationPointToPlane = o3d.pipelines.registration.TransformationEstimationPointToPlane
    CorrespondenceCheckerBasedOnEdgeLength = o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength
    CorrespondenceCheckerBasedOnDistance = o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance
    CorrespondenceCheckerBasedOnNormal = o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal
    RANSACConvergenceCriteria = o3d.pipelines.registration.RANSACConvergenceCriteria
    KDTreeFlann = o3d.geometry.KDTreeFlann
elif _v >= version.parse('0.6.0'):
    # 0.6 – 0.12: structured namespaces, registration still at top level
    PointCloud = o3d.geometry.PointCloud
    Vector3dVector = o3d.utility.Vector3dVector
    Feature = o3d.registration.Feature
    read_point_cloud = o3d.io.read_point_cloud
    registration_ransac_based_on_feature_matching = o3d.registration.registration_ransac_based_on_feature_matching
    registration_icp = o3d.registration.registration_icp
    TransformationEstimationPointToPoint = o3d.registration.TransformationEstimationPointToPoint
    TransformationEstimationPointToPlane = o3d.registration.TransformationEstimationPointToPlane
    CorrespondenceCheckerBasedOnEdgeLength = o3d.registration.CorrespondenceCheckerBasedOnEdgeLength
    CorrespondenceCheckerBasedOnDistance = o3d.registration.CorrespondenceCheckerBasedOnDistance
    CorrespondenceCheckerBasedOnNormal = o3d.registration.CorrespondenceCheckerBasedOnNormal
    RANSACConvergenceCriteria = o3d.registration.RANSACConvergenceCriteria
    KDTreeFlann = o3d.geometry.KDTreeFlann
else:
    # < 0.6: flat namespace
    PointCloud = o3d.PointCloud
    Vector3dVector = o3d.Vector3dVector
    Feature = o3d.Feature
    read_point_cloud = o3d.read_point_cloud
    registration_ransac_based_on_feature_matching = o3d.registration_ransac_based_on_feature_matching
    registration_icp = o3d.registration_icp
    TransformationEstimationPointToPoint = o3d.TransformationEstimationPointToPoint
    TransformationEstimationPointToPlane = o3d.TransformationEstimationPointToPlane
    CorrespondenceCheckerBasedOnEdgeLength = o3d.CorrespondenceCheckerBasedOnEdgeLength
    CorrespondenceCheckerBasedOnDistance = o3d.CorrespondenceCheckerBasedOnDistance
    CorrespondenceCheckerBasedOnNormal = o3d.CorrespondenceCheckerBasedOnNormal
    RANSACConvergenceCriteria = o3d.RANSACConvergenceCriteria
    KDTreeFlann = o3d.KDTreeFlann
