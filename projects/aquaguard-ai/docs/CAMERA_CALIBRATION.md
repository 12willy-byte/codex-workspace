# Camera Calibration Protocol

Every enabled camera needs its own traceable calibration artifact before its observations can be
treated as pool-space measurements. Copying a matrix or ROI from another camera is not valid: a
camera move, lens change, crop, resolution change, or stabilization setting can invalidate both.

## Artifact contents

`CameraCalibrationArtifact` binds:

- physical camera ID and calibration version;
- exact source-frame SHA-256 and pixel dimensions;
- image-to-pool Homography;
- image-space water ROI;
- at least four image/pool control-point correspondences;
- an explicit project-approved maximum reprojection error;
- timezone-aware calibration time and pseudonymous technician ID.

The validator rejects non-finite or singular Homographies, out-of-frame points, duplicate or
collinear control points, degenerate or self-intersecting water regions, and measured reprojection
error above the artifact's explicit limit. The repository does not supply a universal acceptable
error threshold; the site owner must derive and approve one from the safety case and camera view.

```bash
aquaguard-verify-camera-calibration \
  cam-a-calibration.json cam-a-calibration-report.json
```

Exit status `2` means the artifact failed validation. A successful report includes the canonical
artifact SHA-256, water-region pixel area, control-point count, and mean/maximum reprojection
errors. `CameraRuntimeConfig.from_calibration(...)` transfers the validated Homography and water
ROI into runtime configuration and preserves the artifact SHA-256 for audit.

## Required site procedure

1. Freeze camera mount, lens, stream profile, resolution, crop and stabilization settings.
2. Capture the source frame and record its SHA-256.
3. Measure at least four non-collinear pool reference points spanning the working area; use more
   points for independent residual checking.
4. Fit the Homography outside this repository with a reviewed calibration tool, then record all
   source correspondences rather than only the fitted matrix.
5. Draw the image-space water ROI without self-intersections and keep every vertex in frame.
6. Run validation, independently inspect projected points, and approve the site-specific error
   limit before enabling protective features.
7. Recalibrate after any camera/lens/stream geometry change. Version and retain the old artifact;
   do not overwrite it.

This engineering validation does not prove that a real camera is correctly installed. Field
acceptance still requires measured reference points, an independent visual check, scenario replay,
and supervised safety testing. The current repository contains no real camera calibration.
