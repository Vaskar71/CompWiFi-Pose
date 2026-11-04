# Person-in-WiFi-3D Dataset Audit

- Data root: `J:\Zero-shot 3D Multi-person Pose Estimation\data\person_in_wifi_3d\wifi_dataset`
- Requested report: `J:\Zero-shot 3D Multi-person Pose Estimation\docs\DATA_AUDIT.md`
- Status: **read-only audit completed; per-file findings and errors are reported below**.
- Array content was bounded to at most 12 selected IDs per split. List parsing and expected-file existence checks cover every unique listed ID; value statistics are not dataset-wide.
- No dataset files were downloaded, created, modified, preprocessed, or used for training.

## Actual directory inventory

- Files found: **293312**.
- Extensions: `.mat`: 97770, `.npy`: 97770, `.npz`: 97770, `.txt`: 2.

| Directory | Files | Extensions | Example files |
| --- | --- | --- | --- |
| `.` | 0 | none | none |
| `test_data` | 1 | .txt=1 | `test_data\test_data_list.txt` |
| `test_data\csi` | 7824 | .mat=7824 | `test_data\csi\S11_01_308.mat`, `test_data\csi\S11_01_309.mat`, `test_data\csi\S11_01_310.mat` |
| `test_data\keypoint` | 7824 | .npy=7824 | `test_data\keypoint\S11_01_308.npy`, `test_data\keypoint\S11_01_309.npy`, `test_data\keypoint\S11_01_310.npy` |
| `test_data\smpl` | 7824 | .npz=7824 | `test_data\smpl\S11_01_308.npz`, `test_data\smpl\S11_01_309.npz`, `test_data\smpl\S11_01_310.npz` |
| `train_data` | 1 | .txt=1 | `train_data\train_data_list.txt` |
| `train_data\csi` | 89946 | .mat=89946 | `train_data\csi\S11_01_10.mat`, `train_data\csi\S11_01_11.mat`, `train_data\csi\S11_01_12.mat` |
| `train_data\keypoint` | 89946 | .npy=89946 | `train_data\keypoint\S11_01_10.npy`, `train_data\keypoint\S11_01_11.npy`, `train_data\keypoint\S11_01_12.npy` |
| `train_data\smpl` | 89946 | .npz=89946 | `train_data\smpl\S11_01_10.npz`, `train_data\smpl\S11_01_11.npz`, `train_data\smpl\S11_01_12.npz` |

## Train/test lists and expected-file coverage


### train

- Split directory: `J:\Zero-shot 3D Multi-person Pose Estimation\data\person_in_wifi_3d\wifi_dataset\train_data`
- List path: `J:\Zero-shot 3D Multi-person Pose Estimation\data\person_in_wifi_3d\wifi_dataset\train_data\train_data_list.txt`
- Physical lines: 89946; listed entries: **89946**; unique IDs: **89946**; duplicate entries: 0.
- Blank lines: none; lines with extra tokens: none.
- Representative listed IDs: `S11_01_10`, `S11_01_11`, `S11_01_12`, `S11_01_13`, `S11_01_14`, `S11_01_15` (+89940 more)
- IDs unsafe to resolve beneath the split: none.
- IDs not matching the feeder's sequence/frame convention: none.
- Filename-derived person-count distribution (unique IDs): 1: 28121, 2: 36242, 3: 25583.

| Artifact | Directory | Expected | Present | Missing | Unlisted | Missing examples |
| --- | --- | --- | --- | --- | --- | --- |
| raw CSI | yes | 89946 | 89946 | 0 | 0 | none |
| csi_ap | no | 89946 | 0 | 89946 | 0 | `S11_01_10`, `S11_01_11`, `S11_01_12`, `S11_01_13` (+89942 more) |
| amplitude | no | 89946 | 0 | 89946 | 0 | `S11_01_10`, `S11_01_11`, `S11_01_12`, `S11_01_13` (+89942 more) |
| phase | no | 89946 | 0 | 89946 | 0 | `S11_01_10`, `S11_01_11`, `S11_01_12`, `S11_01_13` (+89942 more) |
| keypoint | yes | 89946 | 89946 | 0 | 0 | none |

- Listed IDs missing keypoint files: **0**; examples: none.
- Listed IDs with raw CSI or csi_ap but no keypoint: **0**; examples: none.

### test

- Split directory: `J:\Zero-shot 3D Multi-person Pose Estimation\data\person_in_wifi_3d\wifi_dataset\test_data`
- List path: `J:\Zero-shot 3D Multi-person Pose Estimation\data\person_in_wifi_3d\wifi_dataset\test_data\test_data_list.txt`
- Physical lines: 7824; listed entries: **7824**; unique IDs: **7824**; duplicate entries: 0.
- Blank lines: none; lines with extra tokens: none.
- Representative listed IDs: `S11_01_308`, `S11_01_309`, `S11_01_310`, `S11_01_311`, `S11_01_312`, `S11_01_313` (+7818 more)
- IDs unsafe to resolve beneath the split: none.
- IDs not matching the feeder's sequence/frame convention: none.
- Filename-derived person-count distribution (unique IDs): 1: 2586, 2: 3184, 3: 2054.

| Artifact | Directory | Expected | Present | Missing | Unlisted | Missing examples |
| --- | --- | --- | --- | --- | --- | --- |
| raw CSI | yes | 7824 | 7824 | 0 | 0 | none |
| csi_ap | no | 7824 | 0 | 7824 | 0 | `S11_01_308`, `S11_01_309`, `S11_01_310`, `S11_01_311` (+7820 more) |
| amplitude | no | 7824 | 0 | 7824 | 0 | `S11_01_308`, `S11_01_309`, `S11_01_310`, `S11_01_311` (+7820 more) |
| phase | no | 7824 | 0 | 7824 | 0 | `S11_01_308`, `S11_01_309`, `S11_01_310`, `S11_01_311` (+7820 more) |
| keypoint | yes | 7824 | 7824 | 0 | 0 | none |

- Listed IDs missing keypoint files: **0**; examples: none.
- Listed IDs with raw CSI or csi_ap but no keypoint: **0**; examples: none.

## Sampled CSI and pose array content

NaN/Inf counts apply to the stored NumPy arrays and to the processor-interpreted complex raw arrays, as identified by each row. Ranges exclude NaN/Inf values.

### train sampled arrays

Selected 12 unique IDs: `S11_01_10`, `S11_20_506`, `S12_01_141`, `S12_27_53`, `S13_07_100`, `S13_47_441`, `S21_30_172`, `S22_32_525`, `S23_14_225`, `S23_40_279` (+2 more). Statistics below apply only to successfully read sampled files.

| Artifact | ID | Stored shape | Stored dtype | Interpreted shape | Finite value range | NaN | Inf | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw CSI | `S11_01_10` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-43, 43]; imag [-42, 42]; magnitude [0, 43.9659] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S11_20_506` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-44, 44]; imag [-44, 45]; magnitude [1, 45.8039] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S12_01_141` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-43, 43]; imag [-44, 44]; magnitude [0, 45.5412] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S12_27_53` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-45, 47]; imag [-44, 44]; magnitude [1, 47.3814] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S13_07_100` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-44, 43]; imag [-44, 43]; magnitude [0, 46.8615] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S13_47_441` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-48, 47]; imag [-50, 48]; magnitude [0, 50.4777] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S21_30_172` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-56, 58]; imag [-58, 58]; magnitude [0, 61.9112] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S22_32_525` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-57, 54]; imag [-59, 57]; magnitude [0, 62.9365] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S23_14_225` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-57, 56]; imag [-54, 53]; magnitude [1, 61.2944] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S23_40_279` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-64, 65]; imag [-63, 63]; magnitude [1, 66.1287] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S51_21_46` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-40, 42]; imag [-43, 40]; magnitude [6.08276, 43.4166] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S52_14_60` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-46, 45]; imag [-43, 49]; magnitude [3, 50.6952] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| keypoint | `S11_01_10` | `(1, 14, 3)` | `float64` | unknown | [0.545841, 4.03656] | 0 | 0 | none |
| keypoint | `S11_20_506` | `(1, 14, 3)` | `float64` | unknown | [1.02706, 3.99566] | 0 | 0 | none |
| keypoint | `S12_01_141` | `(2, 14, 3)` | `float64` | unknown | [0.513261, 4.05146] | 0 | 0 | none |
| keypoint | `S12_27_53` | `(2, 14, 3)` | `float64` | unknown | [0.542038, 4.10781] | 0 | 0 | none |
| keypoint | `S13_07_100` | `(3, 14, 3)` | `float64` | unknown | [0.288777, 4.01467] | 0 | 0 | none |
| keypoint | `S13_47_441` | `(3, 14, 3)` | `float64` | unknown | [0.215062, 4.03137] | 0 | 0 | none |
| keypoint | `S21_30_172` | `(1, 14, 3)` | `float64` | unknown | [0.559954, 4.05204] | 0 | 0 | none |
| keypoint | `S22_32_525` | `(2, 14, 3)` | `float64` | unknown | [0.335605, 4.05103] | 0 | 0 | none |
| keypoint | `S23_14_225` | `(3, 14, 3)` | `float64` | unknown | [0.410493, 4.04327] | 0 | 0 | none |
| keypoint | `S23_40_279` | `(3, 14, 3)` | `float64` | unknown | [-0.171579, 4.06135] | 0 | 0 | none |
| keypoint | `S51_21_46` | `(1, 14, 3)` | `float64` | unknown | [1.51907, 3.99478] | 0 | 0 | none |
| keypoint | `S52_14_60` | `(2, 14, 3)` | `float64` | unknown | [0.673312, 4.10025] | 0 | 0 | none |

### test sampled arrays

Selected 12 unique IDs: `S11_01_308`, `S11_19_313`, `S12_01_331`, `S12_22_342`, `S13_07_304`, `S13_44_333`, `S21_26_323`, `S22_29_302`, `S23_05_320`, `S23_35_314` (+2 more). Statistics below apply only to successfully read sampled files.

| Artifact | ID | Stored shape | Stored dtype | Interpreted shape | Finite value range | NaN | Inf | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw CSI | `S11_01_308` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-44, 45]; imag [-42, 43]; magnitude [0, 45.8039] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S11_19_313` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-43, 42]; imag [-43, 43]; magnitude [1.41421, 45.2769] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S12_01_331` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-41, 41]; imag [-42, 42]; magnitude [0, 45.2548] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S12_22_342` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-49, 48]; imag [-50, 49]; magnitude [0, 52.4976] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S13_07_304` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-45, 45]; imag [-46, 45]; magnitude [1.41421, 48.8467] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S13_44_333` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-42, 45]; imag [-42, 45]; magnitude [0, 46.6476] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S21_26_323` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-56, 57]; imag [-62, 59]; magnitude [1, 63.561] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S22_29_302` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-58, 57]; imag [-59, 60]; magnitude [0, 68.9638] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S23_05_320` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-62, 60]; imag [-63, 61]; magnitude [0, 68.4471] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S23_35_314` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-67, 63]; imag [-69, 60]; magnitude [0, 73.736] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S51_07_326` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-51, 46]; imag [-45, 46]; magnitude [1, 52.4786] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| raw CSI | `S52_04_345` | `(20, 30, 3, 3)` | `[('real', '<f8'), ('imag', '<f8')]` | `(3, 3, 30, 20)` | real [-46, 46]; imag [-46, 47]; magnitude [1, 49.8197] | 0 | 0 | none; csi_out attrs: {'MATLAB_class': "b'double'"} |
| keypoint | `S11_01_308` | `(1, 14, 3)` | `float64` | unknown | [0.190047, 4.0354] | 0 | 0 | none |
| keypoint | `S11_19_313` | `(1, 14, 3)` | `float64` | unknown | [1.09026, 4.03701] | 0 | 0 | none |
| keypoint | `S12_01_331` | `(2, 14, 3)` | `float64` | unknown | [0.680135, 4.04834] | 0 | 0 | none |
| keypoint | `S12_22_342` | `(2, 14, 3)` | `float64` | unknown | [0.648836, 4.01935] | 0 | 0 | none |
| keypoint | `S13_07_304` | `(3, 14, 3)` | `float64` | unknown | [0.397644, 4.03593] | 0 | 0 | none |
| keypoint | `S13_44_333` | `(3, 14, 3)` | `float64` | unknown | [0.228897, 4.10975] | 0 | 0 | none |
| keypoint | `S21_26_323` | `(1, 14, 3)` | `float64` | unknown | [1.15062, 3.98843] | 0 | 0 | none |
| keypoint | `S22_29_302` | `(2, 14, 3)` | `float64` | unknown | [0.216362, 4.04842] | 0 | 0 | none |
| keypoint | `S23_05_320` | `(3, 14, 3)` | `float64` | unknown | [0.426398, 4.04464] | 0 | 0 | none |
| keypoint | `S23_35_314` | `(3, 14, 3)` | `float64` | unknown | [0.0544684, 4.02134] | 0 | 0 | none |
| keypoint | `S51_07_326` | `(1, 14, 3)` | `float64` | unknown | [0.728149, 4.01427] | 0 | 0 | none |
| keypoint | `S52_04_345` | `(2, 14, 3)` | `float64` | unknown | [0.509406, 3.95952] | 0 | 0 | none |

## Feeder normalization checks


### train feeder normalization inputs

No sampled `csi_ap` arrays were available for normalization checks.

### test feeder normalization inputs

No sampled `csi_ap` arrays were available for normalization checks.

## Pose array structure and conventions


### train pose structure

| ID | Pose shape | People | Joints | Coordinates | Matches filename count | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `S11_01_10` | `(1, 14, 3)` | 1 | 14 | 3 | yes | none |
| `S11_20_506` | `(1, 14, 3)` | 1 | 14 | 3 | yes | none |
| `S12_01_141` | `(2, 14, 3)` | 2 | 14 | 3 | yes | none |
| `S12_27_53` | `(2, 14, 3)` | 2 | 14 | 3 | yes | none |
| `S13_07_100` | `(3, 14, 3)` | 3 | 14 | 3 | yes | none |
| `S13_47_441` | `(3, 14, 3)` | 3 | 14 | 3 | yes | none |
| `S21_30_172` | `(1, 14, 3)` | 1 | 14 | 3 | yes | none |
| `S22_32_525` | `(2, 14, 3)` | 2 | 14 | 3 | yes | none |
| `S23_14_225` | `(3, 14, 3)` | 3 | 14 | 3 | yes | none |
| `S23_40_279` | `(3, 14, 3)` | 3 | 14 | 3 | yes | none |
| `S51_21_46` | `(1, 14, 3)` | 1 | 14 | 3 | yes | none |
| `S52_14_60` | `(2, 14, 3)` | 2 | 14 | 3 | yes | none |

- Sampled joint-count distribution: {14: 12}.
- Sampled coordinate-dimension distribution: {3: 12}.
- Coordinate units: **unknown** unless independently established by validated dataset documentation.
- Joint ordering, axis directions, coordinate origin, and world/camera/body convention: **unknown** unless independently established by validated dataset documentation.

### test pose structure

| ID | Pose shape | People | Joints | Coordinates | Matches filename count | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `S11_01_308` | `(1, 14, 3)` | 1 | 14 | 3 | yes | none |
| `S11_19_313` | `(1, 14, 3)` | 1 | 14 | 3 | yes | none |
| `S12_01_331` | `(2, 14, 3)` | 2 | 14 | 3 | yes | none |
| `S12_22_342` | `(2, 14, 3)` | 2 | 14 | 3 | yes | none |
| `S13_07_304` | `(3, 14, 3)` | 3 | 14 | 3 | yes | none |
| `S13_44_333` | `(3, 14, 3)` | 3 | 14 | 3 | yes | none |
| `S21_26_323` | `(1, 14, 3)` | 1 | 14 | 3 | yes | none |
| `S22_29_302` | `(2, 14, 3)` | 2 | 14 | 3 | yes | none |
| `S23_05_320` | `(3, 14, 3)` | 3 | 14 | 3 | yes | none |
| `S23_35_314` | `(3, 14, 3)` | 3 | 14 | 3 | yes | none |
| `S51_07_326` | `(1, 14, 3)` | 1 | 14 | 3 | yes | none |
| `S52_04_345` | `(2, 14, 3)` | 2 | 14 | 3 | yes | none |

- Sampled joint-count distribution: {14: 12}.
- Sampled coordinate-dimension distribution: {3: 12}.
- Coordinate units: **unknown** unless independently established by validated dataset documentation.
- Joint ordering, axis directions, coordinate origin, and world/camera/body convention: **unknown** unless independently established by validated dataset documentation.

### Potential metadata evidence

No readable metadata snippets matching unit, coordinate, joint, or skeleton terms were found. Units and conventions remain **unknown**.

## Filename-derived sequence identifiers and frame continuity

Candidate sequence keys are the first two underscore-separated filename fields, matching the feeder's next-ID construction. Their recording, subject, and environment semantics are **unknown** without explicit metadata.

| Split | Candidate groups | Group size range | Unparseable IDs | Duplicate frame indices | Groups with gaps | Missing interior indices |
| --- | --- | --- | --- | --- | --- | --- |
| train | 215 | 46–539 | 0 | 0 | 213 | 28481 |
| test | 195 | 3–49 | 0 | 0 | 48 | 597 |

- train candidate keys: `S11_01`, `S11_02`, `S11_03`, `S11_04`, `S11_05`, `S11_06`, `S11_07`, `S11_09` (+207 more).
- train gap examples: `S11_01: [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]`, `S11_02: [156, 157, 158, 199, 200, 201, 202, 203, 204, 205]`, `S11_03: [300, 301, 302, 303, 304, 305, 306, 307, 308, 309]`, `S11_04: [300, 301, 302, 303, 304, 305, 306, 307, 308, 309]`, `S11_05: [300, 301, 302, 303, 304, 305, 306, 307, 308, 309]` (+208 more).
- Training entries excluded by the feeder's immediate-next-ID rule on the unfiltered list: **2032**; examples: `S11_01_15`, `S11_01_215`, `S11_01_249`, `S11_01_276`, `S11_01_443`, `S11_01_515` (+2026 more). Experiment-specific person-count filtering can change this count.

- test candidate keys: `S11_01`, `S11_02`, `S11_03`, `S11_04`, `S11_05`, `S11_06`, `S11_07`, `S11_09` (+187 more).
- test gap examples: `S11_18: [337, 338, 339, 340]`, `S11_28: [333, 334, 335]`, `S12_03: [325]`, `S12_09: [333, 337, 338, 340, 341, 342]`, `S12_12: [328, 329, 330, 345]` (+43 more).
- Test entries are not removed by an immediate-next-ID requirement in the current feeder.

Candidate sequence-key overlap between train and test: **195**; examples: `S11_01`, `S11_02`, `S11_03`, `S11_04`, `S11_05`, `S11_06`, `S11_07`, `S11_09` (+187 more).
Exact listed-ID overlap between train and test: **0**; examples: none.

## Empirical process.py output versus feeder input

For sampled IDs with amplitude, phase, and csi_ap all present, the audit concatenates amplitude and phase in memory on axis 1 and compares them with csi_ap. `allclose` uses `rtol=1e-5`, `atol=1e-8`, and equal NaN positions. No processor or feeder code is executed.

No complete sampled amplitude/phase/csi_ap triplets were available; empirical compatibility is **not measured**. In particular, whether `csi_ap` exactly equals concatenated amplitude and phase is **unknown for this extracted dataset**. The current files are not feeder-ready because `csi_ap` is absent. No preprocessing was run to resolve that absence.

## Code-derived findings (not real-data measurements)

- `feeder/person_in_wifi_3d.py:58-59` expects `csi_ap/<ID>.npy` and `keypoint/<ID>.npy` beneath each split.
- `data/person_in_wifi_3d/process.py:77-87` reads HDF5 `csi/<ID>.mat` variable `csi_out`, constructs amplitude and sanitized phase, and concatenates them on axis 1.
- `data/person_in_wifi_3d/process.py:89-102` contains enabled saves for `csi_amplitude`, `csi_phase`, and the combined `csi_ap`. This is a source-code capability only: the processor was not run, and no generated output was validated.
- The processor's hard-coded reshape implies a code-level `(3, 90, 20)` amplitude/phase layout and `(3, 180, 20)` combined layout. These are expectations, not verified on-disk processed shapes.
- `feeder/person_in_wifi_3d.py:114-126` normalizes the first 90 and remaining channels independently. Its current guard maps a constant component to zeros. The frequency of constant processed amplitude/phase inputs is **unknown**, because none of those arrays exists in the extracted dataset and no processed arrays were generated.
- `feeder/person_in_wifi_3d.py:62-73` keeps a training item only when its immediately following ID is in the filtered list; `feeder/person_in_wifi_3d.py:99-111` does not apply that rule to test items.
- `feeder/person_in_wifi_3d.py:151-157` pads poses to three people, 14 joints, and three coordinates. The sampled real arrays match the per-person `(14, 3)` portion, but this does not establish joint ordering, units, or coordinate conventions.
- Filename-derived person counts in this report use the feeder's third-character convention and are audit-only. This script is not imported by any training or unlabelled loader; these counts must never become an unlabelled-training signal.

## Audit interpretation and readiness

**Readiness: YELLOW.** The raw dataset is structurally complete against its lists and the bounded content sample is readable, but it is not yet ready for the existing feeder or for claims about recording-disjoint evaluation and pose conventions.

Positive measured evidence:

- Train has **89,946** unique listed IDs and test has **7,824**; neither list has duplicates, blank lines, extra tokens, unsafe IDs, or IDs outside the expected filename pattern.
- Filename-derived one-/two-/three-person counts are **28,121 / 36,242 / 25,583** for train and **2,586 / 3,184 / 2,054** for test. All 24 sampled pose arrays have person dimensions matching those filename counts. This is sampled agreement, not dataset-wide pose-label validation.
- Every listed raw CSI and keypoint file is present, with **0 missing and 0 unlisted** in either split. There are **0 listed CSI-without-keypoint pairs**.
- All 24 sampled raw CSI files have stored shape `(20, 30, 3, 3)`, compound little-endian float64 `real`/`imag` fields, and audit-interpreted shape `(3, 3, 30, 20)` after the script's transpose and complex128 reconstruction. Across the 24 samples, real values span `[-67, 65]`, imaginary values span `[-69, 63]`, and magnitudes span `[0, 73.736]`.
- All 24 sampled keypoint arrays are float64 with shape `(people, 14, 3)`. Their combined sampled value range is `[-0.171579, 4.10975]`. The range alone does not identify units or coordinates.
- NaN and Inf counts are **0 in every sampled raw CSI and keypoint array**. This is not a full-dataset NaN/Inf or corruption scan.

Remaining reasons for yellow rather than green:

- `csi_ap`, `csi_amplitude`, and `csi_phase` directories are absent in both splits; all **89,946 train** and **7,824 test** IDs lack each processed artifact. Existing feeder loading, processed shapes, constant normalization cases, and concatenation equality therefore remain unverified.
- Pose units, joint ordering, axis directions, origin, and world/camera/body conventions are **unknown**. No unit/coordinate/joint metadata was found by the audit, and no visualization-based validation was performed.
- Array contents were checked only for 12 IDs per split. Full-file existence was checked for every listed ID, but full-dataset readability and numerical integrity were not.
- Filename-derived sequence keys are not validated recording, subject, or environment identifiers.

### Next-frame and splitting interpretation

- Candidate sequence keys use the first two filename fields. Train has **215** candidate groups with sizes **46–539**; test has **195** with sizes **3–49**. No frame index is duplicated within a split.
- Train has **213 groups with gaps** and **28,481 missing interior indices**; test has **48 groups with gaps** and **597 missing interior indices**. Thus, consecutive frame numbers are not universally present.
- On the unfiltered train list, **87,914 of 89,946** IDs have an immediate next ID and **2,032** do not; the current training feeder would exclude the latter before any experiment-specific count filtering. Test loading has no equivalent next-ID requirement.
- A provisional filename-grouped train/development split is operationally possible by assigning whole candidate keys to one side. It must not yet be described as recording-, subject-, or environment-disjoint because the key semantics are unknown.
- The supplied official train/test split is **not candidate-sequence-key disjoint**: all **195** test candidate keys also occur in train, although exact listed-ID overlap is **0**. This does not by itself prove sample leakage, but it rules out claiming sequence-key disjointness for the official split.

## Exact commands and outcomes

All commands were run from `J:/Zero-shot 3D Multi-person Pose Estimation` with the existing `C:/Users/vaska/venvs/compwifi-piw3d-py311` environment. `PYTHONDONTWRITEBYTECODE=1` prevented bytecode creation. The audit script and relevant feeder/processor source were also inspected with read-only file tools.

Root confirmation:

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -c "from pathlib import Path; root = Path('data/person_in_wifi_3d/wifi_dataset').resolve(); print('PIW3D_DATA_ROOT:', root); print('root is directory:', root.is_dir()); print('train_data is directory:', (root / 'train_data').is_dir()); print('test_data is directory:', (root / 'test_data').is_dir()); print('train list is file:', (root / 'train_data' / 'train_data_list.txt').is_file()); print('test list is file:', (root / 'test_data' / 'test_data_list.txt').is_file()); assert root.is_dir() and (root / 'train_data').is_dir() and (root / 'test_data').is_dir()"
```

Output confirmed root `J:\Zero-shot 3D Multi-person Pose Estimation\data\person_in_wifi_3d\wifi_dataset`; the root, both split directories, and both list files all exist.

CLI inspection:

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe audit_person_in_wifi_3d.py --help
```

The command exited successfully and showed required `--data_root` and `--output` options plus `--max_samples` (default 6).

Representative-selection check:

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -c "from pathlib import Path; import audit_person_in_wifi_3d as a; root = Path('data/person_in_wifi_3d/wifi_dataset'); exec('for split, dirname in a.SPLITS.items():\n sample_list = a.load_sample_list(root / dirname, split)\n selected = a.select_samples(sample_list[\"unique_ids\"], 12)\n print(split, \"filename-derived counts:\", dict(sorted(sample_list[\"filename_counts\"].items(), key=lambda item: (item[0] is None, str(item[0])))))\n print(split, \"selected IDs:\", selected)\n print(split, \"selected counts:\", [a.filename_info(sample_id)[\"person_count\"] for sample_id in selected])\n assert {1, 2, 3}.issubset({a.filename_info(sample_id)[\"person_count\"] for sample_id in selected})')"
```

The assertions passed. The selected train IDs were `S11_01_10`, `S11_20_506`, `S12_01_141`, `S12_27_53`, `S13_07_100`, `S13_47_441`, `S21_30_172`, `S22_32_525`, `S23_14_225`, `S23_40_279`, `S51_21_46`, and `S52_14_60`. The selected test IDs were `S11_01_308`, `S11_19_313`, `S12_01_331`, `S12_22_342`, `S13_07_304`, `S13_44_333`, `S21_26_323`, `S22_29_302`, `S23_05_320`, `S23_35_314`, `S51_07_326`, and `S52_04_345`. Both sets include all three filename-derived person-count categories.

Real-data audit:

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe audit_person_in_wifi_3d.py --data_root data/person_in_wifi_3d/wifi_dataset --output docs/DATA_AUDIT.md --max_samples 12
```

Output: `Audit report written to: J:\Zero-shot 3D Multi-person Pose Estimation\docs\DATA_AUDIT.md`. Exit code was 0. No preprocessing, training, evaluation, package installation, or Step 4 work was performed.
