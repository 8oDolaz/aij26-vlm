# Tests

```bash
python -m unittest discover -s tests
```

No GPU, model or video needed: the generator tests use synthetic frames and detections, the packer tests tiny fake
checkpoints.

- `test_phases.py`: grasp phase labelling from the gripper signal
- `test_depth_pairs.py`: depth point sampling, gap and stability filters, balancing, markers and prompts
- `test_scene_qa.py`: SAM 3 box filtering, scene building and every question template
- `test_fetch_data.py`: video file selection under a budget
- `test_pack_submission.py`: ZIP layout, bf16 conversion, judge paths and the checks that must refuse a bad submission
