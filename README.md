# DA-SPL

An independent reconstruction of the published IEEE DA-SPL report-generation
method. This workspace has only two project tracks:

- [Paper Method](experiments/paper_method/README.md): active, independently
  implemented ConViT + DAM + PLN + LEM image-input core.
- [Professor Validation](experiments/professor_validation/README.md): reserved;
  no follow-up data processing or experiments are implemented yet.

The authority is the local published IEEE PDF, DOI
`10.1109/BIBE66822.2025.00121`. Read the
[method contract](docs/paper_method_contract.md) before training. The paper
omits or ambiguously specifies several implementation details. This is an
explicitly qualified reconstruction, **not a verified full three-modality or
historical ten-fold reproduction**.

## Experiment Tracking

[Results and decisions](docs/experiment_log.md) records completed experiments,
their interpretation and limitations, and dated interim results. The active
follow-up is the controlled [30-epoch LEM-off run](experiments/paper_method/README.md#longer-lem-off-training).
Its status is tracked separately from completed five-epoch experiments.

## Initial Run

Synthetic tests, without pretrained inference or optimizer steps:

```bash
bash experiments/paper_method/test.sh
```

Check the existing data, imports and cached backbone without building a model:

```bash
bash experiments/paper_method/run.sh --check
```

After acknowledging the documented reconstruction choices, on an allocated GPU:

```bash
bash experiments/paper_method/run.sh --accept-reconstruction
```

The default is a fresh five-epoch debugging run, then evaluation. Output:
`artifacts/paper_method/image_core_seed123/`. Existing runs are never overwritten;
use `--run-dir NEW_DIRECTORY` for a separate run. Actual training and inference
are launched by the user, not as part of the assistant's unit tests.

The old GitHub source, repair branches, shared legacy runtime, old preprocessing
scripts and old tests were removed from the live project. Previous results,
weights and their source archives remain under `artifacts/` for historical
comparison. They are not imported or loadable by the new framework. Raw data,
processed data, pretrained cache, PDFs and the installed environment are retained.

The existing processed split is deliberately unchanged; its report format and
split are reconstructed, not asserted to be the original paper protocol. The
new framework does not regenerate, silently rewrite or relabel it.

See [Project Structure](docs/project_structure.md) for module boundaries.

## Environment

`environment.yml` is the only maintained environment specification. The existing
`da-spl-repro` environment is retained without rebuilding. For a fresh Linux setup:

```bash
PIP_NO_BUILD_ISOLATION=0 conda env create -f environment.yml
conda activate da-spl-repro
```

The pip setting lets legacy evaluation packages use the build dependencies
installed by Conda. A fresh environment needs network access, Git and a C/C++
compiler. Entrypoints use local imports; no editable install or second package
configuration is needed.

## Citation

```bibtex
@INPROCEEDINGS{DA-SPL,
  author={Huang, Cheng and Xie, Weizheng and Han, Zeyu and Lee, Tsengdar and Kooner, Karanjit and Wang, Jui-Kai and Zhang, Ning and Zhang, Jia},
  booktitle={2025 IEEE 25th International Conference on Bioinformatics and Bioengineering (BIBE)},
  title={Automated Glaucoma Report Generation via Dual-Attention Semantic Parallel-LSTM and Multimodal Clinical Data Integration},
  year={2025},
  pages={698-705},
  doi={10.1109/BIBE66822.2025.00121}}
```
