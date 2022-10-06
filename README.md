# Comparative Study Between Distance Measures On Supervised Optimum-Path Forest Classification

*This repository holds all the necessary code to run the very-same experiments described in the paper "Comparative Study Between Distance Measures On Supervised Optimum-Path Forest Classification".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```
@misc{derosa2022comparative,
      title={Comparative Study Between Distance Measures On Supervised Optimum-Path Forest Classification}, 
      author={Gustavo Henrique de Rosa and Mateus Roder and João Paulo Papa},
      year={2022},
      eprint={2202.03854},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

---

## Structure

 * `data`: Folder that holds the input dataset files, already in OPF file format;
 * `outputs`: Folder that holds the output files, such as `.npy`, `.pkl` and `.txt`;
 * `utils`
   * `loader.py`: Utility to load datasets and split them into training, validation and testing sets.
   
---

## Package Guidelines

### Installation

Install all the pre-needed requirements using:

```Python
pip install -r requirements.txt
```

### Data configuration

Please [download](https://www.recogna.tech/files/opf_distance/data.tar.gz) the datasets in the OPF file format and then put them on a `data/` folder.

---

## Usage

### Classify with Optimum-Path Forest

The first step is to classify the data using an OPF-based classifier with a particular distance measure. To accomplish such a step, one needs to use the following script:

```Python
python classify_with_opf.py -h
```

*Note that `-h` invokes the script helper, which assists users in employing the appropriate parameters.*

### Process Classification Report

After conducting the classification task, one needs to process its report into readable outputs. Please, use the following script to accomplish such a procedure:

```Python
python process_report.py -h
```

*Note that this script converts the .pkl reports into readable .txt outputs.*

### (Optinal) Plot Confusion Matrix

After performing the classification task, there is an optinal script that allows to plot its confusion matrix. Please, use the following script to accomplish such a procedure:

```Python
python plot_confusion_matrix.py -h
```

### Bash Script

Instead of invoking every script to conduct the experiments, it is also possible to use the provided shell script, as follows:

```Bash
./pipeline.sh
```

Such a script will conduct every step needed to accomplish the experimentation used throughout this paper. Furthermore, one can change any input argument that is defined in the script.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository.

---
