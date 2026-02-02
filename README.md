# &#x20;#Trauma-Former: Real-Time Dynamic Prediction of Trauma-Induced Coagulopathy

# 

# &#x20;##📖 Citation

# If you use this code in your research, please cite our paper:

# Huang, X., Lin, W.. Real-Time Dynamic Prediction of Trauma-Induced Coagulopathy via a 5G-Enabled Digital Twin Framework: A Time-Series Transformer Approach. 



# &#x20;##🔬 Overview

# This repository contains the complete implementation of the Trauma-Former model and the 5G-enabled Digital Twin framework described in our paper. The system includes:

# 1\. \*\*Synthetic Data Generation Engine: Creates physiologically-informed trauma patient trajectories (N=1,240)

# 2\. \*\*Trauma-Former Model: Time-Series Transformer for real-time TIC prediction

# 3\. \*\*Baseline Models: LSTM and Shock Index implementations

# 4\. \*\*In Silico Clinical Trial Framework: Complete validation pipeline

# 

# &#x20;##🚀 Quick Start

# &#x20;### Installation

# ```bash

# &#x20;# Clone repository

# git clone https://github.com/Trauma-Former-Supplementary.git

# cd Trauma-Former-Supplementary

# 

# &#x20;# Install dependencies

# pip install -r requirements.txt

# 

# &#x20;# Or using conda

# conda env create -f environment.yml

# conda activate trauma-former

# ```

# 

# &#x20;Generate Synthetic Dataset

# ```bash

# python data/synthetic\_data\_generator.py --num\_samples 1240 --output\_dir ./data/synthetic

# ```

# 

# &#x20;Train the Model

# ```bash

# python train.py --config experiments/train\_config.yaml

# ```

# 

# &#x20;Run Inference Simulation

# ```bash

# python inference\_simulator.py --config experiments/inference\_config.yaml

# ```

# 

# &#x20;📊 Model Performance (From Paper)

# | Metric | Trauma-Former | LSTM | Shock Index |

# |--------|---------------|------|-------------|

# | AUROC | 0.942 | 0.881 | 0.785 |

# | Early Warning Time | 18.4 min | 12.1 min | 5.2 min |

# | System Latency | 47.2 ms | - | - |

# 

# &#x20;🏗️ System Architecture

# 

# &#x20;5G Digital Twin Pipeline

# 1\. Physical Sensing Layer: Simulates ambulance vital sign monitors (1 Hz)

# 2\. Network Transmission Layer: 5G URLLC network slicing simulation

# 3\. Cyber-Physical Layer: Trauma-Former inference engine with Redis state database

# 

# &#x20;Trauma-Former Architecture

# \- Input: 30-second windows of 4 vital signs (HR, SBP, DBP, SpO₂)

# \- Embedding: Linear projection + sinusoidal positional encoding

# \- Transformer: 2 encoder layers, 4 attention heads (d\_model=128)

# \- Output: Sigmoid activation for TIC probability

# 

# &#x20;📁 Repository Structure

# ```

# Trauma-Former-Supplementary/

# ├── README.md                           This file

# ├── LICENSE                             MIT License

# ├── requirements.txt                    Python dependencies

# ├── environment.yml                     Conda environment

# ├── config.yaml                         Main configuration file

# ├── train.py                            Main training script

# ├── inference\_simulator.py              Real-time inference simulator

# ├── data/

# │   ├── synthetic\_data\_generator.py     Synthetic data generation engine

# │   └── preprocessor.py                 Data preprocessing utilities

# ├── models/

# │   ├── trauma\_former.py                Trauma-Former model implementation

# │   ├── lstm\_baseline.py                LSTM baseline model

# │   ├── shock\_index.py                  Shock Index calculator

# │   └── attention\_visualizer.py         Attention visualization tools

# ├── experiments/

# │   ├── train\_config.yaml               Training configuration

# │   ├── inference\_config.yaml           Inference configuration

# │   └── robustness\_test\_config.yaml     Robustness testing configuration

# ├── utils/

# │   ├── metrics.py                      Evaluation metrics

# │   ├── data\_loader.py                  Data loading utilities

# │   └── system\_monitor.py               System performance monitoring

# ├── notebooks/

# │   ├── 01\_data\_generation\_and\_validation.ipynb

# │   ├── 02\_model\_training.ipynb

# │   └── 03\_results\_visualization.ipynb

# ├── tests/

# │   ├── test\_data\_generation.py

# │   ├── test\_model\_inference.py

# │   └── test\_system\_latency.py

# ├── figures/

# │   └── generate\_figures.py             Scripts to regenerate paper figures

# └── deployment/

# &#x20;   ├── edge\_simulator.py               Edge device simulator

# &#x20;   ├── redis\_config.yaml               Redis database configuration

# &#x20;   └── docker-compose.yml              Docker deployment configuration

# ```

# 

# &#x20;🔧 Requirements

# \- Python 3.8+

# \- PyTorch 1.12.0+

# \- CUDA 11.3+ (for GPU acceleration)

# \- Redis 6.0+ (for state database simulation)

# 

# Detailed requirements in `requirements.txt`:

# ```

# &#x20;Core dependencies

# torch==2.1.0

# torchvision==0.16.0

# torchaudio==2.1.0

# 

# &#x20;Data processing

# numpy==1.24.3

# pandas==2.0.3

# scipy==1.11.3

# scikit-learn==1.3.0

# 

# &#x20;Visualization

# matplotlib==3.7.2

# seaborn==0.12.2

# plotly==5.17.0

# 

# &#x20;Utilities

# tqdm==4.65.0

# pyyaml==6.0

# joblib==1.3.1

# 

# &#x20;Database (for Redis simulation)

# redis==5.0.0

# 

# &#x20;Testing

# pytest==7.4.2

# pytest-cov==4.1.0

# 

# &#x20;Notebooks

# jupyter==1.0.0

# ipykernel==6.25.1

# ```

# 

# &#x20;📖 Detailed Documentation

# 

# &#x20;Data Generation

# The synthetic data generation engine (`data/synthetic\_data\_generator.py`) creates physiologically-informed trauma patient trajectories based on statistical distributions derived from 600 real-world trauma cases. The generated dataset includes:

# 

# \- 1,240 patient trajectories (620 TIC-positive, 620 control)

# \- 4 vital signs: Heart rate (HR), systolic blood pressure (SBP), diastolic blood pressure (DBP), SpO₂

# \- 30-second windows at 1 Hz sampling rate

# \- Realistic noise and artifacts to simulate prehospital conditions

# 

# &#x20;Model Training

# The training script (`train.py`) implements the complete training pipeline:

# 

# 1\. Data loading and preprocessing

# 2\. Model initialization (Trauma-Former or LSTM baseline)

# 3\. Training with early stopping

# 4\. Evaluation on test set

# 5\. Model checkpointing and results logging

# 

# &#x20;Inference Simulation

# The inference simulator (`inference\_simulator.py`) simulates the real-time 5G-enabled Digital Twin framework:

# 

# 1\. Edge simulation: Simulates ambulance vital sign monitors

# 2\. Network simulation: 5G URLLC with configurable latency and reliability

# 3\. Digital Twin engine: Real-time inference with Redis state persistence

# 4\. Performance monitoring: Latency, accuracy, and system metrics

# 

# &#x20;Configuration Management

# All experiments are configured via YAML files in the `experiments/` directory:

# 

# \- `train\_config.yaml`: Training hyperparameters

# \- `inference\_config.yaml`: Inference simulation parameters

# \- `robustness\_test\_config.yaml`: Robustness testing configurations

# 

# &#x20;🔍 Model Interpretability

# 

# Trauma-Former includes built-in interpretability features:

# 

# 1\. Attention Visualization: Visualize which time points and features the model attends to

# 2\. Feature Importance: Understand the contribution of each vital sign to predictions

# 3\. Uncertainty Estimation: Monte Carlo dropout for prediction confidence intervals

# 

# &#x20;📊 Performance Validation

# 

# &#x20;Validation Metrics

# \- AUROC: Area Under Receiver Operating Characteristic Curve

# \- Early Warning Time: Time between model alert and simulated deterioration

# \- System Latency: End-to-end latency from sensing to alert

# \- Robustness: Performance under noise, missing data, and sensor faults

# 

# &#x20;Reproducibility

# All experiments are fully reproducible with fixed random seeds. Configuration files capture all hyperparameters and experimental settings.

# 

# &#x20;🚀 Advanced Usage

# 

# &#x20;Custom Training

# ```bash

# &#x20;Train with custom parameters

# python train.py --config experiments/train\_config.yaml \\

# &#x20;   --data\_dir ./data/custom\_dataset \\

# &#x20;   --model\_name custom\_trauma\_former \\

# &#x20;   --batch\_size 64 \\

# &#x20;   --learning\_rate 0.0005

# ```

# 

# &#x20;Robustness Testing

# ```bash

# &#x20;Test model robustness to data imperfections

# python -m tests.test\_model\_inference \\

# &#x20;   --model\_path ./results/best\_model.pth \\

# &#x20;   --noise\_level 0.1 \\

# &#x20;   --missing\_data\_ratio 0.3

# ```

# 

# &#x20;Real-time Simulation

# ```bash

# &#x20;Run extended simulation with detailed logging

# python inference\_simulator.py \\

# &#x20;   --model ./results/best\_model.pth \\

# &#x20;   --duration 60 \\

# &#x20;   --output\_dir ./simulation\_results\_detailed \\

# &#x20;   --log\_level DEBUG

# ```

# 

# &#x20;🛠️ Development

# 

# &#x20;Running Tests

# ```bash

# &#x20;Run all tests

# pytest tests/

# 

# &#x20;Run specific test module

# pytest tests/test\_data\_generation.py

# 

# &#x20;Run tests with coverage report

# pytest tests/ --cov=src --cov-report=html

# ```

# 

# &#x20;Code Style

# This project follows PEP 8 guidelines. Use the following tools for code quality:

# 

# ```bash

# &#x20;Format code with black

# black .

# 

# &#x20;Check code style with flake8

# flake8 .

# 

# &#x20;Sort imports with isort

# isort .

# ```

# 

# &#x20;📈 Results Reproduction

# 

# To reproduce the results from the paper:

# 

# 1\. Generate the synthetic dataset with default parameters

# 2\. Train Trauma-Former using the provided configuration

# 3\. Evaluate performance using the test set

# 4\. Run robustness tests to validate model resilience

# 5\. Simulate real-time inference to measure system latency

# 

# Expected results should match those reported in Table 2 of the paper.

# 

# &#x20;🤝 Contributing

# 

# We welcome contributions to improve this repository. Please follow these steps:

# 

# 1\. Fork the repository

# 2\. Create a feature branch (`git checkout -b feature/improvement`)

# 3\. Commit your changes (`git commit -m 'Add some improvement'`)

# 4\. Push to the branch (`git push origin feature/improvement`)

# 5\. Open a Pull Request

# 

# &#x20;📝 License

# 

# This project is licensed under the MIT License - see the \[LICENSE](LICENSE) file for details.

# 

# &#x20;🙏 Acknowledgments

# 

# This research was supported by the Fujian Medical University QiHang Fund (Grant number: 2023QH1263).

# 

# We thank the emergency medicine staff and the information technology team at The Second Affiliated Hospital of Fujian Medical University for their clinical and technical support.

# 

# &#x20;📚 References

# 

# See the paper for complete references. 



# 

# &#x20;📧 Contact

# 

# For questions about the code or methodology, please contact:

# 

# \- Dr. Wenjia Lin: Department of Emergency Medicine, The Second Affiliated Hospital of Fujian Medical University

# &#x20; - Email: DoctorLin1990@163.com

# &#x20; - Address: No. 34, Zhongshan North Road, Quanzhou, Fujian 362000, China

# 

# \- GitHub Issues: \[Link to your repository issues]

# 

# &#x20;🐛 Known Issues and Limitations

# 

# 1\. Synthetic Data: While statistically faithful, synthetic data cannot fully capture biological heterogeneity

# 2\. Limited Features: Current implementation uses only 4 vital signs; additional parameters may improve performance

# 3\. Computational Requirements: GPU acceleration recommended for training and real-time simulation

# 

# &#x20;🔄 Changelog

# 

# &#x20;v1.0.0 (2026-01-01)

# \- Initial release of Trauma-Former supplementary materials

# \- Complete implementation of synthetic data generation engine

# \- Trauma-Former model with multi-head attention

# \- Real-time inference simulation framework

# \- Comprehensive testing and validation scripts

# 

# \---

# 

# Note: This repository is for research purposes only. The models and code are not intended for direct clinical use without further validation and regulatory approval.

# 

# \---

# 

# Last updated: January 2026  

# Version: 1.0.0  

# Maintainers: Xiaolei Huang, Wenjia Lin

