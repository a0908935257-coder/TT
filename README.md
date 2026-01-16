# trading

Algorithmic trading system with ML-based strategies for automated trading.

## Quick Start

1. **Read CLAUDE.md first** - Contains essential rules for Claude Code
2. Follow the pre-task compliance checklist before starting any work
3. Use proper module structure under `src/main/python/`
4. Commit after every completed task

## Project Structure

```
trading/
├── CLAUDE.md              # Essential rules for Claude Code
├── README.md              # This file
├── .gitignore             # Git ignore patterns
├── src/
│   ├── main/
│   │   ├── python/        # Python source code
│   │   │   ├── core/      # Core trading algorithms
│   │   │   ├── utils/     # Utility functions
│   │   │   ├── models/    # ML model definitions
│   │   │   ├── services/  # Trading services
│   │   │   ├── api/       # API endpoints
│   │   │   ├── training/  # Model training pipelines
│   │   │   ├── inference/ # Prediction/inference code
│   │   │   └── evaluation/# Model evaluation metrics
│   │   └── resources/
│   │       ├── config/    # Configuration files
│   │       ├── data/      # Seed/sample data
│   │       └── assets/    # Static assets
│   └── test/
│       ├── unit/          # Unit tests
│       ├── integration/   # Integration tests
│       └── fixtures/      # Test fixtures
├── data/
│   ├── raw/               # Raw market data
│   ├── processed/         # Processed datasets
│   ├── external/          # External data sources
│   └── temp/              # Temporary processing files
├── notebooks/
│   ├── exploratory/       # Data exploration
│   ├── experiments/       # ML experiments
│   └── reports/           # Analysis reports
├── models/
│   ├── trained/           # Trained model files
│   ├── checkpoints/       # Training checkpoints
│   └── metadata/          # Model metadata
├── experiments/
│   ├── configs/           # Experiment configurations
│   ├── results/           # Experiment results
│   └── logs/              # Training logs
├── docs/                  # Documentation
├── tools/                 # Development tools
├── scripts/               # Automation scripts
├── examples/              # Usage examples
└── output/                # Generated outputs
```

## Development Guidelines

- **Always search first** before creating new files
- **Extend existing** functionality rather than duplicating
- **Use Task agents** for operations >30 seconds
- **Single source of truth** for all functionality
- **Commit frequently** after each completed feature

## Getting Started

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies (when requirements.txt exists)
pip install -r requirements.txt

# Run tests
pytest src/test/
```
