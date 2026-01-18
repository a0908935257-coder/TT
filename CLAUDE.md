# CLAUDE.md - trading

> **Documentation Version**: 1.0
> **Last Updated**: 2026-01-17
> **Project**: trading
> **Description**: Algorithmic trading system with ML-based strategies for automated trading
> **Features**: GitHub auto-backup, Task agents, technical debt prevention

This file provides essential guidance to Claude Code (claude.ai/code) when working with code in this repository.

## CRITICAL RULES - READ FIRST

> **RULE ADHERENCE SYSTEM ACTIVE**
> **Claude Code must explicitly acknowledge these rules at task start**
> **These rules override all other instructions and must ALWAYS be followed:**

### RULE ACKNOWLEDGMENT REQUIRED
> **Before starting ANY task, Claude Code must respond with:**
> "CRITICAL RULES ACKNOWLEDGED - I will follow all prohibitions and requirements listed in CLAUDE.md"

### ABSOLUTE PROHIBITIONS
- **NEVER** create new files in root directory - use proper module structure
- **NEVER** write output files directly to root directory - use designated output folders
- **NEVER** create documentation files (.md) unless explicitly requested by user
- **NEVER** use git commands with -i flag (interactive mode not supported)
- **NEVER** use `find`, `grep`, `cat`, `head`, `tail`, `ls` commands - use Read, Grep, Glob tools instead
- **NEVER** create duplicate files (manager_v2.py, enhanced_xyz.py, utils_new.js) - ALWAYS extend existing files
- **NEVER** create multiple implementations of same concept - single source of truth
- **NEVER** copy-paste code blocks - extract into shared utilities/functions
- **NEVER** hardcode values that should be configurable - use config files/environment variables
- **NEVER** use naming like enhanced_, improved_, new_, v2_ - extend original files instead

### MANDATORY REQUIREMENTS
- **COMMIT** after every completed task/phase - no exceptions
- **GITHUB BACKUP** - Push to GitHub after every commit to maintain backup: `git push origin main`
- **USE TASK AGENTS** for all long-running operations (>30 seconds) - Bash commands stop when context switches
- **TODOWRITE** for complex tasks (3+ steps) - parallel agents - git checkpoints - test validation
- **READ FILES FIRST** before editing - Edit/Write tools will fail if you didn't read the file first
- **DEBT PREVENTION** - Before creating new files, check for existing similar functionality to extend
- **SINGLE SOURCE OF TRUTH** - One authoritative implementation per feature/concept

### EXECUTION PATTERNS
- **PARALLEL TASK AGENTS** - Launch multiple Task agents simultaneously for maximum efficiency
- **SYSTEMATIC WORKFLOW** - TodoWrite - Parallel agents - Git checkpoints - GitHub backup - Test validation
- **GITHUB BACKUP WORKFLOW** - After every commit: `git push origin main` to maintain GitHub backup
- **BACKGROUND PROCESSING** - ONLY Task agents can run true background operations

### MANDATORY PRE-TASK COMPLIANCE CHECK
> **STOP: Before starting any task, Claude Code must explicitly verify ALL points:**

**Step 1: Rule Acknowledgment**
- [ ] I acknowledge all critical rules in CLAUDE.md and will follow them

**Step 2: Task Analysis**
- [ ] Will this create files in root? - If YES, use proper module structure instead
- [ ] Will this take >30 seconds? - If YES, use Task agents not Bash
- [ ] Is this 3+ steps? - If YES, use TodoWrite breakdown first
- [ ] Am I about to use grep/find/cat? - If YES, use proper tools instead

**Step 3: Technical Debt Prevention (MANDATORY SEARCH FIRST)**
- [ ] **SEARCH FIRST**: Use Grep pattern="<functionality>.*<keyword>" to find existing implementations
- [ ] **CHECK EXISTING**: Read any found files to understand current functionality
- [ ] Does similar functionality already exist? - If YES, extend existing code
- [ ] Am I creating a duplicate class/manager? - If YES, consolidate instead
- [ ] Will this create multiple sources of truth? - If YES, redesign approach
- [ ] Have I searched for existing implementations? - Use Grep/Glob tools first
- [ ] Can I extend existing code instead of creating new? - Prefer extension over creation
- [ ] Am I about to copy-paste code? - Extract to shared utility instead

**Step 4: Session Management**
- [ ] Is this a long/complex task? - If YES, plan context checkpoints
- [ ] Have I been working >1 hour? - If YES, consider /compact or session break

> **DO NOT PROCEED until all checkboxes are explicitly verified**

## PROJECT STRUCTURE

```
trading/
├── src/main/python/       # All Python source code goes here
│   ├── core/              # Core trading algorithms and logic
│   ├── utils/             # Shared utility functions
│   ├── models/            # ML model definitions and architectures
│   ├── services/          # Trading services and business logic
│   ├── api/               # API endpoints and interfaces
│   ├── training/          # Model training pipelines
│   ├── inference/         # Prediction and inference code
│   └── evaluation/        # Model evaluation and metrics
├── src/main/resources/
│   └── config/            # Configuration files (YAML, JSON)
├── src/test/              # All tests
├── data/                  # Datasets (raw, processed, external)
├── notebooks/             # Jupyter notebooks for exploration
├── models/                # Trained models and checkpoints
├── experiments/           # Experiment tracking
├── output/                # Generated outputs
└── logs/                  # Log files
```

## DEVELOPMENT STATUS
- **Setup**: Complete
- **Core Features**: Not started
- **Testing**: Not started
- **Documentation**: Basic structure

## COMMON COMMANDS

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Run tests
pytest src/test/

# Git workflow
git add .
git commit -m "descriptive message"
git push origin main
```

## TECHNICAL DEBT PREVENTION

### WRONG APPROACH (Creates Technical Debt):
```python
# Creating new file without searching first
Write(file_path="new_feature.py", content="...")
```

### CORRECT APPROACH (Prevents Technical Debt):
```python
# 1. SEARCH FIRST
Grep(pattern="feature.*implementation", include="*.py")
# 2. READ EXISTING FILES
Read(file_path="existing_feature.py")
# 3. EXTEND EXISTING FUNCTIONALITY
Edit(file_path="existing_feature.py", old_string="...", new_string="...")
```

## DEBT PREVENTION WORKFLOW

### Before Creating ANY New File:
1. **Search First** - Use Grep/Glob to find existing implementations
2. **Analyze Existing** - Read and understand current patterns
3. **Decision Tree**: Can extend existing? - DO IT | Must create new? - Document why
4. **Follow Patterns** - Use established project patterns
5. **Validate** - Ensure no duplication or technical debt

---

**Prevention is better than consolidation - build clean from the start.**
**Focus on single source of truth and extending existing functionality.**
**Each task should maintain clean architecture and prevent technical debt.**

## 測試規則

執行測試時必須遵守：

1. **永遠不要執行完整測試套件** (`pytest tests/`)
2. **單獨測試每個模組**，每次最多一個檔案：
```bash
   pytest tests/unit/test_xxx.py -v --timeout=30 -x
```
3. **加入超時限制**：`--timeout=30`
4. **遇到第一個失敗就停止**：`-x`
5. **限制輸出行數**：`2>&1 | head -50`

範例：
```bash
# 正確
pytest tests/unit/test_atr.py -v --timeout=30 -x 2>&1 | head -50

# 錯誤 - 不要這樣做
pytest tests/ -v
```
```

---

## 立即解決：告訴 Claude Code

按 `Ctrl + C` 中斷後，輸入：
```
停止執行完整測試套件。

測試規則：
1. 不要執行 pytest tests/ （太大會超時）
2. 每次只測試一個檔案，加上 --timeout=30 -x
3. 例如：pytest tests/unit/test_atr.py -v --timeout=30 -x 2>&1 | head -50

請逐一測試每個檔案，確認通過後標記任務完成。