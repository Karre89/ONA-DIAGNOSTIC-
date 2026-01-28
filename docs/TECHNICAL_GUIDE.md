# ONA HEALTH — TECHNICAL GUIDE
## Understanding the AI: Training, Deployment, and the Learning Loop

**For:** Founders, Technical Staff, ML Engineers
**Version:** 1.0
**Last Updated:** January 2026

---

# TABLE OF CONTENTS

1. [The Big Picture](#1-the-big-picture)
2. [PyTorch vs ONNX Explained](#2-pytorch-vs-onnx-explained)
3. [The Training Pipeline](#3-the-training-pipeline)
4. [The Deployment Pipeline](#4-the-deployment-pipeline)
5. [The Feedback Loop](#5-the-feedback-loop)
6. [Model Versioning & Updates](#6-model-versioning--updates)
7. [Retraining Schedule](#7-retraining-schedule)
8. [Getting Started: Your First Model](#8-getting-started-your-first-model)
9. [Quick Reference](#9-quick-reference)
10. [Glossary](#10-glossary)
11. [Data Privacy & Compliance](#11-data-privacy--compliance)

---

# 1. THE BIG PICTURE

## What We're Building

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                         ONA HEALTH AI SYSTEM                            │
│                                                                         │
│  ┌─────────────────┐          ┌─────────────────┐                      │
│  │                 │          │                 │                      │
│  │  TRAINING       │          │  DEPLOYMENT     │                      │
│  │  (You + Colab)  │ ──────►  │  (Hospitals)    │                      │
│  │                 │  ONNX    │                 │                      │
│  │  PyTorch        │  file    │  ONNX Runtime   │                      │
│  │  2GB            │  50MB    │  100MB          │                      │
│  │                 │          │                 │                      │
│  └─────────────────┘          └────────┬────────┘                      │
│                                        │                                │
│                                        │ Feedback                       │
│                                        ▼                                │
│                               ┌─────────────────┐                      │
│                               │                 │                      │
│                               │  CLOUD          │                      │
│                               │  (Collects      │                      │
│                               │   feedback)     │                      │
│                               │                 │                      │
│                               └────────┬────────┘                      │
│                                        │                                │
│                                        │ Data for                       │
│                                        │ retraining                     │
│                                        ▼                                │
│                               ┌─────────────────┐                      │
│                               │  RETRAIN        │                      │
│                               │  (Every 3-6 mo) │ ─────► New ONNX      │
│                               └─────────────────┘                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## The Core Concept

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  PyTorch  =  Teaching a chef how to cook                    │
│              (takes time, needs big school)                 │
│                                                             │
│  ONNX     =  The recipe book the chef wrote                 │
│              (small, anyone can follow it)                  │
│                                                             │
│  TRAINING = Chef goes to school (PyTorch)                   │
│  EXPORT   = Chef writes down recipes (ONNX)                 │
│  DEPLOY   = Kitchen follows recipes (ONNX Runtime)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# 2. PYTORCH VS ONNX EXPLAINED

## What Each One Does

| | PyTorch | ONNX |
|---|---------|------|
| **Purpose** | TRAIN (teach) the model | RUN (use) the model |
| **Size** | 2,000+ MB | 50-100 MB |
| **RAM needed** | 4+ GB | 1 GB |
| **Device cost** | $500+ computer | $100 device works |
| **Startup time** | 30 seconds | 5 seconds |
| **Can train?** | ✅ Yes | ❌ No |
| **Can predict?** | ✅ Yes | ✅ Yes |
| **Where used** | Your laptop / Colab | Hospital edge box |

## Why We Use Both

```
STAGE 1: Training (Your Computer / Google Colab)
────────────────────────────────────────────────

   You need PyTorch here because you're TEACHING the model

   ┌──────────────────┐
   │   PyTorch        │
   │   (2GB+)         │  ← Big, but OK for your laptop/Colab
   │                  │
   │   • Load data    │
   │   • Train model  │
   │   • Test model   │
   │   • Export       │
   └────────┬─────────┘
            │
            │  Export to ONNX
            ▼
   ┌──────────────────┐
   │  model.onnx      │  ← Just the "brain", 50MB file
   │  (50MB)          │
   └──────────────────┘


STAGE 2: Deployment (Edge Device at Hospital)
─────────────────────────────────────────────

   You only need ONNX here because you're USING the model

   ┌──────────────────┐
   │   ONNX Runtime   │
   │   (100MB)        │  ← Small, runs on cheap hardware
   │                  │
   │   • Load model   │
   │   • Run on X-ray │
   │   • Get score    │
   └──────────────────┘
```

## Why This Matters for Africa

```
Hospital in Rural Kenya:
────────────────────────

Option A: Deploy PyTorch
├── Need: $500+ mini PC with 8GB RAM
├── Download: 2GB+ of libraries
├── Startup time: 30+ seconds
└── ❌ Too expensive, too slow

Option B: Deploy ONNX
├── Need: $100 device with 2GB RAM
├── Download: 100MB of libraries
├── Startup time: 5 seconds
└── ✅ Affordable, fast
```

## ONNX: The "Frozen Brain"

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ONNX is like a FROZEN brain.                               │
│                                                             │
│  • It knows what it knew when you exported it               │
│  • It CANNOT get smarter on its own                         │
│  • It gives the same answer today as tomorrow               │
│                                                             │
│  This is GOOD because:                                      │
│  • Predictable behavior                                     │
│  • No uncontrolled changes                                  │
│  • Consistent across all hospitals                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# 3. THE TRAINING PIPELINE

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                        TRAINING PIPELINE                                │
│                                                                         │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐            │
│   │  DATA   │ ─► │ TRAIN   │ ─► │VALIDATE │ ─► │ EXPORT  │            │
│   │         │    │         │    │         │    │         │            │
│   │ X-rays  │    │ PyTorch │    │ Test on │    │ To ONNX │            │
│   │ Labels  │    │ on GPU  │    │ held-out│    │         │            │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘            │
│       │                                             │                   │
│       │                                             ▼                   │
│       │                                       ┌─────────┐              │
│       │         WHERE: Google Colab           │model.onnx│              │
│       │         COST: Free                    │  (50MB) │              │
│       │         TIME: 2-4 hours               └─────────┘              │
│       │                                                                 │
└───────┼─────────────────────────────────────────────────────────────────┘
        │
        │
        ▼
   DATA SOURCES
   ────────────
   • Public datasets (Shenzhen, Montgomery, NIH)
   • Your clinic partner data
   • Clinician feedback (Agree/Disagree)
   • GeneXpert confirmed cases
```

## Data Sources

### Public Datasets (Free, Start Here)

| Dataset | Size | What It Has | Access |
|---------|------|-------------|--------|
| **Shenzhen TB** | 662 images | 336 TB+, 326 normal | [Kaggle](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen) |
| **Montgomery TB** | 138 images | 80 TB+, 58 normal | [Kaggle](https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery) |
| **NIH ChestX-ray14** | 112,000 images | 14 conditions labeled | [NIH](https://nihcc.app.box.com/v/ChestXray-NIHCC) |
| **VinDr-CXR** | 18,000 images | Multiple conditions | [Kaggle](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection) |

### Your Own Data (Collect Over Time)

| Source | What It Provides | Value |
|--------|------------------|-------|
| **Clinic X-rays** | Real African patient images | ⭐⭐⭐ High |
| **Clinician feedback** | Agree/Disagree labels | ⭐⭐⭐ High |
| **GeneXpert results** | Confirmed TB diagnosis | ⭐⭐⭐⭐ Highest |

## Training Steps

```
STEP 1: Prepare Data
────────────────────
• Collect X-ray images
• Get labels (TB positive / negative)
• Split: 80% training, 20% testing
• Make sure test set is from DIFFERENT hospitals (site-split)


STEP 2: Load Pretrained Model
─────────────────────────────
• Use TorchXRayVision (already trained on 800K+ images)
• Don't start from scratch!


STEP 3: Fine-tune on Your Data
──────────────────────────────
• Run on Google Colab (free GPU)
• Takes 2-4 hours
• Model learns African-specific patterns


STEP 4: Validate
────────────────
• Test on held-out data
• Check: Sensitivity ≥ 90%, Specificity ≥ 70%
• If not good enough, collect more data and repeat


STEP 5: Export to ONNX
──────────────────────
• Convert PyTorch model to ONNX format
• Output: model.onnx file (~50MB)
• This is what goes to hospitals
```

## Free GPU Options

| Platform | GPU | Cost | Time Limit |
|----------|-----|------|------------|
| **Google Colab** | T4 (free), A100 (paid) | Free tier works | ~12 hours/session |
| **Kaggle Notebooks** | P100 | Free | 30 hours/week |
| **Lightning.ai** | Various | Free credits | Good for training |

---

# 4. THE DEPLOYMENT PIPELINE

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                       DEPLOYMENT PIPELINE                               │
│                                                                         │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐            │
│   │  ONNX   │ ─► │ UPLOAD  │ ─► │ STAGED  │ ─► │ ALL     │            │
│   │  FILE   │    │ TO CLOUD│    │ ROLLOUT │    │HOSPITALS│            │
│   │         │    │         │    │         │    │         │            │
│   │ 50MB    │    │ Model   │    │ 5% first│    │ 100%    │            │
│   │         │    │ registry│    │ then all│    │         │            │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## What's on the Edge Device

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Edge Agent (at hospital) contains:                         │
│                                                             │
│  ALREADY INSTALLED:                                         │
│  • Python 3.11 ✓                                            │
│  • FastAPI ✓                                                │
│  • SQLite ✓                                                 │
│                                                             │
│  ADD FOR AI:                                                │
│  • onnxruntime (100MB)                                      │
│  • model.onnx (50MB)                                        │
│                                                             │
│  TOTAL ADDED: ~150MB                                        │
│  (Instead of 2GB+ for full PyTorch)                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Staged Rollout Process

```
WHY STAGED ROLLOUT?
───────────────────
Don't send new model to ALL hospitals at once.
What if it has a bug? You'd break everything!


THE STAGES:
───────────

Stage 1: LAB (0% of hospitals)
└── Test internally only
└── Run on sample images
└── Check for obvious errors

Stage 2: SHADOW (0% visible)
└── New model runs alongside old model
└── Compare outputs
└── Users don't see new model yet

Stage 3: CANARY (5% of hospitals)
└── Send to 5% of hospitals
└── Watch for problems for 1 week
└── Check: error rate, disagreement rate

Stage 4: PRODUCTION (100%)
└── Send to all hospitals
└── Monitor for 2 weeks
└── Ready!


AUTO-ROLLBACK:
──────────────
If new model causes problems:
• Error rate spikes → Automatic rollback
• Too many NOT_CONFIDENT → Automatic rollback
• Latency increases → Automatic rollback

Old model restored automatically. No manual work needed.
```

## How Edge Device Gets Updates

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  AUTOMATIC UPDATE PROCESS (OTA = Over The Air)              │
│                                                             │
│  1. Cloud has new model version ready                       │
│          ↓                                                  │
│  2. Edge Box checks for updates (during heartbeat)          │
│          ↓                                                  │
│  3. Cloud says: "New model available: v1.1"                │
│          ↓                                                  │
│  4. Edge Box downloads in background                        │
│          ↓                                                  │
│  5. Model swapped during low-activity period                │
│          ↓                                                  │
│  6. Hospital now has smarter AI!                           │
│                                                             │
│  ⚠️  If problems detected → Automatic rollback to old model │
│                                                             │
│  Hospital staff don't need to do anything.                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# 5. THE FEEDBACK LOOP

## Why Feedback Matters

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  WITHOUT clinician feedback:                                │
│  • AI stays the same forever                                │
│  • Same mistakes repeated                                   │
│  • No improvement                                           │
│                                                             │
│  WITH clinician feedback:                                   │
│  • AI learns from mistakes                                  │
│  • Gets better every quarter                                │
│  • Adapts to local populations                              │
│  • Catches things it used to miss                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## How Feedback Becomes Training Data

```
STEP 1: AI makes a prediction
─────────────────────────────
X-ray comes in
     ↓
AI analyzes: "92% TB probability"
     ↓
AI says: 🔴 HIGH RISK


STEP 2: Clinician reviews
─────────────────────────
Clinician looks at X-ray
     ↓
Clinician decides: Was AI right or wrong?
     ↓
Clicks one button:

   👍 AGREE        👎 DISAGREE      ❓ UNSURE
   "AI is right"   "AI is wrong"    "Can't tell"


STEP 3: Feedback becomes a label
────────────────────────────────
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  What gets recorded:                                        │
│                                                             │
│  • Study ID: STU-20260127-B99A2B                           │
│  • AI prediction: HIGH (92%)                                │
│  • Clinician response: DISAGREE                             │
│  • Clinician notes: "Old scarring, not active TB"          │
│  • Timestamp: 2026-01-27 11:17:33                          │
│                                                             │
│  This is now LABELED TRAINING DATA                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘


STEP 4: Ground truth confirms (when available)
──────────────────────────────────────────────
GeneXpert result: NEGATIVE
     ↓
Confirms: Clinician was right, AI was wrong
     ↓
HIGH VALUE training example for next model
```

## The Four Types of Feedback

| Your Click | GeneXpert Result | What It Means | Training Value |
|------------|------------------|---------------|----------------|
| 👍 Agree (HIGH) | Positive | AI correct, you correct | ⭐⭐⭐ High |
| 👎 Disagree (HIGH) | Negative | AI wrong (false positive) | ⭐⭐⭐ High |
| 👎 Disagree (LOW) | Positive | AI missed it (false negative) | ⭐⭐⭐⭐ Critical! |
| 👍 Agree (LOW) | Negative | AI correct, you correct | ⭐⭐ Normal |

**Most valuable:** When clinician catches something AI missed.

## The Complete Loop

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                    THE ONA LEARNING LOOP                                │
│                                                                         │
│     ┌──────────┐                                                       │
│     │ Patient  │                                                       │
│     │ X-ray    │                                                       │
│     └────┬─────┘                                                       │
│          │                                                              │
│          ▼                                                              │
│     ┌──────────┐      ┌──────────┐      ┌──────────┐                  │
│     │   AI     │ ───► │ Clinician│ ───► │ Feedback │                  │
│     │ Analysis │      │  Review  │      │ Recorded │                  │
│     └──────────┘      └──────────┘      └────┬─────┘                  │
│                                              │                         │
│          ┌───────────────────────────────────┘                         │
│          │                                                              │
│          ▼                                                              │
│     ┌──────────┐      ┌──────────┐      ┌──────────┐                  │
│     │ Training │ ───► │   New    │ ───► │  Better  │                  │
│     │   Data   │      │  Model   │      │    AI    │                  │
│     └──────────┘      └──────────┘      └────┬─────┘                  │
│                                              │                         │
│          ┌───────────────────────────────────┘                         │
│          │                                                              │
│          ▼                                                              │
│     ┌──────────┐                                                       │
│     │  Model   │  ← Automatically downloaded to Edge Box               │
│     │  Update  │                                                       │
│     └──────────┘                                                       │
│                                                                         │
│     RESULT: AI gets smarter over time!                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Why Controlled Learning is Better

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  UNCONTROLLED LEARNING (Bad):                               │
│  ─────────────────────────────                              │
│  • Hospital A's model learns one thing                      │
│  • Hospital B's model learns something different            │
│  • Models become inconsistent                               │
│  • One bad feedback could break the model                   │
│  • No quality control                                       │
│                                                             │
│  CONTROLLED LEARNING (What we do):                          │
│  ─────────────────────────────────                          │
│  • All feedback collected centrally                         │
│  • ML team reviews it                                       │
│  • Model retrained carefully                                │
│  • Tested before deployment                                 │
│  • Same model goes to ALL hospitals                         │
│  • Quality controlled                                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# 6. MODEL VERSIONING & UPDATES

## Version Naming

```
ona-cxr-tb-v1.0
│   │   │   │
│   │   │   └── Version number (major.minor)
│   │   └────── Target condition (TB)
│   └────────── Modality (Chest X-ray)
└────────────── Product name (Ona)


Examples:
• ona-cxr-tb-v1.0  → First TB model
• ona-cxr-tb-v1.1  → Minor improvement
• ona-cxr-tb-v2.0  → Major update (new architecture or big data add)
• ona-ct-bleed-v1.0 → CT hemorrhage model (future)
```

## Model Registry

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  MODEL REGISTRY (Cloud)                                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Version         │ Status     │ Performance          │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ ona-cxr-tb-v1.0 │ Production │ Sens: 90% Spec: 72%  │   │
│  │ ona-cxr-tb-v1.1 │ Canary 5%  │ Sens: 92% Spec: 74%  │   │
│  │ ona-cxr-tb-v1.2 │ Testing    │ Sens: 93% Spec: 75%  │   │
│  │ ona-cxr-tb-v0.9 │ Archived   │ (old version)        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Each version is tested before wide release                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## What Triggers an Update?

| Trigger | Action | Example |
|---------|--------|---------|
| **Scheduled** | Quarterly retrain | Every 3 months |
| **Data milestone** | 1,000+ new labeled images | Enough new data to improve |
| **Performance drop** | Sensitivity drops >5% | Model degrading |
| **New region** | Expand to new country | Somalia data added |
| **Emergency** | Dangerous mistakes | Missing too many TB cases |

---

# 7. RETRAINING SCHEDULE

## How Often?

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  SHORT ANSWER: Every 3-6 months                             │
│                                                             │
│  (Unless something goes wrong — then faster)                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## The Schedule

| Phase | Frequency | Why |
|-------|-----------|-----|
| **Early days** (first 6 months) | Every 1-2 months | Learning fast, lots of new data |
| **Stable operation** | Every 3-6 months | Regular improvements |
| **Emergency** | Immediately | Dangerous mistakes found |

## Retrain Decision Tree

```
SCHEDULED (Normal):
───────────────────
□ Every quarter (3 months)
□ When you have 1,000+ new labeled images
□ When you expand to a new country


EARLY (Something's Wrong):
──────────────────────────
□ Clinicians disagreeing too much (>30% disagree rate)
□ Missed TB cases found (false negatives)
□ Too many false alarms (alert fatigue)
□ New X-ray machine type performs poorly


SKIP (Don't Bother):
────────────────────
□ Only 50 new images — not enough
□ Model is performing well — don't fix what's not broken
□ No new feedback — nothing to learn from
```

## Year 1 Timeline

```
Month 0:  Deploy v1.0 (pretrained TorchXRayVision)
          ↓
Month 2:  Collect 1,000+ images from pilots
          Review feedback
          Retrain → v1.1
          ↓
Month 4:  Collect 2,000 more images
          Add Somalia data
          Retrain → v1.2
          ↓
Month 6:  Major update with 5,000+ images
          Retrain → v2.0
          ↓
Month 9:  Quarterly update
          Retrain → v2.1
          ↓
Month 12: Quarterly update
          Retrain → v2.2


YEAR 2+:
────────
Every 3-6 months: Scheduled retrain
As needed: Emergency fixes
```

## How Long Does Retraining Take?

| Step | Time | Who Does It |
|------|------|-------------|
| 1. Collect & prepare data | 1-2 days | Mostly waiting |
| 2. Run training on Colab | 2-4 hours | GPU does the work |
| 3. Validate on test set | 1 hour | Run tests |
| 4. Export to ONNX | 5 minutes | One command |
| 5. Test on edge device | 1 hour | Quick check |
| 6. Staged rollout | 1-2 weeks | Gradual deployment |

**TOTAL:** ~2 weeks from start to all hospitals updated
**YOUR ACTIVE TIME:** ~1 day of actual work

## The Simple Rule

```
╔═════════════════════════════════════════════════════════════╗
║                                                             ║
║  RETRAIN WHEN:                                              ║
║                                                             ║
║  ✓ You have 1,000+ new labeled images                       ║
║           OR                                                ║
║  ✓ 3 months have passed                                     ║
║           OR                                                ║
║  ✓ Clinicians are complaining about accuracy                ║
║                                                             ║
║  Whichever comes FIRST.                                     ║
║                                                             ║
╚═════════════════════════════════════════════════════════════╝
```

---

# 8. GETTING STARTED: YOUR FIRST MODEL

## Week 1: Get Something Working

```
DAY 1-2: Setup
──────────────
□ Create Google Colab account (free)
□ Download TorchXRayVision
□ Run on sample images
□ Confirm it outputs TB scores


DAY 3-4: Validate
─────────────────
□ Download Shenzhen TB dataset (662 images, free)
□ Run model on all 662 images
□ Calculate sensitivity/specificity
□ Document baseline performance


DAY 5: Deploy
─────────────
□ Convert model to ONNX
□ Integrate into Ona Edge (replace stub)
□ Test end-to-end
□ Celebrate! 🎉


RESULT: Real AI in production by end of week!
```

## The Recommended Model

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  TorchXRayVision                                            │
│  https://github.com/mlmed/torchxrayvision                  │
│                                                             │
│  Why this model:                                            │
│  ✓ Pretrained on 828,000+ chest X-rays                     │
│  ✓ Already detects TB, pneumonia, 18 conditions            │
│  ✓ Open source (Apache 2.0 license)                        │
│  ✓ Well documented                                          │
│  ✓ Easy to fine-tune                                        │
│  ✓ Converts to ONNX (runs on CPU)                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start Code

```python
# Real TB detection with TorchXRayVision

import torchxrayvision as xrv
import torch

# Load pretrained model (trained on 828K+ X-rays)
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()

# Load and preprocess X-ray
img = xrv.utils.read("patient_xray.png")       # Read image
img = xrv.datasets.normalize(img, 255)         # Normalize to [0,1]
img = torch.from_numpy(img).unsqueeze(0)       # Add batch dimension

# Get predictions
with torch.no_grad():
    outputs = model(img)

# Get TB-related scores (Lung Opacity is a key indicator)
# Full pathology list: model.pathologies
lung_opacity_idx = model.pathologies.index("Lung Opacity")
tb_score = outputs[0, lung_opacity_idx].item()
print(f"TB probability: {tb_score:.2%}")

# Output: TB probability: 87.3%
```

## Export to ONNX

```python
# Convert trained model to ONNX for edge deployment

import torch

# Create dummy input (1 grayscale image, 224x224)
dummy_input = torch.randn(1, 1, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "ona-cxr-tb-v1.0.onnx",
    input_names=['image'],
    output_names=['scores'],
    dynamic_axes={'image': {0: 'batch'}}  # Allow variable batch size
)

print("Model exported to ona-cxr-tb-v1.0.onnx")
# File size: ~50MB (vs 2GB+ for PyTorch)
```

---

# 9. QUICK REFERENCE

## PyTorch vs ONNX

| | PyTorch | ONNX |
|---|---------|------|
| **Use for** | Training | Deployment |
| **Size** | 2GB+ | 100MB |
| **Where** | Colab/laptop | Hospital |
| **Can learn?** | Yes | No (frozen) |

## Retraining Triggers

| Trigger | Action |
|---------|--------|
| 3 months passed | Scheduled retrain |
| 1,000+ new images | Data-driven retrain |
| >30% disagree rate | Emergency retrain |
| New country added | Expansion retrain |

## Key Metrics

| Metric | Target | Meaning |
|--------|--------|---------|
| Sensitivity | ≥90% | Catches most TB cases |
| Specificity | ≥70% | Not too many false alarms |
| NOT_CONFIDENT | 2-8% | Knows when it's unsure |

## Model Stages

| Stage | % of Hospitals | Duration |
|-------|----------------|----------|
| LAB | 0% | Internal testing |
| SHADOW | 0% visible | 1 week parallel run |
| CANARY | 5% | 1 week monitoring |
| PRODUCTION | 100% | Full deployment |

---

# 10. GLOSSARY

| Term | Simple Definition |
|------|-------------------|
| **PyTorch** | Big toolbox for teaching AI (2GB) |
| **ONNX** | Small file that runs AI predictions (50MB) |
| **ONNX Runtime** | Software that runs ONNX files (100MB) |
| **Training** | Teaching the AI what TB looks like |
| **Inference** | AI making predictions on new X-rays |
| **Fine-tuning** | Improving a pretrained model with new data |
| **Pretrained model** | AI already trained on lots of images |
| **Sensitivity** | % of TB cases the AI catches |
| **Specificity** | % of normal cases correctly identified |
| **False positive** | AI says TB, but patient is healthy |
| **False negative** | AI says healthy, but patient has TB |
| **Staged rollout** | Sending updates to some hospitals first |
| **Rollback** | Reverting to previous model if problems |
| **OTA update** | Over-the-air update (automatic download) |
| **Ground truth** | Confirmed diagnosis (e.g., GeneXpert result) |
| **Feedback loop** | Clinician input that improves future models |
| **Model registry** | Database of all model versions |
| **Edge device** | Computer at the hospital running AI |
| **Colab** | Free Google service with GPUs for training |

---

# 11. DATA PRIVACY & COMPLIANCE

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  DATA PRIVACY REQUIREMENTS                                  │
│                                                             │
│  BEFORE TRAINING:                                           │
│  • All X-rays must be de-identified                        │
│  • Remove: Patient name, ID, dates, hospital info          │
│  • Our edge agent does this automatically                   │
│                                                             │
│  DURING TRAINING:                                           │
│  • Training happens on aggregated anonymized data           │
│  • Individual hospitals NEVER share raw images              │
│  • Only de-identified data syncs to cloud                   │
│                                                             │
│  COMPLIANCE:                                                │
│  • Kenya: Data Protection Act 2019                         │
│  • USA: HIPAA (if applicable)                              │
│  • EU: GDPR (if applicable)                                │
│  • WHO: Guidelines on AI for health                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Handling Class Imbalance

```
PROBLEM: Real-world data is imbalanced
─────────────────────────────────────
In real clinics: ~5-10% TB positive, 90-95% negative
If you train on this directly, model learns to always say "negative"

SOLUTIONS:
──────────
1. OVERSAMPLE: Duplicate TB-positive cases in training
2. UNDERSAMPLE: Use fewer negative cases
3. WEIGHTED LOSS: Penalize TB misses more heavily
4. BALANCED BATCHES: Each batch has 50% TB, 50% normal

RECOMMENDED: Weighted loss function
└── Gives 3-5x penalty for missing TB case
└── This is already common in medical AI
```

---

# SUMMARY

```
╔═════════════════════════════════════════════════════════════╗
║                                                             ║
║  THE ONA AI SYSTEM IN ONE PAGE                              ║
║                                                             ║
║  TRAIN (You, quarterly):                                    ║
║  • Use PyTorch on Google Colab (free)                       ║
║  • Fine-tune on your data                                   ║
║  • Export to ONNX file                                      ║
║                                                             ║
║  DEPLOY (Automatic):                                        ║
║  • ONNX file goes to hospitals                              ║
║  • Runs on cheap hardware                                   ║
║  • Works offline                                            ║
║                                                             ║
║  LEARN (Continuous):                                        ║
║  • Clinicians give feedback                                 ║
║  • Feedback syncs to cloud                                  ║
║  • Used for next training cycle                             ║
║                                                             ║
║  UPDATE (Every 3-6 months):                                 ║
║  • New model trained                                        ║
║  • Staged rollout (5% → 100%)                              ║
║  • Auto-rollback if problems                                ║
║                                                             ║
║  RESULT: AI that gets smarter over time!                    ║
║                                                             ║
╚═════════════════════════════════════════════════════════════╝
```

---

*Document Version: 1.0*
*Part of Ona Health Documentation Suite*
*onahealth.africa*
