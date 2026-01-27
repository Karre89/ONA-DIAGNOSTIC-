# ONA HEALTH — THE AI LEARNING LOOP
## How Your Feedback Makes the AI Smarter

---

## The Big Picture

Every time you use Ona, you're not just helping patients — you're **teaching the AI**.

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
│     │  Model   │  ← Automatically downloaded to your Edge Box          │
│     │  Update  │                                                       │
│     └──────────┘                                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## How Your Clicks Become Training Data

### Step 1: AI Makes a Prediction

```
Patient X-ray arrives
        ↓
AI analyzes: "92% TB probability"
        ↓
AI says: 🔴 HIGH RISK
```

### Step 2: You Review and Respond

```
You examine the X-ray and patient
        ↓
You click one of three buttons:

   👍 AGREE        👎 DISAGREE      ❓ UNSURE
   "AI is right"   "AI is wrong"    "Can't tell"
```

### Step 3: Your Feedback Becomes a Label

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  What gets recorded:                                        │
│                                                             │
│  • Study ID: STU-20260127-B99A2B                           │
│  • AI prediction: HIGH (92%)                                │
│  • Your response: AGREE ✓                                   │
│  • Your notes: "Classic upper lobe cavitation"             │
│  • Timestamp: 2026-01-27 11:17:33                          │
│                                                             │
│  This is now LABELED TRAINING DATA                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Step 4: Ground Truth Confirms (When Available)

```
Later, GeneXpert result comes back:
        ↓
    ✅ POSITIVE for TB
        ↓
This CONFIRMS the AI was right!
        ↓
Even stronger training signal
```

---

## The Four Types of Feedback

| Your Click | GeneXpert Result | What It Means | Training Value |
|------------|------------------|---------------|----------------|
| 👍 Agree (HIGH) | Positive | AI correct, you correct | ⭐⭐⭐ High |
| 👎 Disagree (HIGH) | Negative | AI wrong (false positive) | ⭐⭐⭐ High |
| 👎 Disagree (LOW) | Positive | AI missed it (false negative) | ⭐⭐⭐⭐ Critical! |
| 👍 Agree (LOW) | Negative | AI correct, you correct | ⭐⭐ Normal |

**The most valuable feedback:** When you **disagree** with the AI, especially if you catch something the AI missed.

---

## The Complete Learning Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  AT THE HOSPITAL (Edge)                                                │
│  ─────────────────────                                                 │
│                                                                         │
│  1. X-ray arrives                                                      │
│  2. AI predicts: HIGH/MEDIUM/LOW                                       │
│  3. Clinician reviews                                                  │
│  4. Clinician clicks: Agree/Disagree/Unsure                           │
│  5. Clinician records actions (sputum, GeneXpert, refer)              │
│  6. Later: GeneXpert result recorded                                   │
│                                                                         │
│         │                                                               │
│         │ Sync (when internet available)                               │
│         ▼                                                               │
│                                                                         │
│  IN THE CLOUD                                                          │
│  ────────────                                                          │
│                                                                         │
│  7. Feedback aggregated from all sites                                 │
│  8. Weekly analysis: Where is AI making mistakes?                      │
│  9. Disagreement cases reviewed by experts                             │
│  10. New training dataset created                                      │
│                                                                         │
│         │                                                               │
│         ▼                                                               │
│                                                                         │
│  ML TEAM (Quarterly)                                                   │
│  ───────────────────                                                   │
│                                                                         │
│  11. Retrain model with new data                                       │
│  12. Validate on held-out test set                                     │
│  13. If better → promote to production                                 │
│                                                                         │
│         │                                                               │
│         │ Staged rollout                                                │
│         ▼                                                               │
│                                                                         │
│  BACK TO EDGE                                                          │
│  ────────────                                                          │
│                                                                         │
│  14. New model version downloaded automatically                        │
│  15. AI is now smarter!                                                │
│  16. Cycle repeats...                                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components Explained

### 1. Clinician Feedback (You!)

| Component | What It Does |
|-----------|--------------|
| **Agree Button** | Confirms AI was right → Reinforces learning |
| **Disagree Button** | Tells AI it was wrong → Corrects mistakes |
| **Unsure Button** | Flags difficult cases → Identifies edge cases |
| **Notes Field** | Explains WHY → Helps understand failure modes |

**Your clicks are the most valuable data we collect.**

---

### 2. Ground Truth (GeneXpert Results)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  GeneXpert = The Gold Standard                              │
│                                                             │
│  When we can link:                                          │
│                                                             │
│    X-ray + AI prediction + GeneXpert result                │
│                                                             │
│  We get CONFIRMED training data:                            │
│                                                             │
│    • AI said HIGH + GeneXpert POSITIVE = True Positive     │
│    • AI said HIGH + GeneXpert NEGATIVE = False Positive    │
│    • AI said LOW + GeneXpert POSITIVE = False Negative ❌  │
│    • AI said LOW + GeneXpert NEGATIVE = True Negative      │
│                                                             │
│  This is the BEST data for improving the model.            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 3. Active Learning (AI Asks for Help)

When the AI is uncertain, it says **NOT CONFIDENT**:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  AI internally: "I'm only 55% sure... that's not enough"   │
│                                                             │
│         ↓                                                   │
│                                                             │
│  Shows: ⚪ NOT CONFIDENT                                    │
│                                                             │
│         ↓                                                   │
│                                                             │
│  You review carefully and provide feedback                  │
│                                                             │
│         ↓                                                   │
│                                                             │
│  These "hard cases" are EXTRA valuable for training!       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why this matters:** The AI learns most from cases it finds difficult. Your expert review on these cases has outsized impact.

---

### 4. Model Registry (Version Control)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Model Registry (in the Cloud)                              │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Version       │ Status      │ Performance           │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ ona-cxr-tb-v1.0 │ Production │ Sens: 90%, Spec: 72% │   │
│  │ ona-cxr-tb-v1.1 │ Canary 5%  │ Sens: 92%, Spec: 74% │   │
│  │ ona-cxr-tb-v1.2 │ Testing    │ Sens: 93%, Spec: 75% │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Each version is tested before wide release                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 5. OTA Updates (Over-The-Air)

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  How Your Edge Box Gets Smarter                             │
│                                                             │
│  1. Cloud has new model version ready                       │
│         ↓                                                   │
│  2. Edge Box checks for updates (during heartbeat)          │
│         ↓                                                   │
│  3. Cloud says: "New model available: v1.1"                │
│         ↓                                                   │
│  4. Edge Box downloads in background                        │
│         ↓                                                   │
│  5. Model swapped during low-activity period                │
│         ↓                                                   │
│  6. You now have the smarter AI!                           │
│                                                             │
│  ⚠️ If new model has problems → Automatic rollback!         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**You don't need to do anything.** Updates happen automatically.

---

## Staged Rollout (Safety First)

We don't push new models to everyone at once:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  STAGED ROLLOUT PROCESS                                     │
│                                                             │
│  Stage 1: LAB (0%)                                         │
│  └── Internal testing only                                  │
│                                                             │
│  Stage 2: SHADOW (0% visible)                              │
│  └── Runs alongside production, results compared           │
│  └── Users don't see new model yet                         │
│                                                             │
│  Stage 3: CANARY (5%)                                      │
│  └── 5% of sites get new model                             │
│  └── Watch for problems                                     │
│                                                             │
│  Stage 4: PRODUCTION (100%)                                │
│  └── Everyone gets the update                              │
│                                                             │
│  ⚠️ Auto-rollback if:                                       │
│     • Error rate spikes                                     │
│     • Too many NOT_CONFIDENT                                │
│     • Latency increases                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Why Your Feedback Matters

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
│  ═══════════════════════════════════════════════════════   │
│                                                             │
│  Every click of "Agree" or "Disagree" makes a difference.  │
│                                                             │
│  You're not just using AI — you're TRAINING it.            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## The Data Flywheel

```
                    ┌─────────────────┐
                    │   More Sites    │
                    │   Using Ona     │
                    └────────┬────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │                                       │
         ▼                                       │
┌─────────────────┐                    ┌─────────────────┐
│  More Feedback  │                    │  Better Model   │
│  (Agree/Disagree)│                   │  (Higher Accuracy)│
└────────┬────────┘                    └────────┬────────┘
         │                                       │
         │                                       │
         ▼                                       │
┌─────────────────┐                              │
│  More Training  │ ─────────────────────────────┘
│     Data        │
└─────────────────┘

This is a FLYWHEEL — the more you use it, the better it gets!
```

---

## Summary

| Component | What It Does | Who Does It |
|-----------|--------------|-------------|
| **Clinician Feedback** | Agree/Disagree creates training labels | You (clinicians) |
| **Ground Truth** | GeneXpert results confirm actual TB cases | Lab |
| **Active Learning** | Model asks for review on uncertain cases | AI + You |
| **Model Registry** | Cloud stores versioned models | Ona Team |
| **OTA Updates** | Edge devices download new models automatically | Automatic |
| **Staged Rollout** | New models tested before wide release | Ona Team |
| **Auto-Rollback** | Bad models are reverted automatically | Automatic |

---

## Remember

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   Your feedback is not just helpful — it's ESSENTIAL.        ║
║                                                               ║
║   Every time you click Agree or Disagree, you're:            ║
║   • Teaching the AI what TB looks like in YOUR population    ║
║   • Helping it learn from its mistakes                       ║
║   • Making it better for the NEXT patient                    ║
║                                                               ║
║   The AI can only be as good as the feedback it receives.    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

---

*Document Version: 1.0*
*Part of Ona Health Documentation Suite*
*onahealth.africa*
