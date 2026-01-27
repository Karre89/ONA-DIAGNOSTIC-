# ONA Health - User Guide

## Quick Start

### Starting the Platform
```bash
cd C:\Users\kayse\imaging-platform
docker-compose -f infra/docker-compose.yml up -d
```

Open your browser to: **http://localhost:8080**

---

## Navigation Overview

### Header Bar
```
┌─────────────────────────────────────────────────────────────────┐
│  [👁 ONA]  EDGE AGENT          [Online ●]  [EN|SW|SO]  [Admin]  │
└─────────────────────────────────────────────────────────────────┘
     ↑              ↑                  ↑          ↑         ↑
   Logo         Subtitle          Status    Language    Settings
```

- **ONA Logo**: Click to return to dashboard
- **Status Indicator**: Green = connected to cloud, Red = offline
- **Language Selector**: EN (English), SW (Swahili), SO (Somali)
- **Admin**: System settings and sync status

---

## Page 1: Dashboard (Home)

**URL:** `http://localhost:8080/`

### What You See:

```
┌─────────────────────────────────────────────────────────────────┐
│                      TODAY'S SUMMARY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│   │   12    │  │    3    │  │    4    │  │    5    │           │
│   │ TOTAL   │  │  HIGH   │  │ MEDIUM  │  │   LOW   │           │
│   └─────────┘  └─────────┘  └─────────┘  └─────────┘           │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  🔴 HIGH RISK CASES - IMMEDIATE ATTENTION                       │
├─────────────────────────────────────────────────────────────────┤
│  │ STU-20260127-A1B2C3 │ HIGH RISK │ 78% │ 12:45 │ [View →]   │
│  │ STU-20260127-D4E5F6 │ HIGH RISK │ 65% │ 11:30 │ [View →]   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  OTHER RESULTS                                                   │
├─────────────────────────────────────────────────────────────────┤
│  │ STU-20260127-G7H8I9 │ MEDIUM    │ 45% │ 10:15 │ [View →]   │
│  │ STU-20260127-J0K1L2 │ LOW       │ 12% │ 09:00 │ [View →]   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### How to Use:

1. **Check HIGH RISK first** - These patients need immediate attention
2. **Click any row** to see full details
3. **Stats update in real-time** as new X-rays are processed

---

## Page 2: Result Detail

**URL:** `http://localhost:8080/result/{result_id}`

### What You See:

```
┌─────────────────────────────────────────────────────────────────┐
│  ← Back to Dashboard                                             │
│                                                                  │
│  STU-20260127-A1B2C3                          [HIGH RISK]        │
│                                                                  │
├────────────────────────────┬────────────────────────────────────┤
│                            │                                     │
│    ┌──────────────────┐   │    ┌──────────────────┐            │
│    │                  │   │    │                  │            │
│    │   CHEST X-RAY    │   │    │    AI HEATMAP    │            │
│    │                  │   │    │   (red = focus)  │            │
│    │                  │   │    │                  │            │
│    └──────────────────┘   │    └──────────────────┘            │
│                            │                                     │
├────────────────────────────┴────────────────────────────────────┤
│                      AI ASSESSMENT                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                    │
│   │   78%   │    │   92%   │    │  850ms  │                    │
│   │TB Score │    │ Quality │    │ Process │                    │
│   └─────────┘    └─────────┘    └─────────┘                    │
│                                                                  │
│   Findings: Upper lobe opacity with possible cavitation pattern │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                   REQUIRED ACTIONS (HIGH RISK)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   [✓] Collect sputum sample immediately                         │
│   [ ] Send for GeneXpert testing                                 │
│   [ ] Refer patient if needed                                    │
│                                                                  │
│   [Save Actions]                                                 │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                  YOUR CLINICAL ASSESSMENT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐                    │
│   │  👍     │    │  👎     │    │   ❓    │                    │
│   │ Agree   │    │Disagree │    │ Unsure  │                    │
│   └─────────┘    └─────────┘    └─────────┘                    │
│                                                                  │
│   Notes: ____________________________________________            │
│                                                                  │
│   [Submit Assessment]                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### How to Use:

1. **Review the X-ray** (left) and **AI heatmap** (right)
   - Red/orange areas = where AI detected abnormalities

2. **Check AI scores**:
   - **TB Score**: Probability of tuberculosis (0-100%)
   - **Quality**: Image quality score
   - **Process**: How long AI took to analyze

3. **For HIGH RISK cases** - Complete the required actions:
   - ✓ Collect sputum sample
   - ✓ Order GeneXpert test
   - ✓ Refer to TB center if needed

4. **Provide your assessment**:
   - **Agree**: You confirm AI's finding
   - **Disagree**: You think AI is wrong
   - **Unsure**: Need more information

5. **Add notes** explaining your clinical reasoning

---

## Page 3: All Studies

**URL:** `http://localhost:8080/studies`

### What You See:

```
┌─────────────────────────────────────────────────────────────────┐
│                        ALL STUDIES                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  │ Study ID           │ Status │ Risk   │ Date       │ Action │ │
│  ├────────────────────┼────────┼────────┼────────────┼────────┤ │
│  │ STU-20260127-A1B2  │ READY  │ HIGH   │ 2026-01-27 │ [View] │ │
│  │ STU-20260127-C3D4  │ READY  │ MEDIUM │ 2026-01-27 │ [View] │ │
│  │ STU-20260127-E5F6  │ READY  │ LOW    │ 2026-01-27 │ [View] │ │
│  │ STU-20260126-G7H8  │ READY  │ LOW    │ 2026-01-26 │ [View] │ │
│                                                                  │
│  [← Previous]  Page 1 of 5  [Next →]                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### How to Use:

1. **Browse all historical studies**
2. **Click View** to see any study's details
3. **Use pagination** to see older results

---

## Page 4: Admin Panel

**URL:** `http://localhost:8080/admin`

### What You See:

```
┌─────────────────────────────────────────────────────────────────┐
│                       ADMIN PANEL                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DEVICE STATUS                                                   │
│  ├─ Device ID: edge-box-001                                     │
│  ├─ Tenant: Ona Kenya                                           │
│  ├─ Site: Kenyatta Hospital                                     │
│  └─ Registered: 2026-01-15 10:30:00                             │
│                                                                  │
│  SYNC STATUS                                                     │
│  ├─ Pending: 3 items                                            │
│  ├─ Failed: 0 items                                             │
│  └─ Last sync: 2 minutes ago                                    │
│                                                                  │
│  MODEL INFO                                                      │
│  ├─ Version: ona-cxr-tb-v1.0                                    │
│  └─ Updated: 2026-01-10                                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### How to Use:

1. **Check device registration** - Ensure connected to correct tenant/site
2. **Monitor sync queue** - Items waiting to upload to cloud
3. **Verify model version** - Check you have latest AI model

---

## Risk Levels Explained

| Level | Color | TB Score | What It Means | Action Required |
|-------|-------|----------|---------------|-----------------|
| **HIGH** | 🔴 Red | ≥60% | Strong TB suspicion | Collect sputum, GeneXpert, possible referral |
| **MEDIUM** | 🟡 Yellow | 30-59% | Moderate concern | Clinical correlation, consider follow-up |
| **LOW** | 🟢 Green | <30% | Likely normal | Routine follow-up if symptomatic |
| **REVIEW** | ⚪ Gray | Any | Poor image quality | Repeat X-ray needed |

---

## Workflow: Processing a New Patient

```
Step 1: Patient gets chest X-ray
            ↓
Step 2: X-ray machine sends image to ONA (DICOM)
            ↓
Step 3: AI analyzes image (< 2 seconds)
            ↓
Step 4: Result appears on dashboard
            ↓
Step 5: Clinician reviews result
            ↓
Step 6: For HIGH RISK:
        - Collect sputum
        - Order GeneXpert
        - Consider referral
            ↓
Step 7: Clinician provides feedback (Agree/Disagree)
            ↓
Step 8: Data syncs to cloud (when online)
```

---

## Offline Mode

ONA works **without internet**:

- ✅ X-rays still get analyzed
- ✅ Results still appear on dashboard
- ✅ Clinicians can still provide feedback
- ⏳ Sync queue grows until connection restored
- 🔄 Auto-syncs when internet returns

**Status indicator shows:**
- 🟢 **Online** - Connected to cloud
- 🔴 **Offline** - Working locally

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `H` | Go to Home/Dashboard |
| `A` | Go to Admin |
| `←` | Previous page |
| `→` | Next page |

---

## Troubleshooting

### "No results showing"
→ Run: `curl -X POST http://localhost:8080/api/ingest-sample`

### "Images not loading"
→ Check containers: `docker-compose ps`

### "Sync stuck"
→ Check cloud connection in Admin panel

### "Wrong language"
→ Click language selector (EN/SW/SO) in header

---

## API Endpoints (For Developers)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard |
| `/result/{id}` | GET | Result detail |
| `/studies` | GET | All studies list |
| `/admin` | GET | Admin panel |
| `/api/ingest-sample` | POST | Create test study |
| `/health` | GET | Health check |
| `/images/study/{id}` | GET | X-ray image |
| `/images/heatmap/{id}` | GET | Heatmap image |

---

## Support

- GitHub: https://github.com/Karre89/ONA-DIAGNOSTIC-
- Issues: https://github.com/Karre89/ONA-DIAGNOSTIC-/issues
