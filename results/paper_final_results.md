# Final Experimental Results

## Predictive Performance

The best-performing model was **Random Forest**, achieving a mean ROC-AUC of **0.7311** (SD = 0.0026).

## Risk-Level Effectiveness

Observed late-delivery rates increased monotonically across risk levels:

- LOW: 0.3855
- MEDIUM: 0.6789
- HIGH: 0.9320

The absolute HIGH–LOW separation was **0.5465**, corresponding to a relative increase of **141.76%**.

## Probability Calibration

- Brier Score: **0.194068**
- Expected Calibration Error: **0.004642**

## Economic Impact

- Baseline expected cost: **$9,897,690.45**
- Decision-layer cost: **$9,561,435.12**
- Cost saving: **$336,255.34**
- Saving rate: **3.40%**

## Economic Robustness

The decision layer was economically beneficial in **30 of 49** evaluated cost scenarios.

## Break-Even Analysis

A total of **7** break-even scenarios were evaluated.

## SAP Workflow Evaluation

All **3** evaluated workflow cases reached final approval: **True**.

## Human-in-the-Loop

The evaluation included **36104** observations, of which **12672** involved human intervention or approval, corresponding to **35.10%**.
