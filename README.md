# Credit Risk Model Project

## Credit Scoring Business Understanding

**1. Influence of Basel II Accord on Model Interpretability and Documentation**  
The Basel II Accord emphasizes accurate risk measurement and regulatory compliance, necessitating interpretable models like Logistic Regression, where feature contributions to risk scores are transparent. Well-documented models ensure traceability and auditability, allowing Bati Bank to justify its credit risk assessments to regulators, aligning with Basel II’s requirements for robust risk management and transparency.

**2. Necessity and Risks of a Proxy Variable**  
Without a direct “default” label, a proxy variable, derived from RFM metrics, is essential to categorize customers as high or low risk for model training. However, this proxy may inaccurately represent true default behavior, risking misclassification of customers, which could lead to lost business opportunities (false positives) or increased financial losses from defaults (false negatives).

**3. Trade-offs Between Simple and Complex Models**  
In a regulated financial context, Logistic Regression with WoE offers interpretability, enabling clear explanations of risk scores to regulators, but may lack the accuracy of complex models like Gradient Boosting. Gradient Boosting captures complex patterns for higher predictive power but is less interpretable, posing challenges for regulatory compliance. The trade-off involves balancing interpretability for regulatory adherence with accuracy for better risk prediction.