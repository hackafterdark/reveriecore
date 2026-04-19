### 🧠 Deep Dive: Importance Score Mechanics and Behavior

Now that the field is in the schema, the brainstorming must shift to the **logic** behind it. This is where the real intelligence lies.

#### 1. Calculation Mechanism (BART Zero-Shot Entailment)
The importance score is no longer based on keywords or simple sentiment. Instead, ReverieCore uses the **BART-Large-MNLI** model to evaluate the "weight" of a memory through semantic entailment.

**A. Weighted Label Entailment:**
The model evaluates the memory against four semantic labels, each assigned a numerical weight:
*   **"critical"**: Weight 5.0
*   **"important"**: Weight 3.0
*   **"minor"**: Weight 1.0 (Baseline)
*   **"trivial"**: Weight -2.0 (Penalty)

**B. Scoring Calculation:**
The final score is a **Softmax-Weighted Sum** of these labels. The plugin takes the raw logits (confidence scores) for each label and calculates the final importance:

$$\text{Importance} = \sum_{i=1}^{n} (\text{Softmax Confidence}_i \times \text{Label Weight}_i)$$

*   Result is clipped to a **[1.0, 5.0]** range.
*   A memory that strongly entails "critical" will move toward 5.0.
*   A memory that strongly entails "trivial" will stay near 1.0.

#### 2. Intelligence Benefits
*   **Contextual Understanding**: BART understands that "I'm having a hard time with this" is more important than "This is a blue chair" without needing to see the word "critical".
*   **Adaptive Ranking**: The score directly scales the relevance of the memory during retrieval, ensuring high-importance facts surface over similar but trivial noise.

#### 3. Retrieval Integration (How the Score is Used)

When the agent queries:
1.  **Vector Search:** Query is embedded, and `sqlite-vec` finds the Top K closest neighbors based purely on **Semantic Similarity**.
2.  **Score Filtering/Re-ranking:** The results from Step 1 are then passed to a final sorting layer. The final retrieved order is determined by a combination of:
    $$\text{Final Rank} = (W_1 \times \text{Similarity Score}) + (W_2 \times \text{Importance Score})$$
    (Where $W_1$ and $W_2$ are configurable weights, allowing us to tune whether semantic match or importance matters more.)