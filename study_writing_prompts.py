finding_rewrite_prompt = """
Please generalize and rewrite the <f_NUM> paragraphs findings text of medical images to an structured, accurate, concise and professional FINDINGS section of radiology imaging report. 
The rewrited report needs to contain 7 findings sections, which are named as 'Pulmonary arteries', 'Lungs and Airways', 'Pleura', 'Heart', 'Mediastinum and Hila', 'Chest Wall and Lower Neck', 'Chest Bones'.
Appropriately modify the expression to succinctly summarize the descriptions of similar diseases, the same organs, and the same tissues. Delete the same meaning or the same disease and repeated descriptions. Reduce the contradictory descriptions between abnormalities. Eliminate ambiguity caused by grammatical errors and repetitions.
For each "finding part", only keep the abnormal description belonging to this area, and delete the finding text belonging to other areas. If there is no abnormality, output "normal" or "no acute abnormality" for this part.

Using this output template:
"FINDINGS:
Pulmonary arteries: Findings text of Pulmonary arteries.
Lungs and Airways: Findings text of Lungs and Airways.
...
Chest Bones: Findings text of Chest Bones.
"

For example:
"
FINDINGS:
Pulmonary arteries:  There is a pulmonary embolus which is completely occluding a left lower lobe segmental pulmonary artery (series 2 image 54, 55). There is lack of enhancement of the adjacent lung tissue supplied by the occluded artery, suspicious for a lung infarct.  There is no evidence for right heart strain. 
Lungs and Airways: There is bibasilar airspace disease, which may represent atelectasis,  pneumonia, or aspiration. The central airways are patent. There is no large pulmonary nodule or suspicious mass.
Pleura: There is a trace left-sided pleural effusion. 
Heart: Normal.
Mediastinum and Hila: Normal.
Chest Wall and Lower Neck:  1.7 cm right thyroid cystic nodule.
Chest Bones: No acute abnormality. 
"

Remove expressions like "in the [SEG] area". Do NOT output prompt text (such as 'Here is the rewritten FINDINGS section:'). Only output findings content. Output plain text.
"""

impression_write_prompt = """
Based on the provided FINDINGS text, write an impression for a medical image report. Follow these guidelines:
1. Summarize pulmonary embolism Findings:
   - Always start by summarizing the pulmonary embolism conclusion.
   - For the normal pulmonary arteries cases, the pulmonary embolism conclusion always outputs "No pulmonary embolism are identified.".
2. Summarize Key Abnormal Findings:
   - Emphasize first the most significant and relevant findings to pulmonary embolism. 
   - Address the acute and severe abnormalities noted in the findings, reduce the normal findings
3. Be Clear and Concise:
   - Use clear, direct language.
   - Avoid unnecessary jargon, but ensure the terminology used is precise.
4. Prioritize Clinical Relevance:
   - Focus on findings with significant clinical impact.
   - Note any findings requiring urgent attention or follow-up.
5. Consider Clinical Correlation:
   - Relate imaging findings to the patient's clinical presentation when possible.
   - Recommend correlation with clinical history, physical examination, or other studies.
6. Use Structured Format:
   - Organize the impression in a logical manner.
   - Use numbered points for clarity and readability.

Here is an FINDINGS section example:
"
FINDINGS:
Pulmonary arteries:  There is subsegmental pulmonary embolism in both lower lobes.
Lungs and Airways:  Emphysema. Mild pulmonary edema. Bibasilar atelectasis. Artifact limits detection for small nodules. 18 mm focal groundglass nodule in the left lower lobe on image 81.
Pleura: Small bilateral pleural effusions. No pneumothorax.
Heart and mediastinum: Mild cardiomegaly. Enlarged main pulmonary artery to 3.4 cm.
Chest Wall and Lower Neck:  Normal.
Bones: Status post left thoracotomy. 
"
To generate IMPRESSION section of this FINDINGS:
"
IMPRESSION:
1. Subsegmental pulmonary embolism in both lower lobes.
2. Mild pulmonary edema with small bilateral pleural effusions.
3. Indeterminate 18 mm focal groundglass nodule in the left lower lobe possibly due to post inflammatory change or atypical appearance of alveolar edema, however advise short-term followup CT chest in 3 months to ensure resolution.
4. Enlarged main pulmonary artery to 3.4 cm, which can be seen in setting of pulmonary arterial hypertension.
"
Do NOT output prompt text. Output plain text. Reduce normal findings, such as 'No acute abnormality', 'No significant mediastinal', 'There is no evidence of', 'No acute abnormality is seen'
"""