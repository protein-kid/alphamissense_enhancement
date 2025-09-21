The Model is machined learned with the original AlphaMissense and data about tissue specificity. The model is given an AlphaMissense score and the tissue specificity of the protein, and outputs its approximation of the pathogenity of the missense. This model was trained based on the data in clinvar_with_am.tsv and clinvar_with_am_tissue_matrix.tsv. This model can be used with call_model.py with rf_model.py and tissue_columns.py in the same folder.

This project uses AlphaMissense predictions.  
- Code: Apache License 2.0  
- Data: CC BY 4.0 (Cheng et al., 2023, Science, doi:10.1126/science.adg7492)

Other data is from ClinVar and The Human Protein Atlas.
- CLinVar data is from https://www.ncbi.nlm.nih.gov/clinvar/
- normal_tissue.tsv is available at https://v22.proteinatlas.org/about/download (The Human Protein Atlas)
