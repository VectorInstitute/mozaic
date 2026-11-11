# classifiers
One folder per classifier: model.py (Model, a BaseClassifier), its processor, requirements.txt, README;
pair.py: its T3 version (two clips per question, ../common/pairs.py).

    avhubert/    AV-HuBERT LARGE, T1, T2 and T3
    dfd_hr/      DFD-HR (CLIP ViT-L/14, MoE adapters, routing), T2
    havic/       HAVIC (ViT-B video and audio encoders, interaction), T1
    pe_av/       PE-AV Large (Meta's audio-video foundation encoder), T1, T2 and T3
