# MDC-Net: Multimodal Steel Defect Captioning

A multimodal computer vision project for detecting and generating textual descriptions of surface defects in steel using the MDC-Net architecture.

## Dataset

The project uses the NEU-DET surface defect dataset containing six defect categories:

- Crazing
- Scratches
- Rolled-in Scale
- Pitted Surface
- Patches
- Inclusion

## Current Pipeline

```text
NEU-DET
   ↓
XML Annotation Parsing
   ↓
Data Preprocessing
   ↓
70/20/10 Image-Level Split
   ↓
Data Augmentation
   ↓
MDC-Net Training
   ↓
Evaluation
   ↓
Defect Caption Generation